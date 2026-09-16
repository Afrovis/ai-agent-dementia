"""Known-phantom filter for `YoloPoseBackend` (2026-09-14 bedroom evidence,
redesigned 2026-09-14 after two findings killed the first, runtime-learning
design).

A fixed, non-person object in frame -- a lamp, a headboard corner, a fold in
the bedding -- can score high enough that YOLO reports it as "person" with
an unmoving box, whether or not anyone is in the room at all. Picking "the
most confident detection" (the old single-person selection in
`backends.YoloPoseBackend.detect`) then hands that phantom to
`perceive.classify` as if it were the person -- exactly backwards from what
`perceive` exists to report (HANDOFF.md rule 4).

The first version of this filter tried to learn phantoms live, from
same-frame co-occurrence between a static box and an independently moving
one. Two findings killed that design:

1. On the real bedroom clips, YOLO11s at 640 never actually reports the
   phantom box in the same frame as the real person -- only in frames
   where the person isn't detected at all (arriving, or having left).
   Same-frame co-occurrence, the trigger the old design needed, simply
   never fires; it changed nothing on the offline eval.
2. Worse, it was unsafe in the case it *did* fire: a sleeping person's box
   is also static for long stretches, and any unrelated moving candidate
   elsewhere in frame (a caregiver checking in) would permanently flag the
   sleeper's own track as a phantom, turning `in_bed` into `absent` and
   raising a false wandering alert.

This module now does the opposite of learning at runtime: `KnownPhantoms`
holds a short, explicit, human-reviewed list of phantom boxes, calibrated
ahead of time (see `perceive.calibrate_phantoms`) while the room is known to
be empty, and loaded once at startup the same way `perceive.zones.load_zones`
loads `zones.yaml`. A candidate is excluded only if it overlaps a known
phantom box closely (`IoU >= MATCH_IOU`) *and* its own confidence is below
`PERCEIVE_PHANTOM_MAX_CONFIDENCE` -- a real person who happens to walk
through that exact box, or a strong, confident detection there, is never
suppressed. There is no cross-frame state driving exclusion, so a sleeping
person's track can never be mistaken for a phantom by anything that happens
elsewhere in the frame.

2026-09-15 evidence (a bedroom clip, `yolo26l-pose` backend) found the
"trust a strong score" override itself has a failure mode: a calibrated
phantom (a painting) can score consistently *above*
`PERCEIVE_PHANTOM_MAX_CONFIDENCE` for a run of frames, while the real
person mid-fall -- an inherently harder pose to classify -- scores lower in
those same frames, present as a second candidate but narrowly losing the
argmax. `KnownPhantoms.select` now carries one small, intentionally scoped
piece of cross-call state to catch exactly this: the last confirmed
(non-phantom) box and how long ago it was confirmed. It only ever changes
the outcome when the raw top pick itself matches a *known, pre-calibrated*
phantom box and another candidate in the same frame sits close to that
recent track -- never a general "prefer whatever's near the last
detection" bias, which would risk suppressing a genuinely new detection
elsewhere in frame. See `KnownPhantoms.select` and
`_rescue_from_confident_phantom`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

SERVICE_NAME = "perceive"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

Box = tuple[float, float, float, float]
"""Normalised `(x_min, y_min, x_max, y_max)`, same convention as `PoseResult.bbox`."""

MATCH_IOU = 0.85
"""A candidate counts as "the same object" as a known phantom box only when
they overlap by at least this much -- high on purpose, so this only ever
excludes a detection sitting almost exactly where a calibrated phantom is,
never merely "a person-sized box roughly nearby"."""

DEFAULT_MAX_CONFIDENCE = 0.7
"""Default `PERCEIVE_PHANTOM_MAX_CONFIDENCE`: a candidate overlapping a known
phantom box is only excluded below this confidence. A strong detection in
that exact box is trusted over the calibration -- the filter is a guard
against a weak, ambiguous score, not a blanket ban on the location."""

MAX_TRACK_GAP_CALLS = 15
"""How many `select()` calls a previously confirmed (non-phantom) detection
stays eligible to rescue a real person from a *high-confidence* phantom
override (see `KnownPhantoms.select`). `perceive` calls `detect`/`select`
once per frame `capture` republishes, at `CAPTURE_FPS` (2fps while the room
has motion, dropping to `CAPTURE_IDLE_FPS`, 0.5fps by default, once it has
been still for `CAPTURE_STATIC_SECONDS` -- see CLAUDE.md). 15 calls is
7.5-30s of gap at those rates: generous enough to bridge a stretch where a
falling person's own confidence dips call after call (the validated
2026-09-15 case: 12 consecutive frames losing to a painting), short enough
that a track from a much earlier, unrelated presence in the room cannot
reach across a real absence-then-return."""

TRACK_PROXIMITY_RADIUS = 0.3
"""How close, in normalised centre distance, another surviving candidate
must be to the last confirmed box to count as "the same person, still
moving" rather than "some other object elsewhere in frame" (see
`KnownPhantoms.select`). The 2026-09-15 bedroom evidence measured a real
person's own frame-to-frame centre drift at ~0.10 and the distance from an
unrelated calibrated phantom (a painting) to that same track at ~0.49; 0.3
sits with wide margin on both sides of that gap, so it accepts a genuinely
fast movement (e.g. the fall itself) while still rejecting a static object
clear across the room."""


@dataclass
class Candidate:
    """One detection from one `detect()` call, as seen by the phantom filter.

    Deliberately smaller than `backends.PoseResult`: the filter never looks
    at landmarks, only the box and the overall confidence used to decide
    exclusion and to select among the rest. `index` is opaque to this
    module -- the caller's own handle (e.g. a position in the raw model
    output) for mapping a selected `Candidate` back to whatever else it
    needs, such as keypoints.
    """

    box: Box
    confidence: float
    index: int = 0


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two normalised boxes, 0.0 if they don't overlap."""
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b
    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
    area_b = max(0.0, bx_max - bx_min) * max(0.0, by_max - by_min)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def center(box: Box) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = box
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def center_distance(a: Box, b: Box) -> float:
    """Euclidean distance between the normalised centres of two boxes, used
    by `KnownPhantoms.select`'s track-continuity check -- proximity of the
    box as a whole, not overlap, since a moving person's box legitimately
    shifts between calls."""
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _log_fallback(reason: str, path: str) -> None:
    logger.warning(
        json.dumps(
            {
                "service": SERVICE_NAME,
                "message": "phantoms config unavailable, phantom filter is inactive",
                "reason": reason,
                "path": path,
            }
        )
    )


class KnownPhantoms:
    """A short list of calibrated phantom boxes, and the exclusion rule that
    uses them.

    Almost stateless per call: `select` still never remembers *rejected*
    candidates or uses anything from outside its own known-phantom list to
    exclude a box, so nothing here can accumulate the kind of wrong,
    permanent judgement about one specific track the old runtime-learning
    design could reach. An empty list (the default when
    `PERCEIVE_PHANTOMS_FILE` is unset, missing, or malformed) makes `select`
    behave exactly like the old "take the most confident candidate" rule.

    It does keep one small piece of cross-call state: the last *confirmed*
    (selected, not matching a known phantom box) detection's box, and how
    many calls have passed since then. That exists for exactly one narrow
    purpose -- see `select` -- and is instance-scoped, living as long as the
    backend that owns this `KnownPhantoms` does (one instance per running
    `perceive`, the same lifetime the calibrated box list already has).
    """

    def __init__(self, boxes: list[Box], *, max_confidence: float = DEFAULT_MAX_CONFIDENCE) -> None:
        self._boxes = boxes
        self._max_confidence = max_confidence
        self._last_confirmed_box: Box | None = None
        self._calls_since_confirmed: int = 0

    @property
    def boxes(self) -> list[Box]:
        return list(self._boxes)

    def matches_known_phantom_box(self, box: Box) -> bool:
        """Whether `box` sits almost exactly on a known phantom box
        (`IoU >= MATCH_IOU`), independent of any confidence -- the geometric
        half of `is_excluded`, exposed on its own because `select`'s
        track-continuity rescue (below) needs to ask this about the raw
        top-confidence pick even when its confidence is too high for
        `is_excluded` to have dropped it."""
        return any(iou(box, known) >= MATCH_IOU for known in self._boxes)

    def is_excluded(self, candidate: Candidate) -> bool:
        """A candidate is excluded only if it sits almost exactly on a known
        phantom box *and* is not confident enough to override that."""
        if candidate.confidence >= self._max_confidence:
            return False
        return self.matches_known_phantom_box(candidate.box)

    def _rescue_from_confident_phantom(
        self, top: Candidate, live: list[Candidate]
    ) -> Candidate | None:
        """If `top` only survived because its confidence cleared
        `max_confidence` while still sitting on a known phantom box (2026-09-15
        bedroom evidence: a calibrated painting box scoring 0.75-0.85, just
        above the 0.7 default, while the real person mid-fall scored 0.53-0.61
        in the same frames), prefer a nearby recently-confirmed real
        detection instead, when one is available.

        Deliberately narrow: this only ever fires when `top` itself matches a
        *pre-calibrated* phantom box. It is not a general "prefer whatever is
        near the last track" bias -- a broad version of that would risk
        suppressing a genuinely new detection elsewhere in frame, which is
        dangerous in a fall-detection app. Returns `None` (meaning "no
        rescue, use `top` as normal") unless all three conditions hold: a
        recent confirmed track exists (`MAX_TRACK_GAP_CALLS`), `top` matches
        a known phantom box, and some *other* surviving candidate's box
        centre is within `TRACK_PROXIMITY_RADIUS` of that track. Among any
        such candidates, the closest one wins.
        """
        if self._last_confirmed_box is None:
            return None
        if self._calls_since_confirmed > MAX_TRACK_GAP_CALLS:
            return None
        if not self.matches_known_phantom_box(top.box):
            return None

        nearby = [
            c
            for c in live
            if c is not top
            and center_distance(c.box, self._last_confirmed_box) <= TRACK_PROXIMITY_RADIUS
        ]
        if not nearby:
            return None
        return min(nearby, key=lambda c: center_distance(c.box, self._last_confirmed_box))

    def select(self, candidates: list[Candidate]) -> Candidate | None:
        """Return the highest-confidence candidate that isn't excluded, or
        `None` if every candidate in this frame is.

        Ordinarily this is a plain argmax over the surviving candidates,
        exactly as before. The one exception is `_rescue_from_confident_phantom`:
        when the argmax pick itself sits on a known phantom box (only
        possible when its confidence is too high for `is_excluded` to have
        dropped it) and a recently confirmed track nearby suggests the real
        person is one of the other candidates in this same frame, that
        nearby candidate is selected instead.

        Either way, the selected candidate's box becomes the new "last
        confirmed" track *unless* it is itself the phantom box (i.e. the
        rescue didn't find anything to rescue to) -- a phantom-box selection
        must never seed or extend a track, or it could go on to wrongly
        rescue a future phantom pick of its own.
        """
        live = [c for c in candidates if not self.is_excluded(c)]
        if not live:
            self._calls_since_confirmed += 1
            return None

        top = max(live, key=lambda c: c.confidence)
        rescued = self._rescue_from_confident_phantom(top, live)
        selected = rescued if rescued is not None else top

        if self.matches_known_phantom_box(selected.box):
            self._calls_since_confirmed += 1
        else:
            self._last_confirmed_box = selected.box
            self._calls_since_confirmed = 0
        return selected

    @classmethod
    def load(
        cls,
        path: str | Path | None,
        *,
        max_confidence: float = DEFAULT_MAX_CONFIDENCE,
    ) -> KnownPhantoms:
        """Load calibrated phantom boxes from `path` (YAML, written by
        `perceive.calibrate_phantoms`), or return an inactive (empty)
        `KnownPhantoms` -- never raises. `path` is `None` or empty for
        "phantom filtering off" (the default, matching
        `PERCEIVE_PHANTOMS_FILE` unset); a configured path that is missing,
        unreadable, or malformed logs one warning and also degrades to
        inactive, mirroring `perceive.zones.load_zones`'s fail-quiet
        behaviour -- a bad phantoms file must never crash `perceive`, only
        leave it filtering nothing.
        """
        if not path:
            return cls([], max_confidence=max_confidence)

        candidate_path = Path(path)
        if not candidate_path.exists():
            _log_fallback("file does not exist", str(candidate_path))
            return cls([], max_confidence=max_confidence)

        try:
            with candidate_path.open() as handle:
                raw = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            _log_fallback(f"failed to parse {candidate_path}: {exc}", str(candidate_path))
            return cls([], max_confidence=max_confidence)

        entries = raw.get("phantoms") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            _log_fallback(
                f"{candidate_path} did not contain a 'phantoms' list", str(candidate_path)
            )
            return cls([], max_confidence=max_confidence)

        boxes: list[Box] = []
        for entry in entries:
            box = entry.get("box") if isinstance(entry, dict) else entry
            try:
                x_min, y_min, x_max, y_max = (float(v) for v in box)
            except (TypeError, ValueError):
                _log_fallback(
                    f"malformed phantom box in {candidate_path}: {entry!r}", str(candidate_path)
                )
                continue
            boxes.append((x_min, y_min, x_max, y_max))

        if not boxes:
            _log_fallback(f"{candidate_path} defined no usable phantom boxes", str(candidate_path))

        return cls(boxes, max_confidence=max_confidence)


def cluster_static_boxes(
    boxes_by_frame: list[list[Box]],
    *,
    min_frame_fraction: float = 0.3,
    max_center_drift: float = 0.02,
) -> list[Box]:
    """Cluster the boxes `perceive.calibrate_phantoms` observed across a run
    of frames known to be an empty room, and return the median box of every
    cluster that is both frequent and static enough to trust as a phantom.

    `boxes_by_frame` has one list per frame, in order, and each inner list
    contains every candidate in that frame. `min_frame_fraction` is measured
    against the total number of frames, not just frames with a detection.

    Clustering is temporal and greedy, not spatial k-means: each box joins
    the first existing cluster whose first (defining) box it overlaps by
    `MATCH_IOU` or more, else starts a new cluster. That is enough for this
    use -- calibration is run against a handful of genuinely static objects,
    not a crowded scene -- and keeps this pure and dependency-free like the
    rest of this module.

    A cluster survives only if it was observed in at least
    `min_frame_fraction` of all frames *and* every box in it stayed within
    `max_center_drift` of the cluster's first box -- the same staleness bar
    a real phantom needs to clear, so a moving person passing through
    roughly the same area a few times is never mistaken for one, and a
    box seen only rarely (a stray misdetection) is dropped rather than
    immortalised as a phantom.
    """
    total = len(boxes_by_frame)
    if total == 0:
        return []

    clusters: list[list[tuple[int, Box]]] = []
    for frame_index, frame_boxes in enumerate(boxes_by_frame):
        for box in frame_boxes:
            matched = False
            for cluster in clusters:
                if iou(box, cluster[0][1]) >= MATCH_IOU:
                    cluster.append((frame_index, box))
                    matched = True
                    break
            if not matched:
                clusters.append([(frame_index, box)])

    results: list[Box] = []
    for cluster in clusters:
        frame_count = len({frame_index for frame_index, _box in cluster})
        if frame_count / total < min_frame_fraction:
            continue
        cluster_boxes = [box for _frame_index, box in cluster]
        first_center = center(cluster_boxes[0])
        drift = max(
            ((cx - first_center[0]) ** 2 + (cy - first_center[1]) ** 2) ** 0.5
            for cx, cy in (center(box) for box in cluster_boxes)
        )
        if drift > max_center_drift:
            continue
        results.append(_median_box(cluster_boxes))
    return results


def _median_box(boxes: list[Box]) -> Box:
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        if n % 2 == 1:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    return (
        _median([box[0] for box in boxes]),
        _median([box[1] for box in boxes]),
        _median([box[2] for box in boxes]),
        _median([box[3] for box in boxes]),
    )


def known_phantoms_from_env(env: dict[str, str] | None = None) -> KnownPhantoms:
    """Build the `KnownPhantoms` described by `PERCEIVE_PHANTOMS_FILE` and
    `PERCEIVE_PHANTOM_MAX_CONFIDENCE`, the same env-then-default precedence
    every other `perceive` config uses (HANDOFF.md section 4). The single
    place `YoloPoseBackend` and `perceive.main` both go through so the two
    env vars are read in exactly one spot."""
    env = os.environ if env is None else env
    path = env.get("PERCEIVE_PHANTOMS_FILE", "").strip() or None
    max_confidence = float(env.get("PERCEIVE_PHANTOM_MAX_CONFIDENCE", str(DEFAULT_MAX_CONFIDENCE)))
    return KnownPhantoms.load(path, max_confidence=max_confidence)
