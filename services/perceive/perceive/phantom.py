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
suppressed. There is no cross-frame state, so a sleeping person's track can
never be mistaken for a phantom by anything that happens elsewhere in the
frame.
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

    Stateless per call -- `select` never remembers anything about previous
    frames -- so nothing here can accumulate the kind of wrong, permanent
    judgement about one specific track the old runtime-learning design
    could reach. An empty list (the default when `PERCEIVE_PHANTOMS_FILE`
    is unset, missing, or malformed) makes `select` behave exactly like the
    old "take the most confident candidate" rule.
    """

    def __init__(self, boxes: list[Box], *, max_confidence: float = DEFAULT_MAX_CONFIDENCE) -> None:
        self._boxes = boxes
        self._max_confidence = max_confidence

    @property
    def boxes(self) -> list[Box]:
        return list(self._boxes)

    def is_excluded(self, candidate: Candidate) -> bool:
        """A candidate is excluded only if it sits almost exactly on a known
        phantom box *and* is not confident enough to override that."""
        if candidate.confidence >= self._max_confidence:
            return False
        return any(iou(candidate.box, known) >= MATCH_IOU for known in self._boxes)

    def select(self, candidates: list[Candidate]) -> Candidate | None:
        """Return the highest-confidence candidate that isn't excluded, or
        `None` if every candidate in this frame is."""
        live = [c for c in candidates if not self.is_excluded(c)]
        if not live:
            return None
        return max(live, key=lambda c: c.confidence)

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
