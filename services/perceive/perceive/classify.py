"""Pure pose-to-`PersonState` classification, plus a stateful hysteresis tracker.

Split the way `capture.gate`/`capture.motion` are split: geometry
(`classify_pose`, `centroid_of`) is pure, no I/O, no clock of its own, and
trivially unit-testable with hand-built `PoseResult`s; `StateTracker` is
where the statefulness and the "how many frames in a row" hysteresis live,
with `now` passed in explicitly by the caller rather than read from the
wall clock, matching `capture.gate.MotionGate.admit`.

PLAN.md section 6.2 is explicit that a blanket defeats pose models, so
`in_bed` is deliberately *not* "confidently detected lying-down landmarks".
It is "the caregiver-drawn bed zone, and not clearly standing or sitting
up" -- the same "absence of evidence for anything else" reasoning PLAN.md
prescribes.
"""

from __future__ import annotations

import json
import logging
import math
import random
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from perceive.backends import LANDMARK_NAMES, PoseResult
from perceive.zones import ZoneMap, ZoneName

SERVICE_NAME = "perceive"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, level: int = logging.INFO, **fields: object) -> None:
    """One structured JSON line to stdout, same convention as `perceive.main`
    (CLAUDE.md's "log one structured JSON line per event"). `classify.py` has
    no bus and no camera frame to accidentally log, so this is only ever
    used for the fall-drop/floor-suspect triggers `StateTracker.update`
    fires -- rare, worth a line each, and not the per-frame volume that
    keeps `perceive.main`'s own frame log at `DEBUG`."""
    logger.log(level, json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


PersonStateName = Literal["in_bed", "sitting_up", "standing", "walking", "on_floor", "absent"]


@dataclass(frozen=True)
class ClassifyThresholds:
    """Every tunable geometry threshold `classify_pose`/`StateTracker` use, in
    one place, with the reasoning for each value next to it -- issue #8's
    explicit requirement, so a future tuning pass has one file to read and
    one place to change, not magic numbers scattered through `if`
    statements.

    Two of these (`min_confidence`, `walk_displacement_threshold`) are also
    exposed as env vars (`PERCEIVE_MIN_CONFIDENCE`, `PERCEIVE_WALK_THRESHOLD`)
    because they are the two an operator is most likely to need to retune
    for a specific room/camera without a code change; the rest are geometry
    invariants of "what a lying/sitting/standing body looks like in a
    normalised frame" that should not need per-install tuning.
    """

    min_confidence: float = 0.5
    """Below this overall detection confidence, treat the frame as if no
    person was found at all (`absent`) rather than trust a shaky pose."""

    presence_confidence: float = 0.25
    """A detection scoring at least this much, but below `min_confidence`,
    proves someone is in the frame without being trustworthy enough to
    read a posture from. On the 2026-09-13 bedroom clips YOLO put a
    0.25-0.5 box on a person lying in bed far more often than a 0.5+ one,
    and the 0.5 cut-off turned every one of those into `absent`. Such a
    box in the bed zone counts as `in_bed` (PLAN.md section 6.2 already
    defines `in_bed` as "in the bed zone and not clearly up"); anywhere
    else it holds the tracker's current state instead of reporting
    `absent`. Set equal to `min_confidence` to disable.
    `PERCEIVE_PRESENCE_CONFIDENCE`."""

    lying_extent_ratio: float = 0.5
    """A body's vertical landmark spread at or below this fraction of its
    horizontal spread counts as "lying down" rather than "sitting/standing":
    an upright body is much taller than it is wide in a normalised frame,
    a horizontal one is the opposite."""

    floor_centroid_y: float = 0.6
    """A lying-down centroid at or below this height in the frame (y grows
    downward, so this means "in the lower part of the frame") counts as
    "on the floor" rather than, say, stretching across a low bed edge just
    out of the bed zone."""

    torso_upright_ratio: float = 0.6
    """`torso_vertical_ratio` (see `_geometry`) at or above this counts as
    an upright torso -- shoulders stacked over hips rather than beside
    them. This is what distinguishes `in_bed`/`on_floor` (torso mostly
    horizontal) from `sitting_up`/`standing` (torso mostly vertical)."""

    ankle_below_hip_margin: float = 0.05
    """Ankles at least this much further down the frame than the hips
    counts as "clearly below the hips" -- legs extended and weight-bearing,
    the geometric signature that separates `standing` from `sitting_up`
    (where bent legs put the ankles close to, or even level with, the
    hips)."""

    standing_vertical_extent: float = 0.35
    """A standing person's landmarks span at least this fraction of the
    frame height. Combined with the upright-torso and ankles-below-hips
    checks, this rules out a half-risen, still-bent posture being counted
    as fully `standing`."""

    sitting_thigh_ratio: float = 0.0
    """With the knees visible, a thigh that drops less than this fraction of
    the torso length (`_Geometry.thigh_drop_ratio`) is level or pointing at
    the camera: seated, even when the ankles hang well below the hips.
    Someone sitting on the bed edge with their feet on the floor otherwise
    reads as `standing`, the one posture that says "not in bed". On the
    2026-09-13/14 clips seated frames had a median of 0.11 (p90 0.42) and
    standing or walking frames a p10 of 0.67, so 0.55 separates them.

    `0.0`, the default, disables the rule. It turns seated people in the bed
    zone into bed occupants, so a bed zone that also covers a chair or the
    floor in front of the bed turns them into false `in bed` readings: on
    the same clips it added 33 false bed frames with the old hand-drawn
    zones, against a net gain with zones from `perceive.calibrate_bed`.
    Enable it together with that calibration. `PERCEIVE_SITTING_THIGH_RATIO`."""

    walk_displacement_threshold: float = 0.15
    """A `standing` person's centroid moving at least this much (normalised
    frame width) across the tracker's recent-frame window counts as
    `walking` rather than standing in place. `PERCEIVE_WALK_THRESHOLD`."""

    floor_top_y: float = 1.01
    """Outside the bed zone, a confident detection whose bounding box *top*
    sits at or below this height is `on_floor`, whatever its aspect ratio.
    The lying test above only fires for a body stretched across the frame;
    on the 2026-09-13 clips every floor event was a person sitting,
    kneeling or foreshortened towards the camera, with a box taller than
    wide, so it never fired. The head is what drops in a fall, and from a
    wall-mounted camera a low box top is the plainest sign of it. Like
    `floor_centroid_y`, the value depends on camera height. The default,
    above 1.0, disables the rule. `PERCEIVE_FLOOR_TOP_Y`."""

    absent_confirm_seconds: float = 0.0
    """How long the backend must see nobody before `absent` is reported;
    until then the last state is held. At 2 fps a one- or two-frame dropout
    is ordinary detector noise, not a person leaving, and reporting it as
    `absent` both loses the real state and, for `on_floor`, restarts the
    agent's floor timer. The agent already tolerates `absent` for
    `AGENT_ABSENT_LIMIT_SECONDS` (600 s), so a few seconds here cost no
    alert latency. `0.0` reports `absent` on the first empty frame.
    `PERCEIVE_ABSENT_CONFIRM_SECONDS`."""

    bed_vanish_hold: bool = False
    """Treat a person lost while last seen in the bed zone as `in_bed`
    (through normal hysteresis) rather than `absent`. Getting under a
    blanket is itself the moment the detector loses them, so the tracker
    often never sees a lying pose to confirm `in_bed` from and the bed hold
    (`StateTracker._holds_bed`) never engages. Never applied while the last
    confirmed state is `walking`. `PERCEIVE_BED_VANISH_HOLD`."""

    hold_floor: bool = False
    """Keep reporting `on_floor` while the backend sees nobody, as the bed
    hold does for `in_bed`. A person on the floor beside the bed is often
    partly hidden by it; dropping to `absent` there would reset the agent's
    floor escalation. Getting up is a transition the backend sees.
    `PERCEIVE_HOLD_FLOOR`."""

    floor_height_ratio: float = 0.6
    """`height_ratio` (a detection's box-top height above a self-calibrated
    ground line, divided by a standing person's expected height at that
    same ground row -- see `height_ratio_for` and `StateTracker`'s online
    fit) at or below this counts towards `on_floor`, once two consecutive
    detections agree outside the bed zone (`StateTracker._confirm_ratio_streak`).
    docs/FLOOR_DETECTION_HANDOFF.md section 4's simulation separated floor
    frames (ratio p50 0.30-0.48) from chair-sitting (p10 0.52+) at 0.6, and
    it catches the sitting/kneeling/foreshortened floor postures the
    lying-extent and `floor_top_y` rules above miss because a seated body's
    box is taller than wide. Needs an online or overridden ground line to
    have any effect at all; with neither, this rule cannot fire and
    behaviour is unchanged. `PERCEIVE_FLOOR_HEIGHT_RATIO`."""

    fall_window_seconds: float = 2.5
    """How long a recent upright detection (`height_ratio >= 0.8`) keeps
    counting as the reference point a fall is measured from. If a later
    detection's box top has risen by `fall_drop` since, within this many
    seconds of that reference, `StateTracker` treats it as a fall in
    progress and reports `on_floor` from a single detection -- bypassing
    both `min_confidence` and the two-consecutive check above. Section 9's
    replay of the 2026-09-13 clip 01 second floor event needed this: the
    confident frames near the ground were sparse, at 0.29-0.38 confidence.
    Short by design: a real fall completes in a couple of seconds, while an
    ordinary sit-down spreads the same box-top change over many more --
    long enough that the reference has already expired by the time the sit
    finishes. `PERCEIVE_FALL_WINDOW_SECONDS`."""

    fall_drop: float = 0.15
    """Minimum rise in bounding-box top (0.0-1.0, frame fraction) since the
    `fall_window_seconds`-old upright reference for `StateTracker` to call
    it a fall drop. `PERCEIVE_FALL_DROP`."""

    floor_suspect_seconds: float = 20.0
    """How long `StateTracker.floor_suspect` stays set after a fall drop is
    observed and the person is then lost entirely (no detection at all),
    before giving up on the suspicion. On its own this never reports
    `on_floor` and only prevents `absent` while it is set -- see
    `StateTracker.floor_suspect`'s docstring for what a caller is expected
    to do with it. `PERCEIVE_FLOOR_SUSPECT_SECONDS`."""


@dataclass(frozen=True)
class _Geometry:
    """Derived, frame-local measurements used to classify one `PoseResult`."""

    centroid_x: float
    centroid_y: float
    vertical_extent: float
    horizontal_extent: float
    torso_vertical_ratio: float
    """0.0 (torso lying flat/horizontal) to 1.0 (torso perfectly vertical):
    `abs(hip_y - shoulder_y) / (abs(dx) + abs(dy))` of the shoulder-to-hip
    vector. Named for what it measures, not for any particular unit."""
    ankle_below_hip: float
    """`ankle_y - hip_y`; positive means the ankles sit lower in the frame
    (further from the top, i.e. further down) than the hips."""
    thigh_drop_ratio: float | None = None
    """`(knee_y - hip_y) / torso_length`: about 0.8 for a standing thigh,
    near 0 for a seated one. `None` when a knee, a hip or the torso is
    missing, since there is then nothing to measure."""


def _mean(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def centroid_of(pose: PoseResult) -> tuple[float, float]:
    """The point handed to `zones.zone_for_point` to decide which zone `pose`
    occupies: the mean of every canonical landmark when the backend
    reported all of them, otherwise the bounding-box centre. The mean of
    *all* landmarks, not just the hips, so a person bent over or lying
    across a zone boundary lands on one stable point rather than jittering
    between zones frame to frame as individual joints cross the line. A
    partial set (YOLO omits keypoints it cannot place, see
    `perceive.backends.PoseResult`) has no such stability -- a head and
    two shoulders average to the head -- so the box centre is the better
    point then.
    """
    if len(pose.landmarks) == len(LANDMARK_NAMES):
        xs = [lm.x for lm in pose.landmarks.values()]
        ys = [lm.y for lm in pose.landmarks.values()]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    x_min, y_min, x_max, y_max = pose.bbox
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def zone_for_pose(zones: ZoneMap, pose: PoseResult) -> ZoneName:
    """Return the zone occupied by ``pose``.

    A landmark centroid counts as inside the bed only when the bounding-box
    centre is also inside it. If only the landmark centroid is in the bed,
    classify that point against the remaining zones in their usual order.
    """
    centroid_x, centroid_y = centroid_of(pose)
    zone = zones.zone_for_point(centroid_x, centroid_y)
    if zone != "bed":
        return zone

    x_min, y_min, x_max, y_max = pose.bbox
    bbox_x = (x_min + x_max) / 2.0
    bbox_y = (y_min + y_max) / 2.0
    if zones.zone_for_point(bbox_x, bbox_y) == "bed":
        return zone
    return zones.zone_for_point(centroid_x, centroid_y, include_bed=False)


def _geometry(pose: PoseResult) -> _Geometry:
    """Compute `_Geometry` from `pose`: body extents from its bounding box,
    torso and leg measurements from whichever landmarks the backend
    reported (missing ones read as "unknown", never as a position)."""
    centroid_x, centroid_y = centroid_of(pose)

    # Extents come from the box, not from the landmarks: for MediaPipe the
    # two are identical (its box *is* the landmark envelope), and for YOLO
    # the box is the detector's own estimate of the whole body while the
    # landmark set may be missing the very limbs that define the extent.
    x_min, y_min, x_max, y_max = pose.bbox

    left_shoulder = pose.landmarks.get("left_shoulder")
    right_shoulder = pose.landmarks.get("right_shoulder")
    left_hip = pose.landmarks.get("left_hip")
    right_hip = pose.landmarks.get("right_hip")
    left_ankle = pose.landmarks.get("left_ankle")
    right_ankle = pose.landmarks.get("right_ankle")

    shoulder_x = _mean(
        left_shoulder.x if left_shoulder else None, right_shoulder.x if right_shoulder else None
    )
    shoulder_y = _mean(
        left_shoulder.y if left_shoulder else None, right_shoulder.y if right_shoulder else None
    )
    hip_x = _mean(left_hip.x if left_hip else None, right_hip.x if right_hip else None)
    hip_y = _mean(left_hip.y if left_hip else None, right_hip.y if right_hip else None)
    ankle_y = _mean(left_ankle.y if left_ankle else None, right_ankle.y if right_ankle else None)

    if None not in (shoulder_x, shoulder_y, hip_x, hip_y):
        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        torso_vertical_ratio = abs(dy) / (abs(dx) + abs(dy) + 1e-6)
    else:
        # Shoulders or hips missing entirely: torso orientation is unknown.
        # Default to "not upright" (0.0), the conservative reading for a
        # bed-zone default of `in_bed` rather than accidentally granting
        # `standing`/`sitting_up` on missing data.
        torso_vertical_ratio = 0.0

    ankle_below_hip = (ankle_y - hip_y) if (ankle_y is not None and hip_y is not None) else 0.0

    left_knee = pose.landmarks.get("left_knee")
    right_knee = pose.landmarks.get("right_knee")
    knee_y = _mean(left_knee.y if left_knee else None, right_knee.y if right_knee else None)
    thigh_drop_ratio = None
    if knee_y is not None and None not in (shoulder_x, shoulder_y, hip_x, hip_y):
        torso_length = math.hypot(hip_x - shoulder_x, hip_y - shoulder_y)
        if torso_length > 0.02:
            thigh_drop_ratio = (knee_y - hip_y) / torso_length

    return _Geometry(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        vertical_extent=y_max - y_min,
        horizontal_extent=x_max - x_min,
        torso_vertical_ratio=torso_vertical_ratio,
        ankle_below_hip=ankle_below_hip,
        thigh_drop_ratio=thigh_drop_ratio,
    )


_GROUND_VISIBILITY = 0.3
"""Minimum landmark `visibility` for an ankle or knee to serve as the
ground point in `_ground_point`/`height_ratio_for` -- matches
docs/FLOOR_DETECTION_HANDOFF.md section 4's method."""


def _ground_point(
    pose: PoseResult, min_visibility: float = _GROUND_VISIBILITY
) -> tuple[float | None, bool]:
    """The frame row where `pose`'s feet (or, failing that, knees) meet the
    floor: the largest y (lowest in the frame, since y grows downward)
    among whichever ankle landmarks clear `min_visibility`, or the same
    over the knees if no ankle qualifies. Returns `(None, False)` if
    neither is visible enough -- section 4's "about a third of detected
    floor frames have no visible ankle" case, where the ratio is simply
    undefined for this frame.

    The second element is whether a genuine *ankle* (not a knee) was used,
    which matters for `StateTracker`'s ground-line calibration: only an
    ankle sighting on a confirmed standing/walking frame is trustworthy
    enough to teach the ground line from.
    """
    ankle_ys = [
        lm.y
        for name in ("left_ankle", "right_ankle")
        if (lm := pose.landmarks.get(name)) is not None and lm.visibility >= min_visibility
    ]
    if ankle_ys:
        return max(ankle_ys), True
    knee_ys = [
        lm.y
        for name in ("left_knee", "right_knee")
        if (lm := pose.landmarks.get(name)) is not None and lm.visibility >= min_visibility
    ]
    if knee_ys:
        return max(knee_ys), False
    return None, False


def _ground_point_in_box(pose: PoseResult, ground_y: float) -> bool:
    """Whether `ground_y` -- an ankle or knee row picked by `_ground_point`
    -- plausibly belongs to `pose`'s own bounding box: strictly below the
    box top (a ground point at or above the top is a keypoint the detector
    mislabelled, not a foot) and no more than 0.05 below the box bottom (a
    little slack for a foot the box itself clipped, no more). Shared by
    `height_ratio_for` and `StateTracker._collect_ground_sample` -- a bad
    keypoint must not poison a ratio *or* a calibration sample.
    """
    bbox_top, bbox_bottom = pose.bbox[1], pose.bbox[3]
    return bbox_top < ground_y <= bbox_bottom + 0.05


def height_ratio_for(
    pose: PoseResult,
    ground_line: tuple[float, float] | None,
    min_visibility: float = _GROUND_VISIBILITY,
) -> float | None:
    """How tall `pose` looks relative to a standing person at the same
    ground row (docs/FLOOR_DETECTION_HANDOFF.md section 4):
    `ground_y` from `_ground_point`, `height = ground_y - bbox_top`,
    `expected = a * ground_y + b` from `ground_line`,
    `height_ratio = height / expected`.

    Pure and stateless: `ground_line` is supplied by the caller (typically
    `StateTracker`'s self-calibrated or overridden line), so this function
    carries no history of its own and is directly unit-testable with a
    hand-built `PoseResult` and an injected line.

    Returns `None` -- "this rule does not apply to this frame" -- in every
    case below, all found from real bad-keypoint detections (ratios of
    -3.29 and 2.5 that a naive division let straight through):

    - No ground line yet, or no ankle/knee visible enough.
    - The ground point is not plausibly inside `pose.bbox`
      (`_ground_point_in_box`) -- a mislabelled keypoint, not a foot.
    - `height` (`ground_y - bbox_top`) is at or below 0.02: too small a
      box to say anything about height from.
    - `expected` is at or below 0.05: too close to the top of the frame
      for the fitted line to mean anything there.
    - The resulting ratio falls outside `[0.15, 1.6]`: nothing this
      system has seen, floor or standing, produces a ratio near zero,
      negative, or several times a standing person's own height --
      that is always a bad box or a bad ground point, not a real posture.
    """
    if ground_line is None:
        return None
    ground_y, _used_ankle = _ground_point(pose, min_visibility)
    if ground_y is None:
        return None
    if not _ground_point_in_box(pose, ground_y):
        return None
    height = ground_y - pose.bbox[1]
    if height <= 0.02:
        return None
    a, b = ground_line
    expected = a * ground_y + b
    if expected <= 0.05:
        return None
    ratio = height / expected
    if not (0.15 <= ratio <= 1.6):
        return None
    return ratio


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, `q` in `[0, 100]`. A few lines of
    plain Python rather than a numpy dependency: the ring buffer this feeds
    is at most a few hundred floats."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def fit_ground_line(
    samples: Sequence[tuple[float, float]],
    *,
    min_samples: int = 20,
    spread_floor: float = 0.05,
    max_pairs: int = 4000,
    rng: random.Random | None = None,
) -> tuple[float, float] | None:
    """Theil-Sen fit of `height = a * ground_y + b` over `(ground_y, height)`
    samples from confirmed standing/walking frames (docs/FLOOR_DETECTION_HANDOFF.md
    section 4/9): the median of every pairwise slope, then the median of
    `y - a * x` as the intercept -- robust to the odd bad frame in a way
    least-squares is not, and cheap enough to refit online every time a
    sample is added.

    Abstains (returns `None`) below `min_samples`: too few points to trust.
    Below `spread_floor` of `ground_y` spread (p90 - p10), the camera has
    not yet seen the person stand at different distances, so a slope
    cannot be estimated -- this falls back to a flat line (`a=0.0`) through
    the median height instead of a noisy one, per the handoff doc's
    "degenerate spread" case.

    The number of pairs grows quadratically with `len(samples)`, so beyond
    `max_pairs` a random subsample of pairs stands in to keep an online
    refit cheap; `rng` is injectable for deterministic tests.
    """
    if len(samples) < min_samples:
        return None
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    spread = _percentile(xs, 90) - _percentile(xs, 10)
    if spread < spread_floor:
        return 0.0, statistics.median(ys)

    n = len(samples)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        picker = rng if rng is not None else random.Random()
        pairs = picker.sample(pairs, max_pairs)

    slopes = [(ys[j] - ys[i]) / (xs[j] - xs[i]) for i, j in pairs if abs(xs[j] - xs[i]) > 1e-9]
    if not slopes:
        return 0.0, statistics.median(ys)

    a = statistics.median(slopes)
    intercepts = [y - a * x for x, y in zip(xs, ys, strict=True)]
    b = statistics.median(intercepts)
    return a, b


def load_ground_line(path: str) -> tuple[float, float] | None:
    """Read a `{"a": ..., "b": ...}` ground line back, or `None` on any
    problem reading it -- a missing or corrupt file means "no calibration
    yet", not a startup failure. `StateTracker.__post_init__` calls this
    once, so a container restart is not blind to a calibration a previous
    run already found. `PERCEIVE_GROUND_LINE_FILE`."""
    try:
        data = json.loads(Path(path).read_text())
        return float(data["a"]), float(data["b"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def save_ground_line(path: str, line: tuple[float, float]) -> None:
    """Persist `line` to `path` atomically: write a sibling temp file, then
    `Path.replace` it into place, so a concurrent reader (or a crash
    mid-write) never sees a half-written file. Creates parent directories
    as needed -- `PERCEIVE_GROUND_LINE_FILE`'s suggested default,
    `/app/data/perceive/ground_line.json`, does not exist until then."""
    a, b = line
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps({"a": a, "b": b}))
    tmp.replace(target)


def classify_pose(
    pose: PoseResult | None,
    zone: ZoneName,
    thresholds: ClassifyThresholds = ClassifyThresholds(),
) -> tuple[PersonStateName, float]:
    """Classify one frame's pose into a `PersonState.state`, given the zone
    its centroid falls in. Pure function: no history, so it cannot itself
    tell `standing` from `walking` (that needs recent-frame displacement,
    which is `StateTracker`'s job) or apply hysteresis. `zone` is computed
    by the caller via `zone_for_pose(zones, pose)`, since a
    zone lookup needs the caregiver's `ZoneMap` and this function has none.

    Evaluated in the order issue #8 lists the states:

    1. `absent` -- no detection, or confidence below `thresholds.min_confidence`,
       except a detection at or above `thresholds.presence_confidence` in
       the bed zone, which is `in_bed`: too shaky to read a posture from,
       but proof enough that somebody is lying where the bed is.
    2. `on_floor` -- lying (small vertical extent relative to horizontal),
       low in the frame, and not in the bed zone.
    3. `in_bed` -- in the bed zone and the torso is not upright (see the
       module docstring: this deliberately does not require confident limb
       detection, because a blanket makes that unreliable).
    4. `sitting_up` -- torso upright, and either ankles not clearly below
       the hips or visible knees with a level thigh.
    5. `standing` -- torso upright, ankles clearly below the hips, a large
       vertical extent, and no level thigh.

    Anything left over (torso not upright, not in the bed zone, not
    lying-and-low enough to be `on_floor`, e.g. bending over or a partial
    occlusion) falls through to `sitting_up` -- the conservative middle
    ground between "definitely fine" (`standing`) and "definitely down"
    (`on_floor`/`in_bed`), rather than silently under-reporting an
    ambiguous frame as no risk at all.
    """
    if pose is None:
        return "absent", 0.0
    if pose.confidence < thresholds.min_confidence:
        if pose.confidence >= thresholds.presence_confidence and zone == "bed":
            return "in_bed", pose.confidence
        return "absent", pose.confidence

    geometry = _geometry(pose)

    lying = geometry.vertical_extent <= thresholds.lying_extent_ratio * geometry.horizontal_extent
    low_in_frame = geometry.centroid_y >= thresholds.floor_centroid_y
    if lying and low_in_frame and zone != "bed":
        return "on_floor", pose.confidence
    if zone != "bed" and pose.bbox[1] >= thresholds.floor_top_y:
        return "on_floor", pose.confidence

    torso_upright = geometry.torso_vertical_ratio >= thresholds.torso_upright_ratio

    if zone == "bed" and not torso_upright:
        return "in_bed", pose.confidence

    if torso_upright:
        ankles_below_hips = geometry.ankle_below_hip >= thresholds.ankle_below_hip_margin
        tall_enough = geometry.vertical_extent >= thresholds.standing_vertical_extent
        thigh_level = (
            geometry.thigh_drop_ratio is not None
            and geometry.thigh_drop_ratio < thresholds.sitting_thigh_ratio
        )
        if ankles_below_hips and tall_enough and not thigh_level:
            return "standing", pose.confidence
        return "sitting_up", pose.confidence

    return "sitting_up", pose.confidence


@dataclass
class _PendingChange:
    """A candidate state change `StateTracker` is waiting to confirm."""

    state: PersonStateName
    confidence: float
    count: int


@dataclass
class StateTracker:
    """Adds hysteresis and `walking` detection on top of `classify_pose`.

    A single bad frame -- IR grain, a momentary occlusion -- must not flap
    the agent between session phases, so `update()` withholds a state
    change until `confirm_frames` consecutive detections agree, per issue
    #8. A frame with no detection at all neither counts towards nor resets
    that tally: it is missing evidence, not a disagreement, and a person
    the model only catches every other frame -- the normal reading of
    someone lying under a blanket -- must still be able to confirm
    `in_bed` rather than restart the count on every dropped frame.

    Two states bypass that hysteresis and report on the very frame they
    are first seen: `on_floor` and `absent`. HANDOFF.md rule 5 sends both
    straight to the `escalate_phone` strategy, and PLAN.md section 12 sets
    a 2 s latency target specifically for `on_floor`; waiting several
    frames to confirm a fall is exactly the kind of latency that target
    rules out. Both are also the two states where a false positive is
    merely an unnecessary caregiver alert, while a false negative (missing
    a real fall) is the failure this whole system exists to avoid -- an
    asymmetry hysteresis is the wrong tool for.

    `walking` is derived here, not in `classify_pose`, because it needs a
    short history of centroid positions that only the tracker keeps:
    `classify_pose` calls a moving person `standing`, and `update` upgrades
    consecutive `standing` classifications to `walking` once the centroid's
    horizontal displacement across its recent-frame window passes
    `thresholds.walk_displacement_threshold`. `walking` itself still goes
    through the normal hysteresis -- only `on_floor`/`absent` bypass it.

    Losing the person while they are in bed is held at `in_bed` rather than
    reported as `absent`; see `_holds_bed` for why that asymmetry exists.

    A detection between `thresholds.presence_confidence` and
    `thresholds.min_confidence` outside the bed zone is treated as an
    uninformative frame: someone is there, so it is not `absent`, but the
    pose is too shaky to say what they are doing, so nothing changes --
    neither the current state nor a pending confirmation count.

    docs/FLOOR_DETECTION_HANDOFF.md section 9's two floor-detection fixes
    also live here, layered on top of `classify_pose`'s own rules: a
    self-calibrated ground line (`_collect_ground_sample`/`fit_ground_line`,
    online Theil-Sen from confirmed standing/walking frames, or
    `ground_line` to override it) feeds `height_ratio_for`, which in turn
    feeds a two-consecutive-detection `on_floor` rule
    (`_confirm_ratio_streak`) and a single-frame fall-drop bypass
    (`_update_fall_anchor`/`_fall_drop_active`) for the low-confidence tail
    of a real fall. `floor_suspect` is the read-only signal the latter
    leaves behind when the person is then lost entirely, for a future
    vision check to act on -- see its own docstring.
    """

    IMMEDIATE_STATES: ClassVar[frozenset[str]] = frozenset({"on_floor", "absent"})

    thresholds: ClassifyThresholds = field(default_factory=ClassifyThresholds)
    confirm_frames: int = 3
    centroid_history_len: int = 5
    bed_hold_seconds: float = 0.0
    """How long to keep reporting `in_bed` after the pose backend stops
    seeing anyone, before giving up and reporting `absent`. `0.0`, the
    default, means "indefinitely". `PERCEIVE_BED_HOLD_SECONDS`."""

    ground_line: tuple[float, float] | None = None
    """Explicit `a, b` override for the ground line `height_ratio_for` uses
    (`PERCEIVE_GROUND_LINE="a,b"`). Set, this disables the online Theil-Sen
    fit entirely -- there is nothing left for it to learn -- which is also
    how tests pin a line without needing 20+ samples of setup."""

    ground_line_file: str | None = None
    """Where to persist the latest online-fitted ground line, and to load
    one from at startup (`load_ground_line`/`save_ground_line`). `None`,
    the default, disables persistence outright: nothing is read or
    written. Ignored when `ground_line` is set. `PERCEIVE_GROUND_LINE_FILE`,
    suggested default `/app/data/perceive/ground_line.json`."""

    _current: PersonStateName | None = field(default=None, init=False, repr=False)
    _last_confidence: float = field(default=0.0, init=False, repr=False)
    _last_zone: ZoneName = field(default="other", init=False, repr=False)
    _pending: _PendingChange | None = field(default=None, init=False, repr=False)
    _centroid_x_history: deque[float] = field(default_factory=deque, init=False, repr=False)
    _undetected_since: float | None = field(default=None, init=False, repr=False)
    _last_seen_zone: ZoneName = field(default="other", init=False, repr=False)
    _ground_samples: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=300), init=False, repr=False
    )
    _fitted_ground_line: tuple[float, float] | None = field(default=None, init=False, repr=False)
    _ratio_streak_count: int = field(default=0, init=False, repr=False)
    _ratio_streak_last_time: float | None = field(default=None, init=False, repr=False)
    _fall_anchor_top: float | None = field(default=None, init=False, repr=False)
    _fall_anchor_time: float | None = field(default=None, init=False, repr=False)
    _last_drop_detection: tuple[float, float, ZoneName] | None = field(
        default=None, init=False, repr=False
    )
    """`(confidence, ratio, zone)` of the most recent real detection whose
    `_fall_drop_active` was `True`, or `None`. Consumed (reset to `None`)
    the first time a loss-of-detection frame considers arming
    `floor_suspect` from it -- see `update()`."""
    _floor_suspect_since: float | None = field(default=None, init=False, repr=False)
    _last_height_ratio: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.confirm_frames = max(1, self.confirm_frames)
        self._centroid_x_history = deque(maxlen=self.centroid_history_len)
        if self.ground_line is None and self.ground_line_file:
            # A restart should not be blind to a calibration a previous run
            # already found; best-effort, see `load_ground_line`.
            self._fitted_ground_line = load_ground_line(self.ground_line_file)

    @property
    def floor_suspect(self) -> float | None:
        """`update()`-time timestamp a fall drop was observed just before
        the person was lost entirely, or `None` if there is no active
        suspicion. Bounded by `thresholds.floor_suspect_seconds` and
        cleared the moment a detection clearly resolves it (an upright
        `height_ratio >= 0.8`, or the bed zone) -- see the bookkeeping in
        `update()`.

        Read-only and side-effect free, so a future vision check (not
        built here) can poll this without `StateTracker` needing to know
        anything about who reads it. On its own this is never the reason
        `update()` reports `on_floor` -- only a live detection is.
        """
        return self._floor_suspect_since

    @property
    def last_height_ratio(self) -> float | None:
        """Most recent `height_ratio_for` reading from a pose that had one
        (an ankle or knee visible enough, and a ground line available), or
        `None` if there has never been one. Held across frames with no
        usable ratio the same way `last_seen_zone` is, so a caller like
        `perceive.floor_check.FloorCheckTrigger` can read "how low did we
        last see them" even on a frame where the ratio could not be
        computed. Read-only and side-effect free -- see `floor_suspect`'s
        docstring for the same reasoning."""
        return self._last_height_ratio

    @property
    def last_seen_zone(self) -> ZoneName:
        """The zone of the most recent frame with a detection, held across
        frames where nobody is currently detected -- the same value
        `_holds_bed`/`bed_vanish_hold` already read internally, exposed
        read-only for `perceive.floor_check`'s "last seen outside the bed"
        trigger."""
        return self._last_seen_zone

    def seconds_since_last_detection(self, now: float) -> float:
        """How long since the pose backend last returned a detection at
        all; `0.0` if it currently does. For
        `perceive.floor_check.FloorCheckTrigger`'s "lost, but recently seen
        outside the bed" trigger."""
        if self._undetected_since is None:
            return 0.0
        return now - self._undetected_since

    def confirm_floor(
        self,
        now: float,
        *,
        confidence: float = 1.0,
        source: str = "vision",  # noqa: ARG002
    ) -> tuple[PersonStateName, float] | None:
        """Force `on_floor` from an external confirmation -- the vision
        second opinion in `perceive.floor_check`, never called from
        `update()` itself. Publishes immediately, the same IMMEDIATE-state
        treatment `update()` gives a pose-confirmed `on_floor` (see the
        class docstring): there is no hysteresis to wait out here either.

        A no-op, returning `None`, if the tracker already reports
        `on_floor` -- the same "do not republish an unchanged immediate
        state" rule `update()` applies to a pose-confirmed one, and also
        what keeps this from ever needing to *clear* `on_floor`: it only
        ever pushes towards it. Clearing happens the ordinary way, through
        `update()` seeing the person upright again or back in bed.

        `source` is accepted for the caller's own structured log line
        (`perceive.main.run_once`); `StateTracker` does not persist it.
        `now` is accepted for parity with `update()`'s explicit-clock
        convention but not currently used.
        """
        if self._current == "on_floor":
            return None
        self._current = "on_floor"
        self._last_confidence = confidence
        self._pending = None
        return "on_floor", confidence

    def _effective_ground_line(self) -> tuple[float, float] | None:
        return self.ground_line if self.ground_line is not None else self._fitted_ground_line

    def _collect_ground_sample(self, pose: PoseResult, zone: ZoneName) -> None:
        """Feed one `(ground_y, height)` pair from a confirmed
        standing/walking, confident, ankle-visible frame into the online
        ground-line fit (docs/FLOOR_DETECTION_HANDOFF.md section 4/9).
        A no-op once `ground_line` is set explicitly: an override always
        wins, so there is nothing left to learn.

        Two rejections found by review, both because "standing" here means
        `classify_pose`'s geometry check, which never looks at the zone or
        the ankle's own plausibility:

        - `zone == "bed"`: someone standing *on* the mattress is upright by
          the same geometry a person standing on the floor is, but their
          ankle sits at the wrong row entirely -- fitting the ground line
          from it would teach it the height of the bed, not the floor.
        - The ankle is not plausibly inside `pose.bbox`
          (`_ground_point_in_box`, the same guard `height_ratio_for` uses):
          a mislabelled keypoint must not poison the calibration any more
          than it should poison a single ratio reading.
        """
        if self.ground_line is not None:
            return
        if zone == "bed":
            return
        ground_y, used_ankle = _ground_point(pose)
        if ground_y is None or not used_ankle:
            return
        if not _ground_point_in_box(pose, ground_y):
            return
        height = ground_y - pose.bbox[1]
        self._ground_samples.append((ground_y, height))
        fitted = fit_ground_line(list(self._ground_samples))
        if fitted is None:
            return
        self._fitted_ground_line = fitted
        if self.ground_line_file:
            save_ground_line(self.ground_line_file, fitted)

    def _update_fall_anchor(
        self,
        pose: PoseResult,
        ratio: float | None,
        zone: ZoneName,
        confidence: float,
        now: float,
    ) -> None:
        """Track the most recent upright (`height_ratio >= 0.8`) reference
        point a fall could be measured from. The anchor never survives a
        frame in the bed zone -- falling onto the bed is fine -- and it
        expires after `thresholds.fall_window_seconds` with no fresh
        upright frame to refresh it. That expiry is what keeps a slow,
        controlled sit-down (many seconds end to end) from ever reading as
        a fall drop the way a real fall (a couple of seconds) does: by the
        time a sit-down's box top has moved far enough, the last upright
        sighting is long stale.

        Also requires `confidence >= thresholds.min_confidence`: an anchor
        is a claim "this is what upright looks like here", and a shaky
        detection is exactly the kind of noisy box that should not get to
        make that claim -- review found ratios as wild as -3.29 and 2.5
        coming from bad keypoints, most of them on low-confidence frames.
        """
        if zone == "bed":
            self._fall_anchor_top = None
            self._fall_anchor_time = None
            return
        if ratio is not None and ratio >= 0.8 and confidence >= self.thresholds.min_confidence:
            self._fall_anchor_top = pose.bbox[1]
            self._fall_anchor_time = now
        elif (
            self._fall_anchor_time is not None
            and now - self._fall_anchor_time > self.thresholds.fall_window_seconds
        ):
            self._fall_anchor_top = None
            self._fall_anchor_time = None

    def _fall_drop_active(self, current_top: float, now: float, zone: ZoneName) -> bool:
        """Whether `current_top` has risen far enough below the live anchor,
        within its window, to call this a fall in progress. See
        `_update_fall_anchor` for how the anchor is maintained."""
        if (
            zone == "bed"
            or self._fall_anchor_top is None
            or self._fall_anchor_time is None
            or now - self._fall_anchor_time > self.thresholds.fall_window_seconds
        ):
            return False
        return (current_top - self._fall_anchor_top) >= self.thresholds.fall_drop

    def _confirm_ratio_streak(
        self,
        pose: PoseResult | None,
        ratio: float | None,
        zone: ZoneName,
        confidence: float,
        now: float,
    ) -> bool:
        """Fix 1's two-consecutive-detection confirmation for
        `thresholds.floor_height_ratio` (docs/FLOOR_DETECTION_HANDOFF.md
        section 9): a missed frame (`pose is None`) neither counts towards
        nor resets the streak, a real detection that no longer qualifies
        does reset it, and two qualifying detections more than 3 s apart
        do not count as consecutive -- long enough to bridge an ordinary
        2 fps dropout, short enough that a genuinely separate sighting has
        to start over.
        """
        if pose is None:
            return False
        qualifies = (
            zone != "bed"
            and confidence >= self.thresholds.min_confidence
            and ratio is not None
            and ratio <= self.thresholds.floor_height_ratio
        )
        if not qualifies:
            self._ratio_streak_count = 0
            self._ratio_streak_last_time = None
            return False
        if self._ratio_streak_last_time is not None and now - self._ratio_streak_last_time > 3.0:
            self._ratio_streak_count = 0
        self._ratio_streak_count += 1
        self._ratio_streak_last_time = now
        return self._ratio_streak_count >= 2

    def _holds_bed(self, now: float) -> bool:
        """Whether an undetected person should still be reported as `in_bed`.

        A blanket defeats pose models (PLAN.md section 6.2), so a person
        asleep under a duvet is not an edge case: it is the state the
        system spends most of every night in, and the backend seeing
        nothing there is the expected reading, not a fault. Without this,
        the first covered frame reports `absent`, and HANDOFF.md rule 5
        sends a sustained `absent` straight past every strategy to
        `escalate_phone`. The caregiver's phone would ring on a night when
        nothing happened, and a caregiver woken for nothing turns the
        system off within a week, which protects nobody.

        The safety net is that leaving the bed is a *transition* the
        backend does see. Nobody reaches the door without passing through
        `sitting_up` or `standing` in open view first, and both confirm
        normally, so a real bed exit still moves the state off `in_bed`
        before detection is lost. Only a person who vanishes without ever
        being seen upright is held wrongly.

        That residual case is exactly what a bed pressure sensor resolves,
        which is why PLAN.md defers one to v2, and what the perception
        bench in issue #11 has to measure. `bed_hold_seconds` bounds the
        hold for anyone who would rather have the false alarm; it is
        unbounded by default because on this hardware the nightly false
        alarm is the more certain harm.
        """
        if self._current != "in_bed" or self._undetected_since is None:
            return False
        if self.bed_hold_seconds <= 0.0:
            return True
        return (now - self._undetected_since) < self.bed_hold_seconds

    def update(
        self,
        pose: PoseResult | None,
        zone: ZoneName,
        now: float,  # noqa: ARG002 - accepted for parity with capture.gate's injected-clock convention
    ) -> tuple[PersonStateName, float] | None:
        """Classify one frame and return `(state, confidence)` if this update
        should be published, or `None` if hysteresis is still waiting on
        more agreeing frames (or nothing changed at all).

        `now` is accepted but not currently used for the confirm-frame
        count itself -- hysteresis here counts *frames*, not wall-clock
        time, since `perceive`'s frame rate already varies with
        `capture`'s motion gate -- kept for interface symmetry with the
        rest of the codebase's explicit-clock convention, and because a
        future time-windowed walk detector would need it.

        Fix 1/Fix 3 (docs/FLOOR_DETECTION_HANDOFF.md section 9) layer two
        more ways to reach `on_floor` on top of `classify_pose`'s own
        lying-extent/`floor_top_y` rules, both height-ratio driven and
        both no-ops with no ground line available:

        - Two consecutive confident detections at or below
          `thresholds.floor_height_ratio` outside the bed zone
          (`_confirm_ratio_streak`) -- catches the seated/kneeling/
          foreshortened floor postures a box-shape rule misses.
        - A fall drop (`_fall_drop_active`): a recent upright detection
          followed by the box top rising `thresholds.fall_drop` within
          `thresholds.fall_window_seconds`, reports `on_floor` from a
          single detection, bypassing both `min_confidence` and the
          two-consecutive check above -- the low-confidence tail of a real
          fall must not wait on either.
        """
        state, confidence = classify_pose(pose, zone, self.thresholds)

        if pose is not None:
            ratio = height_ratio_for(pose, self._effective_ground_line())
            if ratio is not None:
                self._last_height_ratio = ratio
            self._update_fall_anchor(pose, ratio, zone, pose.confidence, now)
            drop_active = self._fall_drop_active(pose.bbox[1], now, zone)
            self._last_drop_detection = (pose.confidence, ratio, zone) if drop_active else None

            if (
                drop_active
                and zone != "bed"
                and pose.confidence >= self.thresholds.presence_confidence
                and ratio is not None
                and ratio <= self.thresholds.floor_height_ratio
            ):
                state, confidence = "on_floor", pose.confidence
                self._floor_suspect_since = None
                _log(
                    "fall drop -> on_floor",
                    zone=zone,
                    confidence=round(pose.confidence, 3),
                    ratio=round(ratio, 3),
                )
            elif (
                self._confirm_ratio_streak(pose, ratio, zone, pose.confidence, now)
                and state != "on_floor"
            ):
                state, confidence = "on_floor", pose.confidence

            if zone == "bed" or (ratio is not None and ratio >= 0.8):
                # A clear resolution: either back in bed, or plainly
                # upright again. Either way, any open floor suspicion from
                # a previous drop no longer applies.
                self._floor_suspect_since = None

            if (
                self._current in ("standing", "walking")
                and pose.confidence >= self.thresholds.min_confidence
            ):
                self._collect_ground_sample(pose, zone)

        if (
            pose is not None
            and state == "absent"
            and pose.confidence >= self.thresholds.presence_confidence
        ):
            # Seen, but not readably. Evidence against `absent`, not for
            # anything else: hold what we have and wait for a better frame.
            # This is also what keeps `on_floor` reported while subsequent
            # low-confidence, still-low-ratio detections trickle in: `state`
            # here is `absent` only because confidence is shaky, so holding
            # `self._current` unchanged keeps `on_floor` exactly as it is.
            self._undetected_since = None
            return None

        if pose is None:
            if self._undetected_since is None:
                self._undetected_since = now

            if (
                self._floor_suspect_since is not None
                and now - self._floor_suspect_since > self.thresholds.floor_suspect_seconds
            ):
                self._floor_suspect_since = None
            if self._floor_suspect_since is None and self._last_drop_detection is not None:
                last_confidence, last_ratio, last_zone = self._last_drop_detection
                # A fall drop was seen just before the person vanished
                # entirely -- but only arm on a detection review found
                # trustworthy enough to hang 20 s of suppressed `absent`
                # on: confidently seen (not a noisy box), visibly lowered
                # (ratio <= 0.75, not just a shifted box near ratio 1), and
                # not last seen heading out the door, where a bend-over
                # followed by leaving the room would otherwise arm it.
                # Consumed either way -- one chance per drop, not re-tried
                # on every subsequent empty frame.
                if (
                    last_confidence >= self.thresholds.presence_confidence
                    and last_ratio is not None
                    and last_ratio <= 0.75
                    and last_zone != "door"
                    and last_zone != "bed"
                ):
                    self._floor_suspect_since = now
                    _log(
                        "fall drop then lost -> floor suspect",
                        last_seen_zone=self._last_seen_zone,
                        confidence=round(last_confidence, 3),
                        ratio=round(last_ratio, 3),
                    )
                self._last_drop_detection = None
            if self._floor_suspect_since is not None:
                # Never on its own a reason to report `on_floor` (see
                # `floor_suspect`'s docstring) -- just a reason not to
                # report `absent` either, until a detection or a timeout
                # resolves it.
                return None

            if self._holds_bed(now):
                # Keep the last good confidence and the bed zone: the person
                # has not moved, we simply cannot see them under the covers.
                return None
            if self.thresholds.hold_floor and self._current == "on_floor":
                return None
            if (
                self.thresholds.bed_vanish_hold
                and self._last_seen_zone == "bed"
                # Someone walking through the bed zone is more likely leaving
                # the frame than getting under the covers.
                and self._current not in ("in_bed", "on_floor", "walking")
            ):
                state, zone = "in_bed", "bed"
            elif now - self._undetected_since < self.thresholds.absent_confirm_seconds:
                return None
        else:
            self._undetected_since = None
            self._last_seen_zone = zone

        self._last_confidence = confidence
        self._last_zone = zone

        if state == "standing" and pose is not None:
            centroid_x, _ = centroid_of(pose)
            self._centroid_x_history.append(centroid_x)
            if self._is_walking():
                state = "walking"
        else:
            self._centroid_x_history.clear()

        if state in self.IMMEDIATE_STATES:
            if pose is not None:
                # A real detection that contradicts the pending state
                # restarts its count; a missing one does not (see the
                # class docstring).
                self._pending = None
            if state == self._current:
                return None
            self._current = state
            return state, confidence

        if state == self._current:
            self._pending = None
            return None

        if self._pending is not None and self._pending.state == state:
            self._pending.count += 1
            self._pending.confidence = confidence
        else:
            self._pending = _PendingChange(state=state, confidence=confidence, count=1)

        if self._pending.count >= self.confirm_frames:
            self._current = state
            confirmed_confidence = self._pending.confidence
            self._pending = None
            return state, confirmed_confidence

        return None

    def _is_walking(self) -> bool:
        if len(self._centroid_x_history) < 2:
            return False
        displacement = max(self._centroid_x_history) - min(self._centroid_x_history)
        return displacement >= self.thresholds.walk_displacement_threshold

    def snapshot(self) -> tuple[PersonStateName, float, ZoneName] | None:
        """Return `(state, confidence, zone)` last classified, for the
        heartbeat path: `perceive.main` publishes this at most once every
        `PERCEIVE_HEARTBEAT_SECONDS` even when nothing has changed, so a
        silent `perceive` cannot be mistaken for a calm night (HANDOFF.md
        rule 4). Returns `None` before the first frame has been classified
        -- there is nothing yet to repeat.
        """
        if self._current is None:
            return None
        return self._current, self._last_confidence, self._last_zone
