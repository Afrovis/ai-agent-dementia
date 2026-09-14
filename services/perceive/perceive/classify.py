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

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from perceive.backends import LANDMARK_NAMES, PoseResult
from perceive.zones import ZoneName

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

    walk_displacement_threshold: float = 0.15
    """A `standing` person's centroid moving at least this much (normalised
    frame width) across the tracker's recent-frame window counts as
    `walking` rather than standing in place. `PERCEIVE_WALK_THRESHOLD`."""


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

    return _Geometry(
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        vertical_extent=y_max - y_min,
        horizontal_extent=x_max - x_min,
        torso_vertical_ratio=torso_vertical_ratio,
        ankle_below_hip=ankle_below_hip,
    )


def classify_pose(
    pose: PoseResult | None,
    zone: ZoneName,
    thresholds: ClassifyThresholds = ClassifyThresholds(),
) -> tuple[PersonStateName, float]:
    """Classify one frame's pose into a `PersonState.state`, given the zone
    its centroid falls in. Pure function: no history, so it cannot itself
    tell `standing` from `walking` (that needs recent-frame displacement,
    which is `StateTracker`'s job) or apply hysteresis. `zone` is computed
    by the caller via `zones.zone_for_point(*centroid_of(pose))`, since a
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
    4. `sitting_up` -- torso upright, ankles not clearly below the hips.
    5. `standing` -- torso upright, ankles clearly below the hips, and a
       large vertical extent.

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

    torso_upright = geometry.torso_vertical_ratio >= thresholds.torso_upright_ratio

    if zone == "bed" and not torso_upright:
        return "in_bed", pose.confidence

    if torso_upright:
        ankles_below_hips = geometry.ankle_below_hip >= thresholds.ankle_below_hip_margin
        tall_enough = geometry.vertical_extent >= thresholds.standing_vertical_extent
        if ankles_below_hips and tall_enough:
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
    """

    IMMEDIATE_STATES: ClassVar[frozenset[str]] = frozenset({"on_floor", "absent"})

    thresholds: ClassifyThresholds = field(default_factory=ClassifyThresholds)
    confirm_frames: int = 3
    centroid_history_len: int = 5
    bed_hold_seconds: float = 0.0
    """How long to keep reporting `in_bed` after the pose backend stops
    seeing anyone, before giving up and reporting `absent`. `0.0`, the
    default, means "indefinitely". `PERCEIVE_BED_HOLD_SECONDS`."""

    _current: PersonStateName | None = field(default=None, init=False, repr=False)
    _last_confidence: float = field(default=0.0, init=False, repr=False)
    _last_zone: ZoneName = field(default="other", init=False, repr=False)
    _pending: _PendingChange | None = field(default=None, init=False, repr=False)
    _centroid_x_history: deque[float] = field(default_factory=deque, init=False, repr=False)
    _undetected_since: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.confirm_frames = max(1, self.confirm_frames)
        self._centroid_x_history = deque(maxlen=self.centroid_history_len)

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
        """
        state, confidence = classify_pose(pose, zone, self.thresholds)

        if (
            pose is not None
            and state == "absent"
            and pose.confidence >= self.thresholds.presence_confidence
        ):
            # Seen, but not readably. Evidence against `absent`, not for
            # anything else: hold what we have and wait for a better frame.
            self._undetected_since = None
            return None

        if pose is None:
            if self._undetected_since is None:
                self._undetected_since = now
            if self._holds_bed(now):
                # Keep the last good confidence and the bed zone: the person
                # has not moved, we simply cannot see them under the covers.
                return None
        else:
            self._undetected_since = None

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
