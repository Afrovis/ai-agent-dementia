"""Tests for `perceive.classify`: pure geometry (`classify_pose`) and the
stateful `StateTracker` hysteresis/walking logic. No Redis, no camera, no
model weights -- every `PoseResult` here is hand-built.
"""

from perceive.backends import Landmark, PoseResult
from perceive.classify import ClassifyThresholds, StateTracker, classify_pose

THRESHOLDS = ClassifyThresholds()


def _pose(points: dict[str, tuple[float, float]], confidence: float = 0.9) -> PoseResult:
    landmarks = {name: Landmark(x=x, y=y, visibility=0.9) for name, (x, y) in points.items()}
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return PoseResult(landmarks=landmarks, bbox=bbox, confidence=confidence)


def in_bed_pose() -> PoseResult:
    """Lying flat, torso horizontal, inside the [0.0, 0.4] x [0.2, 0.8] bed zone."""
    return _pose(
        {
            "nose": (0.05, 0.50),
            "left_shoulder": (0.08, 0.48),
            "right_shoulder": (0.08, 0.52),
            "left_hip": (0.30, 0.50),
            "right_hip": (0.30, 0.54),
            "left_knee": (0.35, 0.50),
            "right_knee": (0.35, 0.54),
            "left_ankle": (0.38, 0.50),
            "right_ankle": (0.38, 0.54),
        }
    )


def sitting_up_pose() -> PoseResult:
    """Upright torso, bent legs -- ankles not clearly below the hips."""
    return _pose(
        {
            "nose": (0.50, 0.30),
            "left_shoulder": (0.48, 0.35),
            "right_shoulder": (0.52, 0.35),
            "left_hip": (0.49, 0.55),
            "right_hip": (0.51, 0.55),
            "left_knee": (0.49, 0.60),
            "right_knee": (0.51, 0.60),
            "left_ankle": (0.49, 0.58),
            "right_ankle": (0.51, 0.58),
        }
    )


def standing_pose(centroid_x: float = 0.50) -> PoseResult:
    """Upright torso, ankles well below the hips, large vertical extent."""
    offset = centroid_x - 0.50
    return _pose(
        {
            "nose": (0.50 + offset, 0.10),
            "left_shoulder": (0.48 + offset, 0.20),
            "right_shoulder": (0.52 + offset, 0.20),
            "left_hip": (0.49 + offset, 0.50),
            "right_hip": (0.51 + offset, 0.50),
            "left_knee": (0.49 + offset, 0.75),
            "right_knee": (0.51 + offset, 0.75),
            "left_ankle": (0.49 + offset, 0.95),
            "right_ankle": (0.51 + offset, 0.95),
        }
    )


def on_floor_pose() -> PoseResult:
    """Lying flat, low in the frame, outside the bed zone."""
    return _pose(
        {
            "nose": (0.70, 0.85),
            "left_shoulder": (0.65, 0.83),
            "right_shoulder": (0.75, 0.83),
            "left_hip": (0.68, 0.86),
            "right_hip": (0.78, 0.86),
            "left_knee": (0.72, 0.87),
            "right_knee": (0.82, 0.87),
            "left_ankle": (0.75, 0.88),
            "right_ankle": (0.85, 0.88),
        }
    )


# --- classify_pose: pure geometry, one frame at a time -----------------------------


def test_classify_pose_absent_when_no_detection():
    state, confidence = classify_pose(None, "other", THRESHOLDS)
    assert state == "absent"
    assert confidence == 0.0


def test_classify_pose_absent_when_confidence_too_low():
    pose = standing_pose()
    low_confidence_pose = PoseResult(landmarks=pose.landmarks, bbox=pose.bbox, confidence=0.1)
    state, _ = classify_pose(low_confidence_pose, "other", THRESHOLDS)
    assert state == "absent"


def test_classify_pose_in_bed():
    state, _ = classify_pose(in_bed_pose(), "bed", THRESHOLDS)
    assert state == "in_bed"


def test_classify_pose_sitting_up():
    state, _ = classify_pose(sitting_up_pose(), "other", THRESHOLDS)
    assert state == "sitting_up"


def test_classify_pose_standing():
    state, _ = classify_pose(standing_pose(), "other", THRESHOLDS)
    assert state == "standing"


def test_classify_pose_on_floor():
    state, _ = classify_pose(on_floor_pose(), "other", THRESHOLDS)
    assert state == "on_floor"


def test_classify_pose_lying_in_bed_zone_is_never_on_floor():
    # Same lying geometry as on_floor_pose, but the zone is "bed": PLAN.md
    # section 6.2's "no person standing or sitting in the bed zone" rule.
    state, _ = classify_pose(on_floor_pose(), "bed", THRESHOLDS)
    assert state != "on_floor"


# --- StateTracker: hysteresis and walking -------------------------------------------


def test_tracker_withholds_a_change_until_confirm_frames_agree():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=3)

    assert tracker.update(in_bed_pose(), "bed", now=0.0) is None
    assert tracker.update(in_bed_pose(), "bed", now=1.0) is None
    result = tracker.update(in_bed_pose(), "bed", now=2.0)

    assert result is not None
    assert result[0] == "in_bed"


def test_tracker_resets_confirmation_count_on_disagreement():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=2)

    assert tracker.update(sitting_up_pose(), "other", now=0.0) is None
    # a single disagreeing frame in between must not carry over a partial count
    assert tracker.update(standing_pose(), "other", now=1.0) is None
    assert tracker.update(sitting_up_pose(), "other", now=2.0) is None
    result = tracker.update(sitting_up_pose(), "other", now=3.0)

    assert result is not None
    assert result[0] == "sitting_up"


def test_tracker_on_floor_bypasses_hysteresis():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=5)

    result = tracker.update(on_floor_pose(), "other", now=0.0)

    assert result is not None
    assert result[0] == "on_floor"


def test_tracker_absent_bypasses_hysteresis():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=5)

    result = tracker.update(None, "other", now=0.0)

    assert result is not None
    assert result[0] == "absent"


def test_tracker_does_not_republish_the_same_immediate_state_twice():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=1)

    first = tracker.update(on_floor_pose(), "other", now=0.0)
    second = tracker.update(on_floor_pose(), "other", now=1.0)

    assert first is not None
    assert second is None


def test_tracker_promotes_standing_to_walking_on_displacement():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=2)

    # settle into "standing" first
    tracker.update(standing_pose(0.50), "other", now=0.0)
    settled = tracker.update(standing_pose(0.50), "other", now=1.0)
    assert settled is not None
    assert settled[0] == "standing"

    # centroid now drifts by 0.30, well past the default 0.15 walk threshold
    tracker.update(standing_pose(0.65), "other", now=2.0)
    walking = tracker.update(standing_pose(0.80), "other", now=3.0)

    assert walking is not None
    assert walking[0] == "walking"


def test_tracker_snapshot_is_none_before_first_update():
    tracker = StateTracker(thresholds=THRESHOLDS)
    assert tracker.snapshot() is None


def test_tracker_snapshot_is_none_while_still_pending_confirmation():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=5)
    tracker.update(in_bed_pose(), "bed", now=0.0)  # only 1 of 5 needed -- not yet confirmed
    assert tracker.snapshot() is None


def test_tracker_snapshot_reflects_the_last_confirmed_state():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=1)
    tracker.update(in_bed_pose(), "bed", now=0.0)  # confirm_frames=1: confirmed immediately

    state, confidence, zone = tracker.snapshot()

    assert state == "in_bed"
    assert zone == "bed"
    assert confidence > 0.0


def test_losing_a_covered_sleeper_holds_in_bed_instead_of_reporting_absent():
    """A blanket hiding the sleeper must not read as an empty room.

    PLAN.md section 6.2 says a blanket defeats pose models, so the backend
    seeing nobody in the bed is the expected reading on a normal night, not
    a fault. Reporting `absent` there would send HANDOFF.md rule 5 straight
    to `escalate_phone` and ring the caregiver on a quiet night.
    """
    tracker = StateTracker(confirm_frames=1)
    assert tracker.update(in_bed_pose(), "bed", now=0.0) == ("in_bed", 0.9)

    for i in range(20):
        assert tracker.update(None, "other", now=1.0 + i) is None

    state, _confidence, zone = tracker.snapshot()
    assert state == "in_bed"
    assert zone == "bed"


def test_bed_hold_expires_when_a_timeout_is_configured():
    """`bed_hold_seconds` bounds the hold for anyone who wants the alarm."""
    tracker = StateTracker(confirm_frames=1, bed_hold_seconds=10.0)
    tracker.update(in_bed_pose(), "bed", now=0.0)

    assert tracker.update(None, "other", now=5.0) is None
    assert tracker.update(None, "other", now=9.9) is None
    assert tracker.update(None, "other", now=20.0) == ("absent", 0.0)


def test_leaving_the_bed_still_reaches_absent_through_the_visible_transition():
    """The hold is safe because a real bed exit is seen before it is lost.

    Nobody reaches the door without passing through `standing` in open
    view, so the tracker is no longer on `in_bed` by the time detection
    drops and the hold never applies.
    """
    tracker = StateTracker(confirm_frames=1)
    tracker.update(in_bed_pose(), "bed", now=0.0)
    tracker.update(standing_pose(), "other", now=1.0)

    assert tracker.update(None, "other", now=2.0) == ("absent", 0.0)


def test_a_detection_reaching_the_bed_again_clears_the_hold_timer():
    """Seeing the person resets the clock, so brief dropouts never accumulate."""
    tracker = StateTracker(confirm_frames=1, bed_hold_seconds=10.0)
    tracker.update(in_bed_pose(), "bed", now=0.0)

    assert tracker.update(None, "other", now=8.0) is None
    assert tracker.update(in_bed_pose(), "bed", now=9.0) is None  # unchanged, still in_bed
    assert tracker.update(None, "other", now=17.0) is None  # 8 s into a fresh hold
