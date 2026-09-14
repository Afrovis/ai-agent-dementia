"""Tests for the rules added after the 2026-09-13 clip replay: a low box top
outside the bed is `on_floor`, short detection dropouts do not report
`absent`, a person lost in the bed zone is `in_bed`, and `on_floor` is held
while the person is hidden.

Hand-built poses, no model weights, same as `test_classify`.
"""

from perceive.backends import PoseResult
from perceive.classify import ClassifyThresholds, StateTracker, classify_pose
from tests.test_classify import THRESHOLDS, standing_pose


def _with_bbox(pose: PoseResult, bbox: tuple[float, float, float, float]) -> PoseResult:
    return PoseResult(landmarks=pose.landmarks, bbox=bbox, confidence=pose.confidence)


def _low_pose() -> PoseResult:
    """An upright-looking body whose box starts in the lower half of the frame:
    a person sitting or kneeling on the floor near the camera."""
    return _with_bbox(standing_pose(), (0.55, 0.6, 0.7, 0.87))


# --- classify_pose ------------------------------------------------------------------


def test_floor_top_rule_is_off_by_default():
    state, _ = classify_pose(_low_pose(), "other", THRESHOLDS)
    assert state != "on_floor"


def test_low_box_top_outside_bed_is_on_floor():
    thresholds = ClassifyThresholds(floor_top_y=0.5)
    assert classify_pose(_low_pose(), "other", thresholds)[0] == "on_floor"


def test_low_box_top_inside_bed_zone_is_not_on_floor():
    thresholds = ClassifyThresholds(floor_top_y=0.5)
    assert classify_pose(_low_pose(), "bed", thresholds)[0] != "on_floor"


def test_high_box_top_is_not_on_floor():
    thresholds = ClassifyThresholds(floor_top_y=0.5)
    assert classify_pose(standing_pose(), "other", thresholds)[0] != "on_floor"


# --- StateTracker: absent confirmation ---------------------------------------------


def test_short_dropout_holds_state_until_absent_confirm_seconds():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(absent_confirm_seconds=3.0), confirm_frames=1
    )
    assert tracker.update(standing_pose(), "other", now=0.0) == ("standing", 0.9)

    assert tracker.update(None, "other", now=0.5) is None
    assert tracker.update(None, "other", now=2.5) is None
    assert tracker.snapshot()[0] == "standing"

    assert tracker.update(None, "other", now=3.5) == ("absent", 0.0)


def test_detection_restarts_the_absent_clock():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(absent_confirm_seconds=3.0), confirm_frames=1
    )
    tracker.update(standing_pose(), "other", now=0.0)
    tracker.update(None, "other", now=1.0)
    tracker.update(standing_pose(), "other", now=2.0)

    # The clock starts again at the first empty frame after the detection.
    assert tracker.update(None, "other", now=4.0) is None
    assert tracker.update(None, "other", now=5.5) is None
    assert tracker.update(None, "other", now=7.0) == ("absent", 0.0)


def test_default_reports_absent_on_first_empty_frame():
    tracker = StateTracker(confirm_frames=1)
    tracker.update(standing_pose(), "other", now=0.0)
    assert tracker.update(None, "other", now=0.5) == ("absent", 0.0)


# --- StateTracker: bed vanish -------------------------------------------------------


def test_person_lost_in_bed_zone_confirms_in_bed_and_is_then_held():
    tracker = StateTracker(thresholds=ClassifyThresholds(bed_vanish_hold=True), confirm_frames=2)
    assert tracker.update(standing_pose(), "bed", now=0.0) is None
    assert tracker.update(standing_pose(), "bed", now=0.5) == ("standing", 0.9)

    assert tracker.update(None, "other", now=1.0) is None
    assert tracker.update(None, "other", now=1.5) == ("in_bed", 0.0)
    assert tracker.snapshot() == ("in_bed", 0.0, "bed")

    # The ordinary bed hold takes over from here.
    assert tracker.update(None, "other", now=600.0) is None


def test_person_lost_while_walking_through_bed_zone_is_not_in_bed():
    thresholds = ClassifyThresholds(bed_vanish_hold=True, walk_displacement_threshold=0.1)
    tracker = StateTracker(thresholds=thresholds, confirm_frames=1)
    tracker.update(standing_pose(), "bed", now=0.0)
    moved = PoseResult(
        landmarks={
            name: type(lm)(x=lm.x + 0.3, y=lm.y, visibility=lm.visibility)
            for name, lm in standing_pose().landmarks.items()
        },
        bbox=standing_pose().bbox,
        confidence=0.9,
    )
    assert tracker.update(moved, "bed", now=0.5) == ("walking", 0.9)

    assert tracker.update(None, "other", now=1.0) == ("absent", 0.0)


def test_person_lost_outside_bed_zone_is_still_absent():
    tracker = StateTracker(thresholds=ClassifyThresholds(bed_vanish_hold=True), confirm_frames=1)
    tracker.update(standing_pose(), "other", now=0.0)
    assert tracker.update(None, "other", now=0.5) == ("absent", 0.0)


# --- StateTracker: floor hold -------------------------------------------------------


def test_on_floor_is_held_while_the_person_is_hidden():
    thresholds = ClassifyThresholds(floor_top_y=0.5, hold_floor=True)
    tracker = StateTracker(thresholds=thresholds, confirm_frames=3)
    assert tracker.update(_low_pose(), "other", now=0.0) == ("on_floor", 0.9)

    assert tracker.update(None, "other", now=0.5) is None
    assert tracker.update(None, "other", now=60.0) is None
    assert tracker.snapshot()[0] == "on_floor"


def test_on_floor_without_hold_drops_to_absent():
    tracker = StateTracker(thresholds=ClassifyThresholds(floor_top_y=0.5), confirm_frames=3)
    tracker.update(_low_pose(), "other", now=0.0)
    assert tracker.update(None, "other", now=0.5) == ("absent", 0.0)
