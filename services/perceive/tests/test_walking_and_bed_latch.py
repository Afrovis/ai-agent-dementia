"""Tests for the ground-point walk detector, sticky bed occupancy and the
feet-based zone for upright people (docs/WALKING_BED_OCCUPANCY_PLAN.md).

Hand-built poses, no model weights, same as `test_classify`.
"""

from perceive.backends import Landmark, PoseResult
from perceive.classify import (
    ClassifyThresholds,
    StateTracker,
    ground_xy,
    ground_zone_for_pose,
    zone_for_pose,
)
from perceive.main import PerceiveConfig, build_tracker
from perceive.zones import ZoneMap
from tests.test_classify import in_bed_pose, on_floor_pose, standing_pose

# A bed drawn from the camera's point of view: it covers the middle of the
# frame, where a person standing in front of it has their torso, but not the
# floor row their feet are on.
BED_OVER_TORSO = ZoneMap(polygons={"bed": [(0.0, 0.2), (1.0, 0.2), (1.0, 0.9), (0.0, 0.9)]})


def _transformed(pose: PoseResult, *, dx: float = 0.0, scale: float = 1.0) -> PoseResult:
    """`pose` shifted by `dx` and scaled about its feet (the bottom of its box),
    as a person moving sideways, or towards/away from the camera, looks."""
    bottom = pose.bbox[3]

    def y(value: float) -> float:
        return bottom - (bottom - value) * scale

    landmarks = {
        name: Landmark(x=lm.x + dx, y=y(lm.y), visibility=lm.visibility)
        for name, lm in pose.landmarks.items()
    }
    x_min, y_min, x_max, y_max = pose.bbox
    return PoseResult(
        landmarks=landmarks,
        bbox=(x_min + dx, y(y_min), x_max + dx, y_max),
        confidence=pose.confidence,
    )


# --- ground point ---------------------------------------------------------------------


def test_ground_xy_is_the_ankle_midpoint():
    x, y = ground_xy(standing_pose())
    assert abs(x - 0.50) < 1e-9
    assert abs(y - 0.95) < 1e-9


def test_ground_xy_falls_back_to_the_box_bottom_without_visible_ankles():
    pose = standing_pose()
    landmarks = {k: v for k, v in pose.landmarks.items() if not k.endswith("ankle")}
    no_ankles = PoseResult(landmarks=landmarks, bbox=(0.4, 0.1, 0.6, 0.9), confidence=0.9)
    assert ground_xy(no_ankles) == (0.5, 0.9)


def test_feet_zone_differs_from_centroid_zone_in_front_of_the_bed():
    pose = standing_pose()
    assert zone_for_pose(BED_OVER_TORSO, pose) == "bed"
    assert ground_zone_for_pose(BED_OVER_TORSO, pose) == "other"


# --- walk_mode="ground" ---------------------------------------------------------------


def _walk_towards_camera(tracker: StateTracker) -> list:
    results = []
    for step, scale in enumerate((0.6, 0.7, 0.8, 0.9, 1.0)):
        pose = _transformed(standing_pose(), scale=scale)
        results.append(tracker.update(pose, "other", now=step * 0.5))
    return results


def test_walking_towards_the_camera_is_walking_in_ground_mode():
    tracker = StateTracker(thresholds=ClassifyThresholds(walk_mode="ground"), confirm_frames=1)
    results = _walk_towards_camera(tracker)
    assert ("walking", 0.9) in results
    assert tracker.snapshot()[0] == "walking"


def test_walking_towards_the_camera_is_invisible_to_the_legacy_rule():
    tracker = StateTracker(confirm_frames=1)
    _walk_towards_camera(tracker)
    assert tracker.snapshot()[0] == "standing"


def test_standing_still_with_jitter_stays_standing_in_ground_mode():
    tracker = StateTracker(thresholds=ClassifyThresholds(walk_mode="ground"), confirm_frames=1)
    for step, dx in enumerate((0.0, 0.01, -0.01, 0.0, 0.01, -0.01)):
        tracker.update(_transformed(standing_pose(), dx=dx), "other", now=step * 0.5)
    assert tracker.snapshot()[0] == "standing"


def test_motion_older_than_the_window_does_not_count():
    thresholds = ClassifyThresholds(walk_mode="ground", walk_window_seconds=2.0)
    tracker = StateTracker(thresholds=thresholds, confirm_frames=1)
    tracker.update(standing_pose(0.2), "other", now=0.0)
    # A big move, but spread over far longer than the window.
    tracker.update(standing_pose(0.8), "other", now=10.0)
    assert tracker.snapshot()[0] == "standing"


def test_one_non_upright_frame_does_not_restart_the_walk():
    thresholds = ClassifyThresholds(walk_mode="ground", walk_motion_threshold=0.3)
    tracker = StateTracker(thresholds=thresholds, confirm_frames=1)
    tracker.update(standing_pose(0.30), "other", now=0.0)
    # A mid-stride misread: lying-shaped, outside the bed, but not low enough
    # for the floor rule, so it only ages the window.
    tracker.update(in_bed_pose(), "other", now=0.5)
    assert tracker.update(standing_pose(0.60), "other", now=1.0) == ("walking", 0.9)


# --- bed_latch ------------------------------------------------------------------------


def _latched_tracker(**overrides) -> StateTracker:
    tracker = StateTracker(
        thresholds=ClassifyThresholds(bed_latch=True, **overrides), confirm_frames=1
    )
    assert tracker.update(in_bed_pose(), "bed", now=0.0) == ("in_bed", 0.9)
    return tracker


def test_latched_upright_frame_with_feet_on_the_bed_is_sitting_up():
    tracker = _latched_tracker()
    result = tracker.update(standing_pose(), "bed", now=0.5, ground_zone="bed")
    assert result == ("sitting_up", 0.9)


def test_without_the_latch_the_same_frame_is_standing():
    tracker = StateTracker(confirm_frames=1)
    tracker.update(in_bed_pose(), "bed", now=0.0)
    result = tracker.update(standing_pose(), "bed", now=0.5, ground_zone="bed")
    assert result == ("standing", 0.9)


def test_feet_off_the_bed_release_the_latch():
    tracker = _latched_tracker()
    assert tracker.update(standing_pose(), "bed", now=0.5, ground_zone="other") == (
        "standing",
        0.9,
    )
    # Released: a later upright frame on the bed zone is standing again.
    assert tracker.update(standing_pose(), "bed", now=1.0, ground_zone="bed") is None
    assert tracker.snapshot()[0] == "standing"


def test_walking_releases_the_latch_even_with_feet_on_the_bed():
    tracker = _latched_tracker(walk_motion_threshold=0.3)
    tracker.update(standing_pose(0.2), "bed", now=0.5, ground_zone="bed")
    result = tracker.update(standing_pose(0.6), "bed", now=1.0, ground_zone="bed")
    assert result is not None
    assert result[0] in ("standing", "walking")


def test_latch_never_suppresses_on_floor():
    tracker = _latched_tracker()
    assert tracker.update(on_floor_pose(), "other", now=0.5) == ("on_floor", 0.9)


def test_latched_person_who_sits_up_and_vanishes_is_in_bed():
    tracker = _latched_tracker()
    tracker.update(standing_pose(), "bed", now=0.5, ground_zone="bed")
    assert tracker.snapshot()[0] == "sitting_up"
    assert tracker.update(None, "other", now=1.0) == ("in_bed", 0.0)


# --- upright_zone_from_feet -----------------------------------------------------------


def test_upright_person_reports_the_zone_of_their_feet():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(upright_zone_from_feet=True), confirm_frames=1
    )
    tracker.update(standing_pose(), "bed", now=0.0, ground_zone="other")
    assert tracker.snapshot() == ("standing", 0.9, "other")


def test_person_walking_past_the_bed_and_lost_is_not_held_in_bed():
    thresholds = ClassifyThresholds(bed_vanish_hold=True, upright_zone_from_feet=True)
    tracker = StateTracker(thresholds=thresholds, confirm_frames=1)
    tracker.update(standing_pose(), "bed", now=0.0, ground_zone="other")
    assert tracker.update(None, "other", now=0.5) == ("absent", 0.0)


def test_seated_person_keeps_the_centroid_zone():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(upright_zone_from_feet=True), confirm_frames=1
    )
    tracker.update(in_bed_pose(), "bed", now=0.0, ground_zone="other")
    assert tracker.snapshot() == ("in_bed", 0.9, "bed")


# --- env wiring -----------------------------------------------------------------------


def test_env_vars_reach_the_tracker():
    config = PerceiveConfig.from_env(
        {
            "PERCEIVE_WALK_MODE": "ground",
            "PERCEIVE_WALK_MOTION_THRESHOLD": "0.4",
            "PERCEIVE_WALK_WINDOW_SECONDS": "1.5",
            "PERCEIVE_BED_LATCH": "true",
            "PERCEIVE_UPRIGHT_ZONE_FROM_FEET": "true",
        }
    )
    thresholds = build_tracker(config).thresholds
    assert thresholds.walk_mode == "ground"
    assert thresholds.walk_motion_threshold == 0.4
    assert thresholds.walk_window_seconds == 1.5
    assert thresholds.bed_latch is True
    assert thresholds.upright_zone_from_feet is True


def test_new_rules_are_off_by_default():
    thresholds = build_tracker(PerceiveConfig.from_env({})).thresholds
    assert thresholds.walk_mode == "legacy"
    assert thresholds.bed_latch is False
    assert thresholds.upright_zone_from_feet is False
