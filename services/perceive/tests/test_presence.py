"""Tests for the presence threshold: a detection that is too shaky to read a
posture from but confident enough to prove somebody is in the frame.

Hand-built poses, no model weights, same as `test_classify`.
"""

from perceive.backends import PoseResult
from perceive.classify import ClassifyThresholds, StateTracker, classify_pose
from perceive.main import PerceiveConfig, build_tracker
from tests.test_classify import THRESHOLDS, in_bed_pose, standing_pose


def _with_confidence(pose: PoseResult, confidence: float) -> PoseResult:
    return PoseResult(landmarks=pose.landmarks, bbox=pose.bbox, confidence=confidence)


# --- classify_pose ------------------------------------------------------------------


def test_low_confidence_detection_in_bed_zone_is_in_bed():
    state, confidence = classify_pose(_with_confidence(in_bed_pose(), 0.3), "bed", THRESHOLDS)
    assert state == "in_bed"
    assert confidence == 0.3


def test_low_confidence_detection_outside_bed_zone_is_absent():
    state, _ = classify_pose(_with_confidence(standing_pose(), 0.3), "other", THRESHOLDS)
    assert state == "absent"


def test_below_presence_confidence_is_absent_even_in_bed_zone():
    state, _ = classify_pose(_with_confidence(in_bed_pose(), 0.1), "bed", THRESHOLDS)
    assert state == "absent"


def test_presence_confidence_equal_to_min_confidence_disables_the_rule():
    thresholds = ClassifyThresholds(presence_confidence=0.5)
    state, _ = classify_pose(_with_confidence(in_bed_pose(), 0.3), "bed", thresholds)
    assert state == "absent"


# --- StateTracker -------------------------------------------------------------------


def test_tracker_holds_current_state_on_a_low_confidence_detection_outside_bed():
    tracker = StateTracker(confirm_frames=1)
    assert tracker.update(standing_pose(), "other", now=0.0) == ("standing", 0.9)

    assert tracker.update(_with_confidence(standing_pose(), 0.3), "other", now=0.5) is None
    assert tracker.snapshot() == ("standing", 0.9, "other")

    # Below the presence floor it is a real absence again, immediately.
    assert tracker.update(_with_confidence(standing_pose(), 0.1), "other", now=1.0) == (
        "absent",
        0.1,
    )


def test_tracker_low_confidence_detection_does_not_disturb_a_pending_confirmation():
    tracker = StateTracker(confirm_frames=2)
    assert tracker.update(standing_pose(), "other", now=0.0) is None
    assert tracker.update(_with_confidence(standing_pose(), 0.3), "other", now=0.5) is None
    assert tracker.update(standing_pose(), "other", now=1.0) == ("standing", 0.9)


def test_tracker_low_confidence_detection_in_bed_zone_confirms_in_bed():
    tracker = StateTracker(confirm_frames=2)
    low = _with_confidence(in_bed_pose(), 0.3)
    assert tracker.update(low, "bed", now=0.0) is None
    assert tracker.update(low, "bed", now=0.5) == ("in_bed", 0.3)


# --- configuration ------------------------------------------------------------------


def test_perceive_config_backend_and_presence_defaults():
    config = PerceiveConfig.from_env(env={})
    assert config.yolo_model == "yolo11s-pose.pt"
    assert config.mediapipe_video_mode is False
    assert config.presence_confidence == 0.25


def test_perceive_config_from_env_reads_backend_and_presence_options():
    env = {
        "PERCEIVE_YOLO_MODEL": "yolov8s-pose.pt",
        "PERCEIVE_MEDIAPIPE_VIDEO_MODE": "true",
        "PERCEIVE_PRESENCE_CONFIDENCE": "0.3",
    }
    config = PerceiveConfig.from_env(env=env)
    assert config.yolo_model == "yolov8s-pose.pt"
    assert config.mediapipe_video_mode is True
    assert config.presence_confidence == 0.3


def test_build_tracker_wires_presence_confidence_through():
    tracker = build_tracker(PerceiveConfig(presence_confidence=0.3))
    assert tracker.thresholds.presence_confidence == 0.3
