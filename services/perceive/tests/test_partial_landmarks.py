"""Tests for poses with landmarks the backend could not place.

ultralytics zeroes the coordinates of keypoints under 0.5 confidence; the
YOLO backend must drop those rather than report the top-left corner as a
joint, and the classifier must measure what is left from the detection box.
Hand-built poses and a stub ultralytics result: no model weights.
"""

import io

import pytest
from PIL import Image

from perceive.backends import LANDMARK_NAMES, Landmark, PoseResult, YoloPoseBackend
from perceive.classify import StateTracker, centroid_of, classify_pose
from tests.test_classify import THRESHOLDS, in_bed_pose


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "black").save(buffer, format="JPEG")
    return buffer.getvalue()


class _Tensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values

    def __getitem__(self, index):
        return _Tensor(self._values[index])


class _Boxes:
    def __init__(self, conf, xyxyn):
        self.conf = _Tensor(conf)
        self.xyxyn = _Tensor(xyxyn)

    def __len__(self):
        return len(self.conf.tolist())


class _Keypoints:
    def __init__(self, xyn, conf):
        self.xyn = _Tensor(xyn)
        self.conf = _Tensor(conf)


class _Result:
    def __init__(self, boxes, keypoints):
        self.boxes = boxes
        self.keypoints = keypoints


def _stub_backend(result: _Result) -> YoloPoseBackend:
    backend = YoloPoseBackend.__new__(YoloPoseBackend)
    backend._model = lambda image, verbose: [result]
    return backend


def test_yolo_backend_drops_keypoints_ultralytics_zeroed():
    # COCO-17 layout: only nose (0), shoulders (5, 6) and hips (11, 12) have
    # positions; every other keypoint is the (0, 0) ultralytics writes for
    # a keypoint under 0.5 confidence.
    xyn = [[0.0, 0.0]] * 17
    conf = [0.0] * 17
    for index, point in {
        0: (0.10, 0.60),
        5: (0.15, 0.58),
        6: (0.15, 0.64),
        11: (0.35, 0.60),
        12: (0.35, 0.66),
    }.items():
        xyn[index] = list(point)
        conf[index] = 0.9
    conf[15] = 0.3  # an ankle with some confidence but a zeroed position
    result = _Result(_Boxes([0.42], [[0.05, 0.50, 0.60, 0.75]]), _Keypoints([xyn], [conf]))

    pose = _stub_backend(result).detect(_jpeg())

    assert pose is not None
    assert set(pose.landmarks) == {
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    }
    assert pose.bbox == (0.05, 0.50, 0.60, 0.75)
    assert pose.confidence == 0.42


def _lying_torso_only(bbox=(0.05, 0.65, 0.60, 0.88)) -> PoseResult:
    """Head, shoulders and hips of a person lying flat; legs unplaced. The
    box is the detector's whole-body estimate: wide and low."""
    points = {
        "nose": (0.10, 0.75),
        "left_shoulder": (0.15, 0.73),
        "right_shoulder": (0.15, 0.79),
        "left_hip": (0.35, 0.75),
        "right_hip": (0.35, 0.81),
    }
    landmarks = {name: Landmark(x=x, y=y, visibility=0.9) for name, (x, y) in points.items()}
    return PoseResult(landmarks=landmarks, bbox=bbox, confidence=0.8)


def test_centroid_of_partial_landmarks_is_the_box_centre():
    pose = _lying_torso_only()
    assert centroid_of(pose) == pytest.approx((0.325, 0.765))


def test_centroid_of_full_landmarks_is_still_their_mean():
    pose = in_bed_pose()
    assert len(pose.landmarks) == len(LANDMARK_NAMES)
    xs = [lm.x for lm in pose.landmarks.values()]
    ys = [lm.y for lm in pose.landmarks.values()]
    assert centroid_of(pose) == (sum(xs) / len(xs), sum(ys) / len(ys))


def test_lying_person_with_unplaced_legs_is_on_floor_outside_the_bed():
    state, _ = classify_pose(_lying_torso_only(), "other", THRESHOLDS)
    assert state == "on_floor"


def test_lying_person_with_unplaced_legs_is_in_bed_in_the_bed_zone():
    state, _ = classify_pose(_lying_torso_only(), "bed", THRESHOLDS)
    assert state == "in_bed"


def test_missing_frame_does_not_reset_a_pending_confirmation():
    tracker = StateTracker(thresholds=THRESHOLDS, confirm_frames=3)

    assert tracker.update(in_bed_pose(), "bed", now=0.0) is None
    assert tracker.update(in_bed_pose(), "bed", now=0.5) is None
    # Blanket frame: no detection. Reported as absent, as before ...
    assert tracker.update(None, "other", now=1.0) == ("absent", 0.0)
    # ... but the two agreeing detections still count towards in_bed.
    assert tracker.update(in_bed_pose(), "bed", now=1.5) == ("in_bed", 0.9)
