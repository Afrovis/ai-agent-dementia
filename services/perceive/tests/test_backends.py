"""Tests for `perceive.backends`, without loading real model weights.

No model weights, no camera, no GPU: real mediapipe and ultralytics packages
are never imported here, matching HANDOFF.md's testing convention and issue
#8's requirement that the service imports and its tests run with neither
installed.
"""

import io
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from perceive.backends import (
    LANDMARK_NAMES,
    Landmark,
    MediaPipeBackend,
    PoseResult,
    ScriptedBackend,
    Yolo26MlxPoseBackend,
    YoloPoseBackend,
    build_backend,
)
from perceive.phantom import KnownPhantoms


def _pose(confidence: float = 0.9) -> PoseResult:
    landmarks = {name: Landmark(x=0.5, y=0.5, visibility=0.9) for name in LANDMARK_NAMES}
    return PoseResult(landmarks=landmarks, bbox=(0.4, 0.4, 0.6, 0.6), confidence=confidence)


def test_scripted_backend_returns_results_in_order():
    a, b = _pose(0.9), _pose(0.8)
    backend = ScriptedBackend([a, None, b])

    assert backend.detect(b"ignored") is a
    assert backend.detect(b"ignored") is None
    assert backend.detect(b"ignored") is b


def test_scripted_backend_exhausts_to_none_by_default():
    backend = ScriptedBackend([_pose()])

    backend.detect(b"ignored")
    assert backend.detect(b"ignored") is None
    assert backend.detect(b"ignored") is None


def test_scripted_backend_cycles_when_asked():
    a, b = _pose(0.9), _pose(0.8)
    backend = ScriptedBackend([a, b], cycle=True)

    assert [backend.detect(b"x") for _ in range(4)] == [a, b, a, b]


def test_scripted_backend_with_no_results_always_returns_none():
    backend = ScriptedBackend([])
    assert backend.detect(b"x") is None


def test_build_backend_scripted_reports_no_person():
    backend = build_backend("scripted")
    assert backend.detect(b"x") is None


def test_build_backend_rejects_unknown_kind():
    try:
        build_backend("unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_mediapipe_backend_uses_pinned_solutions_api(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakePose:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    fake_mediapipe = SimpleNamespace(solutions=SimpleNamespace(pose=SimpleNamespace(Pose=FakePose)))
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)

    MediaPipeBackend(min_detection_confidence=0.65)

    assert calls == [
        {
            "static_image_mode": True,
            "model_complexity": 1,
            "min_detection_confidence": 0.65,
        }
    ]


def test_pose_result_carries_every_canonical_landmark():
    pose = _pose()
    assert set(pose.landmarks) == set(LANDMARK_NAMES)


class _Tensor:
    """Minimal stand-in for an ultralytics tensor: `.tolist()` and indexing,
    same pattern as `tests.test_partial_landmarks`."""

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


def _stub_yolo_backend(results_by_call, *, known_phantoms=None) -> YoloPoseBackend:
    """A `YoloPoseBackend` whose `_model` call returns the next entry of
    `results_by_call` each time it is invoked, with a given `KnownPhantoms`
    wired in directly (unlike `tests.test_partial_landmarks._stub_backend`,
    which builds an inactive one -- that file is about keypoint dropping,
    not phantom filtering)."""
    backend = YoloPoseBackend.__new__(YoloPoseBackend)
    calls = iter(results_by_call)
    backend._model = lambda image, verbose, imgsz: [next(calls)]
    backend._imgsz = 640
    backend._detect_conf = None
    backend._known_phantoms = known_phantoms if known_phantoms is not None else KnownPhantoms([])
    return backend


def _boxed_result(boxes_and_confidences):
    """Build an ultralytics-shaped `_Result` with one dummy keypoint set per
    box (all keypoints at a fixed, confident position -- these tests only
    care about which box gets selected)."""
    xyn = [[0.5, 0.5]] * 17
    conf = [0.9] * 17
    keypoints_per_box = [xyn] * len(boxes_and_confidences)
    conf_per_box = [conf] * len(boxes_and_confidences)
    boxes = [box for box, _confidence in boxes_and_confidences]
    confidences = [confidence for _box, confidence in boxes_and_confidences]
    return _Result(_Boxes(confidences, boxes), _Keypoints(keypoints_per_box, conf_per_box))


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "black").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_yolo_backend_passes_contiguous_bgr_array_to_model():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="JPEG")
    received: list[np.ndarray] = []

    backend = YoloPoseBackend.__new__(YoloPoseBackend)
    backend._model = lambda image, verbose, imgsz: received.append(image) or []
    backend._imgsz = 640
    backend._detect_conf = None
    backend._known_phantoms = KnownPhantoms([])

    assert backend.detect(buffer.getvalue()) is None
    assert received[0].flags.c_contiguous
    assert received[0][..., 2].mean() > 240
    assert received[0][..., 0].mean() < 15


def test_yolo_backend_returns_all_unfiltered_candidates_at_requested_confidence():
    phantom_box = (0.19, 0.31, 0.26, 0.46)
    person_box = (0.5, 0.4, 0.7, 0.9)
    received_conf: list[float] = []
    result = _boxed_result([(phantom_box, 0.2), (person_box, 0.4)])

    backend = YoloPoseBackend.__new__(YoloPoseBackend)
    backend._model = lambda image, verbose, imgsz, conf: received_conf.append(conf) or [result]
    backend._imgsz = 640
    backend._detect_conf = 0.15
    backend._known_phantoms = KnownPhantoms([phantom_box], max_confidence=0.7)

    candidates = backend.detect_candidates(_jpeg())

    assert received_conf == [0.15]
    assert [candidate.box for candidate in candidates] == [phantom_box, person_box]


def test_yolo_backend_excludes_a_calibrated_known_phantom():
    phantom_box = (0.19, 0.31, 0.26, 0.46)
    known_phantoms = KnownPhantoms([phantom_box], max_confidence=0.7)

    # Empty room: only the calibrated phantom box comes back, below the
    # confidence floor -- must be excluded entirely.
    backend = _stub_yolo_backend(
        [_boxed_result([(phantom_box, 0.6)])], known_phantoms=known_phantoms
    )
    assert backend.detect(_jpeg()) is None


def test_yolo_backend_selects_the_real_person_over_a_calibrated_phantom():
    phantom_box = (0.19, 0.31, 0.26, 0.46)
    known_phantoms = KnownPhantoms([phantom_box], max_confidence=0.7)
    person_box = (0.5, 0.4, 0.7, 0.9)

    backend = _stub_yolo_backend(
        [_boxed_result([(phantom_box, 0.6), (person_box, 0.3)])],
        known_phantoms=known_phantoms,
    )
    pose = backend.detect(_jpeg())
    assert pose is not None
    assert pose.bbox == pytest.approx(person_box)


class _MlxBoxes:
    """Minimal stand-in for `yolo26mlx`'s `Boxes`: absolute-pixel `.xyxy`
    (no `.xyxyn`, unlike ultralytics), plain lists rather than tensors."""

    def __init__(self, conf, xyxy):
        self.conf = conf
        self.xyxy = xyxy

    def __len__(self):
        return len(self.conf)


class _MlxKeypoints:
    """Minimal stand-in for `yolo26mlx`'s `Keypoints`: absolute-pixel
    `.data`, shape `(N, 17, 3)`, unlike ultralytics' pre-normalised `.xyn`."""

    def __init__(self, data):
        self.data = data


class _MlxResult:
    def __init__(self, boxes, keypoints, orig_shape):
        self.boxes = boxes
        self.keypoints = keypoints
        self.orig_shape = orig_shape


def _mlx_boxed_result(boxes_and_confidences, *, width=100, height=100):
    """Build a `yolo26mlx`-shaped `_MlxResult` in absolute pixel coordinates,
    one dummy keypoint set per box, mirroring `_boxed_result` above."""
    keypoints_per_box = [[[width * 0.5, height * 0.5, 0.9]] * 17] * len(boxes_and_confidences)
    boxes_px = [
        [box[0] * width, box[1] * height, box[2] * width, box[3] * height]
        for box, _confidence in boxes_and_confidences
    ]
    confidences = [confidence for _box, confidence in boxes_and_confidences]
    return _MlxResult(
        _MlxBoxes(confidences, boxes_px), _MlxKeypoints(keypoints_per_box), (height, width)
    )


def _stub_mlx_backend(results_by_call, *, known_phantoms=None) -> Yolo26MlxPoseBackend:
    """A `Yolo26MlxPoseBackend` whose `_model.predict` returns the next entry
    of `results_by_call` each time it is invoked, mirroring `_stub_yolo_backend`."""
    backend = Yolo26MlxPoseBackend.__new__(Yolo26MlxPoseBackend)
    calls = iter(results_by_call)
    backend._model = SimpleNamespace(predict=lambda image, conf: [next(calls)])
    backend._imgsz = 640
    backend._detect_conf = None
    backend._known_phantoms = known_phantoms if known_phantoms is not None else KnownPhantoms([])
    return backend


def test_mlx_backend_normalises_absolute_pixel_boxes_and_keypoints():
    person_box = (0.5, 0.4, 0.7, 0.9)
    result = _mlx_boxed_result([(person_box, 0.8)], width=200, height=100)

    backend = _stub_mlx_backend([result])
    pose = backend.detect(_jpeg())

    assert pose is not None
    assert pose.bbox == pytest.approx(person_box)
    assert pose.landmarks["nose"].x == pytest.approx(0.5)
    assert pose.landmarks["nose"].y == pytest.approx(0.5)
    assert pose.landmarks["nose"].visibility == pytest.approx(0.9)


def test_mlx_backend_excludes_a_calibrated_known_phantom():
    phantom_box = (0.19, 0.31, 0.26, 0.46)
    known_phantoms = KnownPhantoms([phantom_box], max_confidence=0.7)

    backend = _stub_mlx_backend(
        [_mlx_boxed_result([(phantom_box, 0.6)])], known_phantoms=known_phantoms
    )
    assert backend.detect(_jpeg()) is None


def test_mlx_backend_selects_the_real_person_over_a_calibrated_phantom():
    phantom_box = (0.19, 0.31, 0.26, 0.46)
    known_phantoms = KnownPhantoms([phantom_box], max_confidence=0.7)
    person_box = (0.5, 0.4, 0.7, 0.9)

    backend = _stub_mlx_backend(
        [_mlx_boxed_result([(phantom_box, 0.6), (person_box, 0.3)])],
        known_phantoms=known_phantoms,
    )
    pose = backend.detect(_jpeg())
    assert pose is not None
    assert pose.bbox == pytest.approx(person_box)


def test_mlx_backend_passes_confidence_through_to_predict():
    person_box = (0.5, 0.4, 0.7, 0.9)
    received_conf: list[float] = []
    result = _mlx_boxed_result([(person_box, 0.8)])

    backend = Yolo26MlxPoseBackend.__new__(Yolo26MlxPoseBackend)
    backend._model = SimpleNamespace(
        predict=lambda image, conf: received_conf.append(conf) or [result]
    )
    backend._imgsz = 640
    backend._detect_conf = 0.35
    backend._known_phantoms = KnownPhantoms([])

    backend.detect(_jpeg())

    assert received_conf == [0.35]
