"""Tests for `perceive.backends`, without loading real model weights.

No model weights, no camera, no GPU: real mediapipe and ultralytics packages
are never imported here, matching HANDOFF.md's testing convention and issue
#8's requirement that the service imports and its tests run with neither
installed.
"""

import sys
from types import SimpleNamespace

from perceive.backends import (
    LANDMARK_NAMES,
    Landmark,
    MediaPipeBackend,
    PoseResult,
    ScriptedBackend,
    build_backend,
)


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
