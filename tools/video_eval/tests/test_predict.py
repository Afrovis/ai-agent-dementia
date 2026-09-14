from pathlib import Path

import yaml
from perceive.backends import LANDMARK_NAMES, Landmark, PoseResult
from PIL import Image

from video_eval.common import read_jsonl, write_jsonl
from video_eval.predict import backend_label, predict_clip, prediction_tag


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def detect(self, jpeg: bytes) -> PoseResult:
        assert jpeg
        self.calls += 1
        landmarks = {
            name: Landmark(x=0.5, y=0.2 if "shoulder" in name else 0.8, visibility=0.9)
            for name in LANDMARK_NAMES
        }
        return PoseResult(landmarks=landmarks, bbox=(0.4, 0.1, 0.6, 0.9), confidence=0.9)

    def close(self) -> None:
        self.closed = True


def _prepared_clip(root: Path, frame_count: int = 70, bridge_key: str = "bridge_path") -> None:
    clip = root / "clips" / "clip"
    bridge = clip / "bridge"
    bridge.mkdir(parents=True)
    image_path = bridge / "f_000000.jpg"
    Image.new("RGB", (320, 240), "black").save(image_path, format="JPEG")
    records = [
        {
            "frame_index": index,
            "t_s": index / 2,
            bridge_key: str(image_path.relative_to(root)),
            "review_path": "unused",
        }
        for index in range(frame_count)
    ]
    write_jsonl(clip / "frames.jsonl", records)
    (clip / "zones.yaml").write_text(
        yaml.safe_dump({"bed": [[0, 0], [1, 0], [1, 1], [0, 1]]}), encoding="utf-8"
    )


def test_prediction_tag_records_backend_variant_sha_and_gate():
    assert prediction_tag("yolo", "letterbox", True, "1234567890") == ("yolo-letterbox-12345678-g")
    assert prediction_tag("yolo", "squash", False, "1234567890") == "yolo-squash-12345678"


def test_backend_label_defaults_to_plain_backend_name():
    assert backend_label("yolo", None, False) == "yolo"
    assert backend_label("yolo", "yolov8n-pose.pt", False) == "yolo"


def test_backend_label_includes_yolo_model_stem_when_not_default():
    assert backend_label("yolo", "yolov8s-pose.pt", False) == "yolo_yolov8s-pose"


def test_backend_label_marks_mediapipe_video_mode():
    assert backend_label("mediapipe", None, True) == "mediapipe_video"


def test_predict_reads_letterbox640_manifest_key(tmp_path):
    _prepared_clip(tmp_path, frame_count=2, bridge_key="bridge_letterbox_640_path")
    backend = FakeBackend()

    result = predict_clip(
        "clip",
        root=tmp_path,
        backend_name="fake",
        variant="letterbox640",
        no_gate=True,
        confirm_frames=1,
        backend_factory=lambda _: backend,
    )

    records = read_jsonl(tmp_path / "clips" / "clip" / "predictions" / f"{result['tag']}.jsonl")
    assert backend.calls == 2
    assert len(records) == 2


def test_predict_runs_backend_tracker_and_motion_gate_offline(tmp_path, monkeypatch):
    _prepared_clip(tmp_path)
    backend = FakeBackend()
    monkeypatch.setenv("CAPTURE_STATIC_SECONDS", "1")

    result = predict_clip(
        "clip",
        root=tmp_path,
        backend_name="fake",
        confirm_frames=1,
        backend_factory=lambda _: backend,
    )

    records = read_jsonl(tmp_path / "clips" / "clip" / "predictions" / f"{result['tag']}.jsonl")
    assert len(records) == 70
    assert any(row["gated"] for row in records)
    assert backend.calls == sum(not row["gated"] for row in records)
    assert backend.closed
    admitted = next(row for row in records if not row["gated"])
    assert admitted["detected"] is True
    assert admitted["state"] == "sitting_up"
    assert admitted["zone"] == "bed"
    gated = next(row for row in records if row["gated"])
    assert gated["detected"] is None
    assert gated["backend_ms"] == 0.0

    repeated = predict_clip(
        "clip",
        root=tmp_path,
        backend_name="fake",
        confirm_frames=1,
        backend_factory=lambda _: FakeBackend(),
    )
    assert repeated["status"] == "skipped"


def test_no_gate_passes_every_frame_to_backend(tmp_path):
    _prepared_clip(tmp_path, frame_count=4)
    backend = FakeBackend()

    result = predict_clip(
        "clip",
        root=tmp_path,
        backend_name="fake",
        no_gate=True,
        confirm_frames=1,
        backend_factory=lambda _: backend,
    )

    records = read_jsonl(tmp_path / "clips" / "clip" / "predictions" / f"{result['tag']}.jsonl")
    assert backend.calls == 4
    assert all(not row["gated"] for row in records)
