import hashlib
import json

import pytest
from PIL import Image

from video_eval.common import read_jsonl, write_jsonl
from video_eval.label_codex import label_codex
from video_eval.label_local import label_local
from video_eval.labels import validate_label

VALID = {
    "person_visible": True,
    "posture": "upright",
    "location": "other",
    "confidence": 0.9,
    "note": "person is upright",
}


def _prepared_frames(root, count=1):
    review = root / "clips" / "test" / "review"
    review.mkdir(parents=True)
    records = []
    for index in range(count):
        path = review / f"f_{index:06d}.jpg"
        Image.new("RGB", (64, 36), "gray").save(path)
        records.append(
            {
                "frame_index": index,
                "t_s": index / 2,
                "review_path": f"clips/test/review/{path.name}",
            }
        )
    write_jsonl(root / "clips" / "test" / "frames.jsonl", records)


def test_label_validation_rejects_enum_and_long_note():
    validate_label(VALID)
    with pytest.raises(ValueError, match="posture"):
        validate_label({**VALID, "posture": "walking"})
    with pytest.raises(ValueError, match="15 words"):
        validate_label({**VALID, "note": " ".join(["word"] * 16)})


def test_local_retries_invalid_and_propagates_static_frames(tmp_path):
    root = tmp_path / "private"
    _prepared_frames(root, 12)

    class FakeLabeller:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def label(self, _image):
            self.calls += 1
            if self.calls == 1:
                return {**VALID, "posture": "invalid"}
            return VALID

        def close(self):
            self.closed = True

    fake = FakeLabeller()
    result = label_local("test", root=root, labeller_factory=lambda: fake)

    assert result["model_calls"] == 3
    assert fake.calls == 3  # first selected frame retries once; second is frame 10
    assert fake.closed is True
    labels = read_jsonl(root / "clips" / "test" / "labels" / "local.jsonl")
    assert len(labels) == 12
    assert labels[1]["propagated_from"] == 0
    assert "propagated_from" not in labels[10]


def test_codex_refuses_before_review_then_labels_only_manifested_sheet(tmp_path):
    root = tmp_path / "private"
    clip = root / "clips" / "test"
    sheets = clip / "sheets"
    sheets.mkdir(parents=True)
    Image.new("RGB", (100, 100), "black").save(sheets / "s_0001.jpg")
    write_jsonl(
        clip / "sheets.jsonl",
        [
            {
                "sheet_path": "clips/test/sheets/s_0001.jpg",
                "frames": [{"frame_index": 7, "t_s": 3.5}],
            }
        ],
    )
    with pytest.raises(RuntimeError, match="human privacy spot-check"):
        label_codex("test", root=root, runner=lambda *_: "[]")

    manifest = clip / "sheets.jsonl"
    sheet = sheets / "s_0001.jpg"
    (clip / "sheets.reviewed.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "reviewed": True,
                "sheet_count": 1,
                "sheet_sha256": {
                    "clips/test/sheets/s_0001.jpg": hashlib.sha256(sheet.read_bytes()).hexdigest()
                },
            }
        )
    )
    seen = []

    def runner(path, prompt, model):
        seen.append((path, prompt, model))
        return json.dumps([VALID])

    result = label_codex("test", root=root, runner=runner)

    assert result["frames"] == 1
    assert seen[0][0] == (sheets / "s_0001.jpg").resolve()
    assert "frame 7" in seen[0][1]
    labels = read_jsonl(clip / "labels" / "codex.jsonl")
    assert labels[0]["frame_index"] == 7

    sheet.write_bytes(b"changed after review")
    with pytest.raises(RuntimeError, match="changed after human review"):
        label_codex("test", root=root, runner=runner, force=True)
