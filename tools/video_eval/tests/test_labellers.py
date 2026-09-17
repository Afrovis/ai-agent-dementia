import hashlib
import json

import pytest
from PIL import Image

from video_eval.common import read_jsonl, write_jsonl
from video_eval.label_codex import label_codex
from video_eval.label_local import MlxLabeller, OllamaLabeller, label_local
from video_eval.labels import LABEL_SCHEMA, validate_label

VALID = {
    "person_visible": True,
    "posture": "upright",
    "location": "other",
    "confidence": 0.9,
    "note": "person is upright",
}

_MISSING = object()


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


@pytest.mark.parametrize(
    ("person_visible", "expected"),
    [
        ("true", True),
        ("FALSE", False),
        (" True ", True),
        (1, True),
        (0, False),
        (True, True),
        (False, False),
    ],
)
def test_label_validation_coerces_unambiguous_person_visible(person_visible, expected):
    label = validate_label({**VALID, "person_visible": person_visible})

    assert label.person_visible is expected


@pytest.mark.parametrize(
    "person_visible",
    [
        "yes",
        "1",
        2,
        -1,
        None,
        1.0,
        pytest.param(_MISSING, id="missing"),
    ],
)
def test_label_validation_rejects_ambiguous_person_visible(person_visible):
    raw = {key: value for key, value in VALID.items() if key != "person_visible"}
    if person_visible is not _MISSING:
        raw["person_visible"] = person_visible

    with pytest.raises(ValueError, match="person_visible must be a boolean"):
        validate_label(raw)


def test_ollama_labeller_requests_structured_label_schema(monkeypatch):
    labeller = OllamaLabeller(base_url="http://ollama.test", model="test-model")
    seen = {}

    def fake_post(path, payload):
        seen.update(path=path, payload=payload)
        return {"message": {"content": json.dumps(VALID)}}

    monkeypatch.setattr(labeller, "_post", fake_post)

    assert labeller.label(b"jpeg") == VALID
    assert seen["path"] == "/api/chat"
    assert isinstance(seen["payload"]["format"], dict)
    assert seen["payload"]["format"] == LABEL_SCHEMA
    assert seen["payload"]["format"] != "json"


class _FakeSchemaProcessor:
    def clone(self):
        return self


def test_mlx_labeller_parses_generated_json():
    def fake_loader(model_id):
        assert model_id == "test-mlx-model"
        return ("model", "processor", "prompt", 0, _FakeSchemaProcessor())

    seen = {}

    def fake_generator(model, processor, prompt, image_path, logits_processor):
        seen.update(
            model=model,
            processor=processor,
            prompt=prompt,
            image_path=image_path,
            logits_processor=logits_processor,
        )
        return json.dumps(VALID)

    labeller = MlxLabeller(model="test-mlx-model", loader=fake_loader, generator=fake_generator)

    assert labeller.label(b"jpeg") == VALID
    assert seen["model"] == "model"
    assert seen["processor"] == "processor"
    assert seen["prompt"] == "prompt"
    assert seen["image_path"].endswith(".jpg")


def test_mlx_labeller_rejects_invalid_json():
    def fake_loader(_model_id):
        return ("model", "processor", "prompt", 0, _FakeSchemaProcessor())

    labeller = MlxLabeller(
        model="test-mlx-model", loader=fake_loader, generator=lambda *_: "not json"
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        labeller.label(b"jpeg")


def test_local_uses_mlx_backend_and_tags_records(tmp_path):
    root = tmp_path / "private"
    _prepared_frames(root, 1)

    def fake_loader(_model_id):
        return ("model", "processor", "prompt", 0, _FakeSchemaProcessor())

    labeller = MlxLabeller(
        model="test-mlx-model",
        loader=fake_loader,
        generator=lambda *_: json.dumps(VALID),
    )
    result = label_local(
        "test", root=root, backend="mlx", model="test-mlx-model", labeller_factory=lambda: labeller
    )

    assert result["model"] == "test-mlx-model"
    labels = read_jsonl(root / "clips" / "test" / "labels" / "local.jsonl")
    assert labels[0]["labeller"] == "mlx:test-mlx-model"


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
