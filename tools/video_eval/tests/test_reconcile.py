from pathlib import Path

import yaml
from PIL import Image

from video_eval.common import read_jsonl, write_jsonl
from video_eval.reconcile import derive_walking, reconcile_clip, reconcile_value, smooth_intervals


class Detection:
    def __init__(self, x: float):
        self.bbox = (x, 0, x + 10, 20)


class MovingDetector:
    def __init__(self):
        self.index = 0

    def detect(self, _image):
        value = self.index * 20
        self.index += 1
        return [Detection(value)]


def _fixture(root: Path, *, disputed: bool = False) -> None:
    clip = root / "clips" / "clip"
    review = clip / "review"
    review.mkdir(parents=True)
    frames = []
    local = []
    codex = []
    for index in range(6):
        image = review / f"f_{index:06d}.jpg"
        Image.new("RGB", (100, 50), "white").save(image)
        frames.append(
            {
                "frame_index": index,
                "t_s": index * 0.5,
                "review_path": str(image.relative_to(root)),
            }
        )
        local.append(
            {
                "frame_index": index,
                "posture": "upright",
                "location": "other",
                "confidence": 0.8,
            }
        )
        codex.append(
            {
                "frame_index": index,
                "posture": "on_floor" if disputed and index == 0 else "upright",
                "location": "other",
                "confidence": 0.6 if disputed and index == 0 else 0.9,
            }
        )
    write_jsonl(clip / "frames.jsonl", frames)
    write_jsonl(clip / "labels" / "local.jsonl", local)
    write_jsonl(clip / "labels" / "codex.jsonl", codex)
    (clip / "clip.yaml").write_text("clip_id: clip\nscript: unknown\n")


def test_value_rule_and_walking_derivation():
    assert reconcile_value({"posture": "upright"}, {"posture": "upright"}, "posture") == "upright"
    assert (
        reconcile_value(
            {"posture": "upright"}, {"posture": "on_floor", "confidence": 0.8}, "posture"
        )
        == "on_floor"
    )
    assert (
        reconcile_value(
            {"posture": "upright"}, {"posture": "on_floor", "confidence": 0.69}, "posture"
        )
        == "disputed"
    )
    rows = [{"frame_index": i, "state": "upright"} for i in range(5)]
    derive_walking(rows, dict(enumerate((0.1, 0.12, 0.16, 0.2, 0.3))), 0.15)
    assert [row["state"] for row in rows] == [
        "standing",
        "standing",
        "standing",
        "standing",
        "walking",
    ]


def test_smoothing_absorbs_noise_but_never_floor():
    intervals = [
        {"from_s": 0.0, "to_s": 3.0, "state": "in_bed", "zone": "bed"},
        {"from_s": 3.0, "to_s": 3.5, "state": "standing", "zone": "bed"},
        {"from_s": 3.5, "to_s": 6.0, "state": "in_bed", "zone": "bed"},
        {"from_s": 6.0, "to_s": 6.5, "state": "on_floor", "zone": "other"},
    ]
    result = smooth_intervals(intervals)
    assert result[0] == {"from_s": 0.0, "to_s": 6.0, "state": "in_bed", "zone": "bed"}
    assert result[1]["state"] == "on_floor"


def test_reconcile_writes_draft_disagreements_and_confirmation(tmp_path):
    _fixture(tmp_path, disputed=True)
    result = reconcile_clip("clip", root=tmp_path, detector_factory=MovingDetector)
    assert result["disagreements"] == 2  # disputed run plus unknown scenario card
    draft_path = tmp_path / "clips" / "clip" / "labels" / "reference.draft.yaml"
    draft = yaml.safe_load(draft_path.read_text())
    assert draft["timeline"][0]["state"] == "disputed"
    assert "Frames 0" in (draft_path.parent / "disagreements.md").read_text()

    draft["timeline"][0]["state"] = "standing"
    draft_path.write_text(yaml.safe_dump(draft, sort_keys=False))
    confirmed = reconcile_clip("clip", root=tmp_path, confirm=True, confirmer="Recorder")
    assert confirmed["status"] == "confirmed"
    reference = yaml.safe_load((draft_path.parent / "reference.yaml").read_text())
    assert reference["confirmed_by"] == "Recorder"
    assert reference["confirmed_at"]
    assert read_jsonl(tmp_path / "clips" / "clip" / "labels" / "local.jsonl")

    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["clips"] == [
        {
            "clip_id": "clip",
            "confirmed_by": reference["confirmed_by"],
            "confirmed_at": reference["confirmed_at"],
        }
    ]

    # Re-confirming is idempotent: it updates the manifest entry, never
    # duplicates it.
    reconcile_clip("clip", root=tmp_path, confirm=True, confirmer="Second Reviewer")
    manifest_again = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert len(manifest_again["clips"]) == 1
    assert manifest_again["clips"][0]["confirmed_by"] == "Second Reviewer"
