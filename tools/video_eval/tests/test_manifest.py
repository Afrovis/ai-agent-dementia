import yaml

from video_eval.manifest import load_raw_manifest, upsert_clip


def test_upsert_clip_creates_manifest_when_absent(tmp_path):
    path = upsert_clip(tmp_path, "clip-1", confirmed_by="Recorder", confirmed_at="2026-09-15")

    assert path == tmp_path / "manifest.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw["clips"] == [
        {"clip_id": "clip-1", "confirmed_by": "Recorder", "confirmed_at": "2026-09-15"}
    ]


def test_upsert_clip_is_idempotent_and_preserves_other_entries(tmp_path):
    upsert_clip(tmp_path, "clip-a", confirmed_by="Recorder", confirmed_at="2026-09-13")
    upsert_clip(tmp_path, "clip-b", confirmed_by="Recorder", confirmed_at="2026-09-14")

    # Re-confirm clip-a with an updated date; must update in place, not duplicate.
    upsert_clip(tmp_path, "clip-a", confirmed_by="Recorder", confirmed_at="2026-09-15")

    raw = load_raw_manifest(tmp_path)
    assert [entry["clip_id"] for entry in raw["clips"]] == ["clip-a", "clip-b"]
    assert raw["clips"][0]["confirmed_at"] == "2026-09-15"
    assert raw["clips"][1]["confirmed_at"] == "2026-09-14"


def test_load_raw_manifest_returns_empty_clips_when_absent(tmp_path):
    assert load_raw_manifest(tmp_path) == {"clips": []}
