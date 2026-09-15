"""Tests for tier 3: `perception_bench.infrared`."""

import sys
from pathlib import Path

import pytest
import yaml

from perception_bench.infrared import ManifestClip, load_manifest, run_tier3, tag_family


def _write_manifest(root: Path, clips: list[dict]) -> Path:
    manifest = root / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"clips": clips}, sort_keys=False), encoding="utf-8")
    return manifest


def _write_reference(
    root: Path, clip_id: str, *, confirmed: bool = True, timeline: list[dict] | None = None
) -> None:
    labels = root / "clips" / clip_id / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    reference: dict = {
        "clip_id": clip_id,
        "frame_interval_s": 0.5,
        "timeline": timeline
        or [
            {"from_s": 0, "to_s": 2, "state": "standing", "zone": "other"},
            {"from_s": 2, "to_s": 4, "state": "on_floor", "zone": "other"},
        ],
    }
    if confirmed:
        reference["confirmed_by"] = "Recorder"
        reference["confirmed_at"] = "2026-09-13"
    (labels / "reference.yaml").write_text(yaml.safe_dump(reference), encoding="utf-8")


def _write_frames_and_predictions(root: Path, clip_id: str, tag: str) -> None:
    from video_eval.common import write_jsonl

    clip = root / "clips" / clip_id
    frames = [{"frame_index": index, "t_s": index * 0.5} for index in range(8)]
    predictions = []
    for index in range(8):
        state = "standing" if index < 4 else "on_floor"
        predictions.append(
            {
                "frame_index": index,
                "t_s": index * 0.5,
                "state": state,
                "zone": "other",
                "gated": False,
                "detected": True,
                "published": True,
            }
        )
    write_jsonl(clip / "frames.jsonl", frames)
    write_jsonl(clip / "predictions" / f"{tag}.jsonl", predictions)


# --- manifest loading -------------------------------------------------


def test_load_manifest_missing_file_returns_empty_list():
    assert load_manifest(Path("/nonexistent/manifest.yaml")) == []


def test_load_manifest_parses_clip_id_and_confirmation(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        [{"clip_id": "clip-1", "confirmed_by": "Recorder", "confirmed_at": "2026-09-15"}],
    )
    clips = load_manifest(manifest)
    assert clips == [
        ManifestClip(clip_id="clip-1", confirmed_by="Recorder", confirmed_at="2026-09-15")
    ]


def test_load_manifest_rejects_clip_without_clip_id(tmp_path):
    manifest = _write_manifest(tmp_path, [{"confirmed_by": "Recorder"}])
    with pytest.raises(ValueError, match="clip_id"):
        load_manifest(manifest)


def test_load_manifest_rejects_non_mapping(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_manifest(manifest)


def test_load_manifest_rejects_clips_not_a_list(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("clips: not-a-list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        load_manifest(manifest)


def test_example_manifest_in_this_repo_parses_and_skips_missing_clip():
    """`fixtures/ir_manifest.example.yaml` documents the format with a
    worked example; it should parse cleanly even though its clip does not
    exist (it is illustrative, per its own header comment)."""
    pytest.importorskip("video_eval")
    example_path = Path(__file__).parent.parent / "fixtures" / "ir_manifest.example.yaml"
    clips = load_manifest(example_path)
    assert clips == [
        ManifestClip(
            clip_id="clip-id-does-not-exist",
            confirmed_by="Recorder Name",
            confirmed_at="2026-09-15",
        )
    ]

    result = run_tier3(example_path)
    assert result.clip_count == 1
    assert result.missing_reason is None
    assert result.skipped == [("clip-id-does-not-exist", "labels/reference.yaml is missing")]


# --- tag_family ---------------------------------------------------------


def test_tag_family_strips_sha_and_gate_suffix():
    assert tag_family("mediapipe-squash-abcdef12") == "mediapipe-squash"
    assert tag_family("mediapipe-squash-abcdef12-g") == "mediapipe-squash"
    assert tag_family("yolo-letterbox-01234567-g") == "yolo-letterbox"


# --- run_tier3 skip paths -------------------------------------------------


def test_run_tier3_reports_not_measured_when_manifest_absent():
    result = run_tier3(Path("/nonexistent/manifest.yaml"))
    assert result.clip_count == 0
    assert result.missing_reason is not None
    assert "RECORDING.md" in result.missing_reason


def test_run_tier3_reports_install_hint_when_video_eval_missing(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-1"}])
    # `video_eval.paths`/`video_eval.score` may already be cached in
    # `sys.modules` from another test in this run; clear every submodule too,
    # not just the top-level package, so the `from video_eval.x import y`
    # inside `run_tier3` genuinely re-resolves and fails.
    cached = [
        name for name in sys.modules if name == "video_eval" or name.startswith("video_eval.")
    ]
    for name in cached:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "video_eval", None)

    result = run_tier3(manifest)

    assert result.clip_count == 1
    assert result.missing_reason is not None
    assert "video_eval" in result.missing_reason
    assert "install" in result.missing_reason.lower()
    assert result.clip_results == []


def test_run_tier3_skips_unconfirmed_clip(tmp_path):
    pytest.importorskip("video_eval")
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-1"}])
    _write_reference(tmp_path, "clip-1", confirmed=False)

    result = run_tier3(manifest)

    assert result.missing_reason is None
    assert result.clip_results == []
    assert result.skipped == [
        ("clip-1", "labels/reference.yaml is not confirmed (no confirmed_by)")
    ]


def test_run_tier3_skips_clip_with_missing_reference(tmp_path):
    pytest.importorskip("video_eval")
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-1"}])

    result = run_tier3(manifest)

    assert result.missing_reason is None
    assert result.skipped == [("clip-1", "labels/reference.yaml is missing")]


def test_run_tier3_skips_clip_with_no_prediction_files(tmp_path):
    pytest.importorskip("video_eval")
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-1"}])
    _write_reference(tmp_path, "clip-1")

    result = run_tier3(manifest)

    assert result.missing_reason is None
    assert result.skipped == [
        ("clip-1", "no prediction files found under predictions/ (after filtering)")
    ]


# --- scoring --------------------------------------------------------------


def test_run_tier3_scores_confirmed_clip_with_prediction_file(tmp_path):
    pytest.importorskip("video_eval")
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-1"}])
    _write_reference(tmp_path, "clip-1")
    _write_frames_and_predictions(tmp_path, "clip-1", "mediapipe-squash-01234567")

    result = run_tier3(manifest)

    assert result.missing_reason is None
    assert result.skipped == []
    assert len(result.clip_results) == 1
    row = result.clip_results[0]
    assert row.clip_id == "clip-1"
    assert row.tag == "mediapipe-squash-01234567"
    assert row.frame_count == 8
    assert row.standing_recall == 1.0
    assert row.on_floor_recall == 1.0
    assert row.gates["standing_recall"]["met"] is True
    assert row.gates["on_floor_recall"]["met"] is True
    assert row.gates["on_floor_delay"]["met"] is True

    assert len(result.pooled_results) == 1
    pooled = result.pooled_results[0]
    assert pooled.tag_family == "mediapipe-squash"
    assert pooled.clip_ids == ["clip-1"]
    assert pooled.frame_count == 8
    assert pooled.standing_recall == 1.0
    assert pooled.on_floor_recall == 1.0
    assert pooled.gates["standing_recall"]["met"] is True
    assert pooled.gates["on_floor_recall"]["met"] is True


def test_run_tier3_pools_tag_family_across_clips_with_different_shas(tmp_path):
    pytest.importorskip("video_eval")
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-a"}, {"clip_id": "clip-b"}])
    _write_reference(tmp_path, "clip-a")
    _write_reference(tmp_path, "clip-b")
    _write_frames_and_predictions(tmp_path, "clip-a", "mediapipe-squash-01234567")
    _write_frames_and_predictions(tmp_path, "clip-b", "mediapipe-squash-89abcdef-g")

    result = run_tier3(manifest)

    assert len(result.clip_results) == 2
    assert len(result.pooled_results) == 1
    pooled = result.pooled_results[0]
    assert pooled.tag_family == "mediapipe-squash"
    assert sorted(pooled.clip_ids) == ["clip-a", "clip-b"]
    assert pooled.frame_count == 16
    assert pooled.standing_recall == 1.0
    assert pooled.on_floor_recall == 1.0


def test_run_tier3_ir_tag_filters_prediction_tags(tmp_path):
    pytest.importorskip("video_eval")
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-1"}])
    _write_reference(tmp_path, "clip-1")
    _write_frames_and_predictions(tmp_path, "clip-1", "mediapipe-squash-01234567")
    _write_frames_and_predictions(tmp_path, "clip-1", "yolo-squash-01234567")

    result = run_tier3(manifest, tag_glob="mediapipe-*")

    assert [row.tag for row in result.clip_results] == ["mediapipe-squash-01234567"]

    result_none = run_tier3(manifest, tag_glob="nothing-matches-*")
    assert result_none.clip_results == []
    assert result_none.skipped == [
        ("clip-1", "no prediction files found under predictions/ (after filtering)")
    ]


def test_run_tier3_ir_backend_runs_predict_before_scoring(tmp_path, monkeypatch):
    pytest.importorskip("video_eval")
    manifest = _write_manifest(tmp_path, [{"clip_id": "clip-1"}])
    _write_reference(tmp_path, "clip-1")

    def fake_predict_clip(clip_id, *, root, backend_name, variant, force):
        _write_frames_and_predictions(root, clip_id, "fakebackend-squash-01234567")
        return {"status": "complete", "frames": 8, "tag": "fakebackend-squash-01234567"}

    monkeypatch.setattr("video_eval.predict.predict_clip", fake_predict_clip)

    result = run_tier3(manifest, backend="fakebackend")

    assert [row.tag for row in result.clip_results] == ["fakebackend-squash-01234567"]
