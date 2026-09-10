"""Tests for tier 3: `perception_bench.infrared`."""

from pathlib import Path

import pytest

from perception_bench.infrared import load_manifest, run_tier3


def test_load_manifest_missing_file_returns_empty_list():
    assert load_manifest(Path("/nonexistent/ir_manifest.yaml")) == []


def test_run_tier3_reports_not_measured_when_manifest_absent():
    result = run_tier3(Path("/nonexistent/ir_manifest.yaml"))
    assert result.clip_count == 0
    assert result.missing_reason is not None
    assert "RECORDING.md" in result.missing_reason


def test_example_manifest_in_this_repo_parses_and_reports_files_missing():
    """`fixtures/ir_manifest.example.yaml` documents the format with a
    worked example; it should parse cleanly even though its clip file does
    not exist (it is illustrative, per its own header comment)."""
    example_path = Path(__file__).parent.parent / "fixtures" / "ir_manifest.example.yaml"
    clips = load_manifest(example_path)
    assert len(clips) == 1
    clip = clips[0]
    assert clip.frame_interval_s == 0.5
    assert [interval.state for interval in clip.timeline] == [
        "in_bed",
        "sitting_up",
        "standing",
        "walking",
        "on_floor",
        "absent",
    ]
    assert "bed" in clip.zones.polygons

    result = run_tier3(example_path)
    assert result.clip_count == 1
    assert result.missing_reason is not None  # the clip file itself does not exist


def test_load_manifest_rejects_clip_without_timeline(tmp_path):
    manifest = tmp_path / "ir_manifest.yaml"
    manifest.write_text("clips:\n  - path: clips/a.mp4\n")
    with pytest.raises(ValueError, match="timeline"):
        load_manifest(manifest)


def test_load_manifest_rejects_clip_without_path(tmp_path):
    manifest = tmp_path / "ir_manifest.yaml"
    manifest.write_text("clips:\n  - timeline: []\n")
    with pytest.raises(ValueError, match="path"):
        load_manifest(manifest)


def test_load_manifest_paths_are_relative_to_manifest_directory(tmp_path):
    manifest = tmp_path / "sub" / "ir_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text(
        "clips:\n"
        "  - path: clips/a.mp4\n"
        "    timeline:\n"
        "      - {from_s: 0.0, to_s: 1.0, state: in_bed}\n"
    )
    clips = load_manifest(manifest)
    assert clips[0].path == manifest.parent / "clips" / "a.mp4"
