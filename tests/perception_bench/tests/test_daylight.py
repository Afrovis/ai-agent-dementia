"""Tests for tier 2: `perception_bench.daylight`. No network, no dataset
download -- every test uses a temp directory standing in for
`data/perception_bench/daylight/`, or nothing at all.
"""

from pathlib import Path

from perception_bench.daylight import (
    classify_directory_name,
    discover_clips,
    run_tier2,
)


def test_classify_directory_name_matches_documented_classes():
    assert classify_directory_name("Falling_Down") == "on_floor"
    assert classify_directory_name("lying-on-the-floor") == "on_floor"
    assert classify_directory_name("StandingUp") == "standing"
    assert classify_directory_name("walking_01") == "walking"
    assert classify_directory_name("empty_room") == "absent"


def test_classify_directory_name_returns_none_for_unrecognised_class():
    assert classify_directory_name("sitting_down") is None


def test_discover_clips_on_missing_directory_returns_empty_list():
    assert discover_clips(Path("/nonexistent/perception-bench-daylight")) == []


def test_discover_clips_finds_recognised_classes_only(tmp_path):
    (tmp_path / "walking" / "clip1").mkdir(parents=True)
    (tmp_path / "walking" / "clip1" / "frame_0001.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "sitting_down" / "clip1").mkdir(parents=True)  # not recognised
    (tmp_path / "sitting_down" / "clip1" / "frame_0001.jpg").write_bytes(b"\xff\xd8\xff")

    clips = discover_clips(tmp_path)
    assert len(clips) == 1
    assert clips[0].ground_truth == "walking"


def test_run_tier2_skips_cleanly_when_data_absent(tmp_path):
    result = run_tier2(tmp_path / "does-not-exist")
    assert result.accuracy is None
    assert result.skipped_reason is not None
    assert "fetch_daylight.sh" in result.skipped_reason
    assert result.clip_count == 0


def test_run_tier2_skips_cleanly_when_pose_backend_unavailable(tmp_path):
    (tmp_path / "walking" / "clip1").mkdir(parents=True)
    (tmp_path / "walking" / "clip1" / "frame_0001.jpg").write_bytes(b"\xff\xd8\xff")

    # Neither mediapipe nor ultralytics is installed in this bench's own
    # test environment, per this issue's hard constraint -- build_backend
    # raises RuntimeError, which run_tier2 must turn into a skip, not a
    # crash.
    result = run_tier2(tmp_path, pose_backend_name="mediapipe")
    assert result.accuracy is None
    assert result.skipped_reason is not None
    assert result.clip_count == 1
