from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from volunteer_worker.build_sample import build_sample_records, find_source_video


def test_build_sample_records_keeps_only_declared_fields() -> None:
    frames = [{"t_s": 0.0, "frame_index": 0}, {"t_s": 0.5, "frame_index": 1}]
    predictions = [
        {
            "t_s": 0.0,
            "frame_index": 0,
            "state": "standing",
            "state_confidence": 0.9,
            "gated": False,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "landmarks": {"nose": [0.5, 0.5, 0.9]},
            "detected": True,
            "zone": None,
        },
        {
            "t_s": 0.5,
            "frame_index": 1,
            "state": None,
            "state_confidence": None,
            "gated": True,
            "bbox": None,
            "landmarks": {},
            "detected": False,
            "zone": None,
        },
    ]

    records = build_sample_records(frames, predictions)

    assert records == [
        {
            "t_s": 0.0,
            "state": "standing",
            "state_confidence": 0.9,
            "gated": False,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "landmarks": {"nose": [0.5, 0.5, 0.9]},
        },
        {
            "t_s": 0.5,
            "state": None,
            "state_confidence": None,
            "gated": True,
            "bbox": None,
            "landmarks": {},
        },
    ]


def test_build_sample_records_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        build_sample_records([{"t_s": 0.0}, {"t_s": 0.5}], [{"t_s": 0.0}])


def test_find_source_video_locates_file_from_clip_yaml(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips" / "sample-clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.yaml").write_text(
        yaml.dump({"clip_id": "sample-clip", "source": "sample-clip.mov"}),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    video_path = raw_dir / "sample-clip.mov"
    video_path.write_bytes(b"not a real video")

    found = find_source_video(tmp_path, "sample-clip")

    assert found == video_path


def test_find_source_video_missing_raises(tmp_path: Path) -> None:
    clip_dir = tmp_path / "clips" / "sample-clip"
    clip_dir.mkdir(parents=True)
    (clip_dir / "clip.yaml").write_text(
        yaml.dump({"clip_id": "sample-clip", "source": "sample-clip.mov"}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        find_source_video(tmp_path, "sample-clip")
