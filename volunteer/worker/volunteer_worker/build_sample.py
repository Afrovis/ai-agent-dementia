"""Build the static homepage demo (HANDOFF.md section 8, V9) from an
already-recorded, already-analyzed `video_eval` clip.

`sample.json` carries each frame's raw prediction fields (`bbox`,
`landmarks`) untouched. The runtime-to-source coordinate transform is
deferred to `sample.js`, mirroring how `visualize.py` also renders at draw
time rather than baking pixel coordinates into the data, so both stay in
sync with any future live-analysis overlay that shares the same math.

    python -m volunteer_worker.build_sample --eval-root /eval \\
        --clip 2026-09-13_bedroom-sample-02 \\
        --tag yolo_yolo11s-pose-letterbox640-8a36a3e9-g
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

SAMPLE_FIELDS = ("t_s", "state", "state_confidence", "gated", "bbox", "landmarks")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_sample_records(
    frames: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(predictions) != len(frames):
        raise ValueError(
            f"predictions has {len(predictions)} records; frames.jsonl has {len(frames)}"
        )
    return [{field: record.get(field) for field in SAMPLE_FIELDS} for record in predictions]


def find_source_video(eval_root: Path, clip: str) -> Path:
    clip_dir = eval_root / "clips" / clip
    metadata = yaml.safe_load((clip_dir / "clip.yaml").read_text(encoding="utf-8"))
    source_name = metadata["source"]
    candidates = list(eval_root.glob(f"**/{source_name}"))
    if not candidates:
        raise FileNotFoundError(f"source video {source_name!r} not found under {eval_root}")
    return candidates[0]


def encode_sample_video(source_video: Path, duration_s: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-t",
            str(duration_s),
            "-vf",
            "scale=1280:720",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-f",
            "mp4",
            str(tmp_path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg could not build the sample video: {stderr[-2000:]}")
    os.replace(tmp_path, out_path)


def build_sample(eval_root: Path, clip: str, tag: str, data_root: Path) -> None:
    clip_dir = eval_root / "clips" / clip
    frames = read_jsonl(clip_dir / "frames.jsonl")
    predictions = read_jsonl(clip_dir / "predictions" / f"{tag}.jsonl")
    records = build_sample_records(frames, predictions)
    duration_s = max(float(record["t_s"]) for record in frames)

    source_video = find_source_video(eval_root, clip)

    sample_dir = data_root / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    json_path = sample_dir / "sample.json"
    tmp_json_path = json_path.with_name(json_path.name + ".tmp")
    tmp_json_path.write_text(json.dumps(records), encoding="utf-8")
    os.replace(tmp_json_path, json_path)

    encode_sample_video(source_video, duration_s, sample_dir / "sample.mp4")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data-dir", default=os.environ.get("VOLUNTEER_DATA_DIR_MOUNT", "/data"))
    args = parser.parse_args(argv)

    build_sample(Path(args.eval_root), args.clip, args.tag, Path(args.data_dir))
    print(f"wrote {Path(args.data_dir) / 'sample' / 'sample.mp4'} and sample.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
