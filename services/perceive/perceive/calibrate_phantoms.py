"""Calibration CLI: `python -m perceive.calibrate_phantoms`.

Produces the `PERCEIVE_PHANTOMS_FILE` YAML `perceive.phantom.KnownPhantoms`
loads at startup. Run this once, with the room known to be empty, then
restart `perceive` to pick up the result (see services/perceive/AGENTS.md).

Two frame sources:

- `--frames-dir DIR`: read every JPEG in `DIR` in sorted order. This is
  what `tools/video_eval` uses to calibrate against a prepared empty-room
  clip offline -- no camera, no Redis, no live service needed.
- `--redis URL --count N`: read `N` live `Frame` events off the `frames`
  stream the same way `perceive.main.run_once` does (a `nc_shared.bus.Bus`
  consumer group), for a caregiver to run against the real camera with the
  room empty.

Either way, the configured `PoseBackend` (`PERCEIVE_POSE_BACKEND` /
`PERCEIVE_YOLO_MODEL` / `PERCEIVE_YOLO_IMGSZ`, same as the live service)
runs unfiltered -- any existing `PERCEIVE_PHANTOMS_FILE` is ignored here on
purpose, so recalibrating never has last run's phantoms hiding this run's
detections -- and `perceive.phantom.cluster_static_boxes` turns the boxes
it saw into the small, explicit list this script writes out.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import redis
import yaml
from nc_shared.bus import Bus

from perceive.backends import Yolo26MlxPoseBackend, YoloPoseBackend, build_backend
from perceive.main import PerceiveConfig
from perceive.phantom import Box, cluster_static_boxes

FRAME_STREAM = "frames"
FRAME_GROUP = "perceive-calibrate"


def _read_frames_dir(directory: Path) -> list[bytes]:
    """Read every `*.jpg`/`*.jpeg`/`*.png` in `directory`, sorted by name."""
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [p.read_bytes() for p in paths]


def _read_redis_frames(
    redis_url: str, count: int, *, block_ms: int = 2000, timeout_seconds: float = 120.0
) -> list[bytes]:
    """Read up to `count` `Frame` events off the live `frames` stream,
    acking each as it is consumed, mirroring `perceive.main.run_once`."""
    bus = Bus(redis.Redis.from_url(redis_url))
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    jpegs: list[bytes] = []
    deadline = time.monotonic() + timeout_seconds
    while len(jpegs) < count and time.monotonic() < deadline:
        messages = bus.read(FRAME_STREAM, FRAME_GROUP, "calibrate-1", count=1, block_ms=block_ms)
        for msg_id, frame in messages:
            jpegs.append(frame.jpeg)
            bus.ack(FRAME_STREAM, FRAME_GROUP, msg_id)
    return jpegs


def calibrate(
    jpegs: list[bytes],
    *,
    detect_conf: float = 0.15,
    min_frame_fraction: float = 0.3,
) -> list[Box]:
    """Run the configured backend over `jpegs` (unfiltered) and cluster the
    resulting boxes into calibrated phantoms."""
    config = PerceiveConfig.from_env()
    backend = build_backend(
        config.pose_backend,
        model_path=config.yolo_model,
        static_image_mode=not config.mediapipe_video_mode,
        yolo_imgsz=config.yolo_imgsz,
        detect_conf=detect_conf,
        # Explicit empty string, not `None`: `None` would fall back to
        # `PERCEIVE_PHANTOMS_FILE` from the environment, which is exactly
        # the stale-result-hiding-new-results bug recalibrating must avoid.
        phantoms_file="",
    )
    if not isinstance(backend, (YoloPoseBackend, Yolo26MlxPoseBackend)):
        raise ValueError(
            "phantom calibration requires PERCEIVE_POSE_BACKEND=yolo or yolo26mlx"
        )

    boxes: list[list[Box]] = []
    for jpeg in jpegs:
        boxes.append([candidate.box for candidate in backend.detect_candidates(jpeg)])
    return cluster_static_boxes(boxes, min_frame_fraction=min_frame_fraction)


def write_phantoms(path: Path, boxes: list[Box]) -> None:
    data = {"phantoms": [{"box": list(box)} for box in boxes]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate perceive's known-phantom list from frames taken while the room is empty."
        )
    )
    parser.add_argument("--out", required=True, help="path to write the phantoms YAML to")
    parser.add_argument(
        "--frames-dir", help="directory of JPEG/PNG frames (offline mode, e.g. video_eval)"
    )
    parser.add_argument("--redis", help="Redis URL, e.g. redis://bus:6379 (live mode)")
    parser.add_argument("--count", type=int, default=60, help="frames to read in live mode")
    parser.add_argument(
        "--detect-conf",
        type=float,
        default=0.15,
        help="YOLO candidate confidence threshold (default: 0.15)",
    )
    parser.add_argument(
        "--min-frame-fraction",
        type=float,
        default=0.3,
        help="minimum fraction of frames containing a phantom (default: 0.3)",
    )
    args = parser.parse_args(argv)

    if bool(args.frames_dir) == bool(args.redis):
        parser.error("pass exactly one of --frames-dir or --redis")

    if args.frames_dir:
        jpegs = _read_frames_dir(Path(args.frames_dir))
    else:
        jpegs = _read_redis_frames(args.redis, args.count)

    if not jpegs:
        print("no frames read; nothing to calibrate", file=sys.stderr)
        return 1

    boxes = calibrate(
        jpegs,
        detect_conf=args.detect_conf,
        min_frame_fraction=args.min_frame_fraction,
    )
    write_phantoms(Path(args.out), boxes)

    print(f"read {len(jpegs)} frames, found {len(boxes)} known phantom(s):")
    for box in boxes:
        print(f"  box={[round(v, 4) for v in box]}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
