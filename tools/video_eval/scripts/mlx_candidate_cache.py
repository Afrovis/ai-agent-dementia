"""Cache raw yolo26mlx pose-model candidates per frame for phantom_track_finder.py.

`detection_cache.py` drives ultralytics `.pt` models directly; it has no path
for the MLX-native `Yolo26MlxPoseBackend` (different constructor, different
raw-output shape). This is the same idea, narrowed to that one backend: run
it once over every prepared bridge frame of a clip and write every raw
candidate box+confidence, unfiltered by `KnownPhantoms.select()` (that
filtering only happens inside `detect()`, never `detect_candidates()`), to
`clips/<clip>/cache/yolo26mlx-<model-stem>-<variant>.jsonl` in the same
per-line schema `detection_cache.py` uses, so `phantom_track_finder.py` can
read either.

Usage:
    python tools/video_eval/scripts/mlx_candidate_cache.py \
        --data-root ../data-ai-agent-dementia --clip <clip-id> \
        --variant letterbox640 --model-path yolo26l-pose.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from perceive.backends import Yolo26MlxPoseBackend

from video_eval.paths import manifest_key


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-path", default="yolo26l-pose.npz")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf", type=float, default=0.1, help="Candidate floor passed as detect_conf"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    root = args.data_root.resolve()
    clip = root / "clips" / args.clip
    cache = clip / "cache"
    cache.mkdir(exist_ok=True)
    frames = read_jsonl(clip / "frames.jsonl")
    key = manifest_key(args.variant)

    model_stem = Path(args.model_path).stem
    out_path = cache / f"yolo26mlx-{model_stem}-{args.variant}.jsonl"
    if out_path.exists() and not args.force:
        print(f"skip {out_path.name}")
        return 0

    backend = Yolo26MlxPoseBackend(model_path=args.model_path, imgsz=args.imgsz, detect_conf=args.conf)

    started = time.monotonic()
    rows = []
    for frame in frames:
        jpeg = (root / frame[key]).read_bytes()
        before = time.perf_counter()
        candidates = backend.detect_candidates(jpeg)
        ms = (time.perf_counter() - before) * 1000.0
        dets = [
            {"bbox": [round(v, 5) for v in c.box], "conf": round(c.confidence, 4)}
            for c in candidates
        ]
        rows.append(
            {
                "frame_index": int(frame["frame_index"]),
                "t_s": float(frame["t_s"]),
                "ms": round(ms, 2),
                "dets": dets,
            }
        )

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"{out_path.name}: {len(rows)} frames, {time.monotonic() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
