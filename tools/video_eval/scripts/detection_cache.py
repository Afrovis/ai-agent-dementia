"""Cache raw pose-model output per frame so classifier rules can be replayed offline.

`predict` couples inference and classification, so every threshold or rule
experiment re-runs the model. This script runs each model once over every
prepared bridge frame and stores everything a classifier could use: every
candidate box (not only the best), all keypoints with their confidences,
per-frame CPU latency, and the motion-gate decision for the variant.
`rule_replay.py` then rebuilds `PoseResult`s from the cache and drives the
real `StateTracker`.

Output: `clips/<clip>/cache/<model>-<variant>.jsonl` plus `gate-<variant>.json`.

Usage:
    python tools/video_eval/scripts/detection_cache.py --data-root ../data-ai-agent-dementia \
        --clip <clip-id> --variant letterbox --model /path/yolov8n-pose.pt --model mediapipe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from capture.main import CaptureConfig, build_gate
from perceive.backends import MediaPipeBackend, _decode_rgb

from video_eval.paths import manifest_key

COCO_TO_MEDIAPIPE = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
"""MediaPipe 33-point indices in COCO-17 order, so both backends cache one shape."""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gate_decisions(frames: list[dict], root: Path, key: str) -> dict[int, bool]:
    gate = build_gate(CaptureConfig.from_env({}))
    return {
        int(f["frame_index"]): gate.admit((root / f[key]).read_bytes(), float(f["t_s"]))
        for f in frames
    }


def run_yolo(model_path: str, images: list[np.ndarray], conf: float) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    model(images[0], verbose=False, device="cpu")  # warm-up, excluded from latency
    out = []
    for image in images:
        before = time.perf_counter()
        result = model(image, verbose=False, conf=conf, device="cpu")[0]
        ms = (time.perf_counter() - before) * 1000.0
        dets = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxyn.tolist()
            scores = result.boxes.conf.tolist()
            kxy = result.keypoints.xyn.tolist()
            kconf = (
                result.keypoints.conf.tolist()
                if result.keypoints.conf is not None
                else [[1.0] * 17] * len(boxes)
            )
            for i, box in enumerate(boxes):
                dets.append(
                    {
                        "bbox": [round(v, 5) for v in box],
                        "conf": round(scores[i], 4),
                        "kp": [
                            [round(x, 5), round(y, 5), round(c, 4)]
                            for (x, y), c in zip(kxy[i], kconf[i])
                        ],
                    }
                )
        out.append({"ms": round(ms, 2), "dets": dets})
    return out


def run_mediapipe(images: list[np.ndarray], admitted: list[bool], video: bool) -> list[dict]:
    backend = MediaPipeBackend(static_image_mode=not video)
    out = []
    try:
        for image, admit in zip(images, admitted):
            if video and not admit:
                # Video mode only ever sees admitted frames live; feeding the
                # dropped ones would give it continuity it does not have.
                out.append({"ms": 0.0, "dets": [], "skipped": True})
                continue
            before = time.perf_counter()
            result = backend._pose.process(image)
            ms = (time.perf_counter() - before) * 1000.0
            dets = []
            if result.pose_landmarks is not None:
                lms = result.pose_landmarks.landmark
                kp = [
                    [round(lms[i].x, 5), round(lms[i].y, 5), round(lms[i].visibility, 4)]
                    for i in COCO_TO_MEDIAPIPE
                ]
                nine = [kp[i] for i in (0, 5, 6, 11, 12, 13, 14, 15, 16)]
                xs = [p[0] for p in nine]
                ys = [p[1] for p in nine]
                dets.append(
                    {
                        "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "conf": round(sum(p[2] for p in nine) / 9, 4),
                        "kp": kp,
                    }
                )
            out.append({"ms": round(ms, 2), "dets": dets})
    finally:
        backend.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="YOLO .pt path, mediapipe or mediapipe_video",
    )
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO candidate floor")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    root = args.data_root.resolve()
    clip = root / "clips" / args.clip
    cache = clip / "cache"
    cache.mkdir(exist_ok=True)
    frames = read_jsonl(clip / "frames.jsonl")
    key = manifest_key(args.variant)

    gate_path = cache / f"gate-{args.variant}.json"
    if gate_path.exists() and not args.force:
        gate = {int(k): v for k, v in json.loads(gate_path.read_text()).items()}
    else:
        gate = gate_decisions(frames, root, key)
        gate_path.write_text(json.dumps(gate))

    images = [_decode_rgb((root / f[key]).read_bytes()) for f in frames]
    admitted = [gate[int(f["frame_index"])] for f in frames]
    for model in args.model:
        name = model if model.startswith("mediapipe") else Path(model).stem
        out_path = cache / f"{name}-{args.variant}.jsonl"
        if out_path.exists() and not args.force:
            print(f"skip {out_path.name}")
            continue
        started = time.monotonic()
        if model.startswith("mediapipe"):
            rows = run_mediapipe(images, admitted, video=model == "mediapipe_video")
        else:
            rows = run_yolo(model, images, args.conf)
        with out_path.open("w", encoding="utf-8") as handle:
            for frame, row in zip(frames, rows):
                row = {"frame_index": int(frame["frame_index"]), "t_s": float(frame["t_s"]), **row}
                handle.write(json.dumps(row) + "\n")
        timed = [r["ms"] for r in rows if not r.get("skipped")]
        print(
            f"{out_path.name}: {len(rows)} frames, median {np.median(timed):.1f} ms, "
            f"{time.monotonic() - started:.0f} s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
