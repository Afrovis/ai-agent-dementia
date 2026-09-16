"""Run the real capture/perception path against prepared frames offline."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from capture.main import CaptureConfig, build_gate
from perceive.backends import PoseBackend, build_backend
from perceive.classify import ground_zone_for_pose, zone_for_pose
from perceive.main import PerceiveConfig, build_tracker
from perceive.zones import load_zones

from video_eval.common import (
    git_sha,
    matching_meta,
    read_jsonl,
    update_index,
    write_jsonl,
    write_meta,
)
from video_eval.paths import EvalPaths, manifest_key


def prediction_tag(backend: str, variant: str, gated: bool, sha: str) -> str:
    suffix = "-g" if gated else ""
    return f"{backend}-{variant}-{sha[:8]}{suffix}"


def backend_label(backend_name: str, yolo_model: str | None, mediapipe_video_mode: bool) -> str:
    if mediapipe_video_mode:
        return f"{backend_name}_video"
    if yolo_model is not None and Path(yolo_model).stem != "yolov8n-pose":
        return f"{backend_name}_{Path(yolo_model).stem}"
    return backend_name


def predict_clip(
    clip_id: str,
    *,
    root: Path | None = None,
    backend_name: str = "mediapipe",
    variant: str = "squash",
    no_gate: bool = False,
    force: bool = False,
    confirm_frames: int | None = None,
    min_confidence: float | None = None,
    walk_threshold: float | None = None,
    yolo_model: str | None = None,
    presence_confidence: float | None = None,
    mediapipe_video_mode: bool = False,
    backend_factory: Callable[[str], PoseBackend] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    paths = EvalPaths.for_clip(clip_id, root)
    frames = read_jsonl(paths.frames)
    sha = git_sha()
    tag = prediction_tag(
        backend_label(backend_name, yolo_model, mediapipe_video_mode), variant, not no_gate, sha
    )
    output = paths.predictions / f"{tag}.jsonl"
    meta_path = paths.predictions / f"{tag}.meta.json"
    parameters = {
        "backend": backend_name,
        "clip_id": clip_id,
        "confirm_frames": confirm_frames,
        "gated": not no_gate,
        "min_confidence": min_confidence,
        "variant": variant,
        "walk_threshold": walk_threshold,
        "yolo_model": yolo_model,
        "presence_confidence": presence_confidence,
        "mediapipe_video_mode": mediapipe_video_mode,
    }
    if not force and output.exists() and matching_meta(meta_path, parameters):
        return {"status": "skipped", "frames": len(read_jsonl(output)), "tag": tag}

    perceive_config = PerceiveConfig.from_env()
    if confirm_frames is not None:
        perceive_config = replace(perceive_config, confirm_frames=confirm_frames)
    if min_confidence is not None:
        perceive_config = replace(perceive_config, min_confidence=min_confidence)
    if walk_threshold is not None:
        perceive_config = replace(perceive_config, walk_threshold=walk_threshold)
    if presence_confidence is not None:
        perceive_config = replace(perceive_config, presence_confidence=presence_confidence)
    tracker = build_tracker(perceive_config)
    zones_path = paths.clip / "zones.yaml"
    zones = load_zones(zones_path)
    gate = build_gate(CaptureConfig.from_env())
    if backend_factory is not None:
        backend = backend_factory(backend_name)
    else:
        backend = build_backend(
            backend_name, model_path=yolo_model, static_image_mode=not mediapipe_video_mode
        )
    path_key = manifest_key(variant)
    records: list[dict[str, Any]] = []
    try:
        for frame in frames:
            frame_path_value = frame.get(path_key)
            if not frame_path_value:
                raise RuntimeError(f"manifest has no {path_key}; run prepare --variant {variant}")
            jpeg = (paths.root / frame_path_value).read_bytes()
            t_s = float(frame["t_s"])
            gated = not no_gate and not gate.admit(jpeg, t_s)
            pose = None
            backend_ms = 0.0
            published = False
            if not gated:
                before = time.perf_counter()
                pose = backend.detect(jpeg)
                backend_ms = (time.perf_counter() - before) * 1000.0
                zone = zone_for_pose(zones, pose) if pose is not None else "other"
                ground_zone = ground_zone_for_pose(zones, pose) if pose is not None else None
                published = tracker.update(pose, zone, t_s, ground_zone=ground_zone) is not None
            snapshot = tracker.snapshot()
            records.append(
                {
                    "backend_ms": round(backend_ms, 3),
                    "bbox": list(pose.bbox) if pose is not None else None,
                    "detect_confidence": pose.confidence if pose is not None else None,
                    "detected": None if gated else pose is not None,
                    "frame_index": int(frame["frame_index"]),
                    "gated": gated,
                    "published": published,
                    "landmarks": (
                        {
                            name: [point.x, point.y, point.visibility]
                            for name, point in pose.landmarks.items()
                        }
                        if pose is not None
                        else None
                    ),
                    "state": snapshot[0] if snapshot else None,
                    "state_confidence": snapshot[1] if snapshot else None,
                    "t_s": t_s,
                    "zone": snapshot[2] if snapshot else None,
                }
            )
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()

    write_jsonl(output, records)
    write_meta(
        meta_path,
        command="predict",
        parameters=parameters,
        started_at=started,
        versions=("pillow", "pyyaml", "mediapipe", "ultralytics"),
    )
    update_index(paths.root, clip_id, f"predict_{tag}", "complete")
    return {"status": "complete", "frames": len(records), "tag": tag}
