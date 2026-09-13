"""Prepare private video clips in the exact formats used by evaluation."""

from __future__ import annotations

import filecmp
import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from video_eval.common import matching_meta, update_index, write_jsonl, write_meta
from video_eval.paths import EvalPaths

FPS = 2.0
REVIEW_WIDTH = 640
BRIDGE_SIZE = (320, 240)


def probe_video(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,codec_name:stream_tags=rotate:format=duration",
        "-of",
        "json",
        str(video),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is required for video_eval prepare") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed: {exc.stderr.strip()}") from exc
    raw = json.loads(completed.stdout)
    stream = raw["streams"][0]
    return {
        "duration_s": float(raw["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("avg_frame_rate", "unknown"),
        "codec": stream.get("codec_name", "unknown"),
        "rotation": int(stream.get("tags", {}).get("rotate", 0)),
    }


def extract_review_frames(video: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(video),
        "-an",
        "-vf",
        f"fps={FPS:g},scale={REVIEW_WIDTH}:-2:flags=lanczos",
        "-q:v",
        "3",
        "-start_number",
        "0",
        str(destination / "f_%06d.jpg"),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for video_eval prepare") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.strip()}") from exc


def _bridge_image(image: Image.Image, variant: str) -> Image.Image:
    if variant == "squash":
        return image.resize(BRIDGE_SIZE, Image.Resampling.BILINEAR)
    if variant == "letterbox":
        return ImageOps.pad(
            image,
            BRIDGE_SIZE,
            method=Image.Resampling.BILINEAR,
            color="black",
            centering=(0.5, 0.5),
        )
    raise ValueError(f"unknown bridge variant: {variant}")


def produce_bridge_frames(review_frames: list[Path], destination: Path, variant: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in review_frames:
        with Image.open(source) as image:
            bridge = _bridge_image(image.convert("RGB"), variant)
            bridge.save(destination / source.name, format="JPEG", quality=60)


def _write_clip_card(path: Path, clip_id: str, source: Path, probe: dict[str, Any]) -> None:
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raw = {"clip_id": clip_id, "recorded_at": None, "camera": None, "light": None, "script": []}
    raw["clip_id"] = clip_id
    raw["source"] = source.name
    raw["video"] = probe
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def prepare_video(
    video: Path,
    clip_id: str,
    *,
    root: Path | None = None,
    variant: str = "squash",
    force: bool = False,
    probe_fn: Callable[[Path], dict[str, Any]] = probe_video,
    extract_fn: Callable[[Path, Path], None] = extract_review_frames,
) -> dict[str, Any]:
    """Prepare one clip, returning a short status/count result."""
    started = time.monotonic()
    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    paths = EvalPaths.for_clip(clip_id, root)
    parameters = {"clip_id": clip_id, "source": str(video), "variant": variant, "fps": FPS}
    meta_path = paths.clip / "prepare.meta.json"
    bridge_dir = paths.bridge(variant)
    if (
        not force
        and paths.frames.exists()
        and bridge_dir.exists()
        and matching_meta(meta_path, parameters)
    ):
        return {"status": "skipped", "frames": len(read_manifest(paths.frames))}

    paths.clip.mkdir(parents=True, exist_ok=True)
    raw_dir = paths.root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_video = raw_dir / f"{clip_id}{video.suffix.lower()}"
    if raw_video.exists() and not force and not filecmp.cmp(video, raw_video, shallow=False):
        raise RuntimeError(
            f"raw clip {raw_video} already exists with different content; use --force to replace it"
        )
    if force or not raw_video.exists():
        shutil.copy2(video, raw_video)

    probe = probe_fn(raw_video)
    _write_clip_card(paths.clip / "clip.yaml", clip_id, raw_video, probe)
    if force and paths.review.exists():
        shutil.rmtree(paths.review)
    if force:
        for existing_variant in ("squash", "letterbox"):
            existing_bridge = paths.bridge(existing_variant)
            if existing_bridge.exists():
                shutil.rmtree(existing_bridge)
    if not paths.review.exists() or not any(paths.review.glob("f_*.jpg")):
        extract_fn(raw_video, paths.review)
    review_frames = sorted(paths.review.glob("f_*.jpg"))
    if not review_frames:
        raise RuntimeError("ffmpeg produced no review frames")
    if force and bridge_dir.exists():
        shutil.rmtree(bridge_dir)
    produce_bridge_frames(review_frames, bridge_dir, variant)

    records: list[dict[str, Any]] = []
    for frame_index, review in enumerate(review_frames):
        record = {
            "frame_index": frame_index,
            "t_s": frame_index / FPS,
            "review_path": str(review.relative_to(paths.root)),
        }
        squash_path = paths.bridge("squash") / review.name
        letterbox_path = paths.bridge("letterbox") / review.name
        if squash_path.exists():
            record["bridge_path"] = str(squash_path.relative_to(paths.root))
        if letterbox_path.exists():
            record["bridge_letterbox_path"] = str(letterbox_path.relative_to(paths.root))
        records.append(record)
    write_jsonl(paths.frames, records)
    write_meta(meta_path, command="prepare", parameters=parameters, started_at=started)
    update_index(paths.root, clip_id, f"prepare_{variant}", "complete")
    return {"status": "complete", "frames": len(records)}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    from video_eval.common import read_jsonl

    return read_jsonl(path)
