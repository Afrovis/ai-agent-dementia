"""Step 1 of every eval: zone the cameras before running perceive.

GMDCSA24 has no fixed camera per person, so clips are clustered into scenes
(same room and camera) by the median pixel difference of their first frames.
Per scene this traces the bed zone (perceive.calibrate_bed) and fits a ground
line (Theil-Sen, the same fit the tracker runs online) from ADL clips only, so
fall clips never leak into calibration. A tracker resets per clip and a 5 s
clip never collects the 20 samples it needs to learn a ground line online.

Writes <OUT>/zones/<scene>.yaml and <OUT>/scenes.json.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml
from perceive.backends import build_backend
from perceive.calibrate_bed import calibrate, yolo_segmenter
from perceive.classify import fit_ground_line
from perceive.main import PerceiveConfig, build_tracker
from PIL import Image, ImageFilter
from run_eval import OUT, Clip, extract_frames, load_clips, squash_jpeg
from perceive.zones import ZoneMap

SAME_SCENE_DIFF = 14.0  # median abs grey difference (0-255) below which two clips share a camera


def first_frame(video: Path) -> Image.Image:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "png", "-"],
        check=True, capture_output=True,
    )
    return Image.open(io.BytesIO(proc.stdout)).convert("RGB")


def signature(image: Image.Image) -> np.ndarray:
    small = image.convert("L").resize((64, 36)).filter(ImageFilter.GaussianBlur(1))
    return np.asarray(small, dtype=np.float32)


def cluster(clips: list[Clip]) -> tuple[dict[str, str], dict[str, Image.Image]]:
    reps: list[tuple[str, np.ndarray]] = []
    assign: dict[str, str] = {}
    frames: dict[str, Image.Image] = {}
    for clip in clips:
        image = first_frame(clip.video)
        sig = signature(image)
        frames[clip.key] = image
        for scene, rep in reps:
            if float(np.median(np.abs(sig - rep))) < SAME_SCENE_DIFF:
                assign[clip.key] = scene
                break
        else:
            scene = f"scene{len(reps) + 1:02d}"
            reps.append((scene, sig))
            assign[clip.key] = scene
    return assign, frames


def jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.resize((320, 240), Image.Resampling.BILINEAR).save(buf, "JPEG", quality=60)
    return buf.getvalue()


def ground_line_for(scene_clips: list[Clip], backend, config: PerceiveConfig):
    """Feed one tracker every ADL frame of the scene and fit the collected samples."""
    tracker = build_tracker(config)
    zones = ZoneMap()
    t_s = 0.0
    from perceive.classify import ground_zone_for_pose, zone_for_pose

    for clip in scene_clips:
        with tempfile.TemporaryDirectory() as tmp:
            for path in extract_frames(clip.video, Path(tmp)):
                pose = backend.detect(squash_jpeg(path))
                zone = zone_for_pose(zones, pose) if pose is not None else "other"
                ground = ground_zone_for_pose(zones, pose) if pose is not None else None
                tracker.update(pose, zone, t_s, ground_zone=ground)
                t_s += 0.5
        t_s += 10.0  # gap so per-clip motion never links across clips
    samples = list(tracker._ground_samples)
    return fit_ground_line(samples, min_samples=20), len(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-model", default="yolo11s-pose.pt")
    parser.add_argument("--seg-model", default="yolo11m-seg.pt")
    args = parser.parse_args()

    clips = load_clips()
    assign, frames = cluster(clips)
    (OUT / "zones").mkdir(parents=True, exist_ok=True)
    backend = build_backend("yolo", model_path=args.yolo_model, static_image_mode=True)
    config = PerceiveConfig.from_env()
    segmenter = yolo_segmenter(args.seg_model)

    scenes = {}
    for scene in sorted(set(assign.values())):
        members = [c for c in clips if assign[c.key] == scene]
        polygon = calibrate([jpeg(frames[c.key]) for c in members[:40]], segmenter)
        zones = {"bed": [[round(x, 4), round(y, 4)] for x, y in polygon]} if polygon else {}
        (OUT / "zones" / f"{scene}.yaml").write_text(yaml.safe_dump(zones, default_flow_style=None))
        adl = [c for c in members if c.kind == "ADL"]
        line, n = ground_line_for(adl, backend, config) if adl else (None, 0)
        scenes[scene] = {
            "clips": [c.key for c in members],
            "n_adl": len(adl),
            "bed": bool(polygon),
            "ground_line": list(line) if line else None,
            "ground_samples": n,
            "subjects": sorted({c.subject for c in members}),
        }
        print(scene, {k: v for k, v in scenes[scene].items() if k != "clips"}, len(members), flush=True)
    (OUT / "scenes.json").write_text(
        json.dumps({"assign": assign, "scenes": scenes}, indent=2)
    )
    for scene, image in {assign[k]: frames[k] for k in assign}.items():
        image.resize((480, 270)).save(OUT / "zones" / f"{scene}.jpg")


if __name__ == "__main__":
    main()
