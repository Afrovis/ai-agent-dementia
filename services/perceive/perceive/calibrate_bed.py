"""Draw the bed zone from the camera image with a segmentation model.

A hand-drawn or guessed bed rectangle is the largest single source of bed
occupancy errors on the 2026-09-13/14 bedroom clips: it covered the wall
above the headboard and the floor in front of the bed, and missed the foot
of the mattress. The bed does not move, so this runs once at setup (and
again whenever the camera or the bed moves), not per frame: a COCO
instance-segmentation model finds the bed in every frame it can, the masks
vote, and the area most frames agree on becomes the `bed` polygon in
`zones.yaml`. The other zones in that file are left untouched.

Voting across frames is what lets this run while someone is in the room: a
person or blanket hides part of the bed in some frames, never the same part
in all of them. An empty, made bed still gives the cleanest outline.

    python -m perceive.calibrate_bed --zones /app/config/zones.yaml \
        --redis redis://bus:6379 --count 40
    python -m perceive.calibrate_bed --zones zones.yaml --frames-dir frames/

Restart `perceive` afterwards; it only reads `zones.yaml` at startup.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import yaml

from perceive.calibrate_phantoms import _read_frames_dir, _read_redis_frames

BED_CLASS = 59
"""`bed` in the COCO label set every stock ultralytics `-seg` model uses."""

DEFAULT_MODEL = "yolo11m-seg.pt"
DEFAULT_GROW_UP = 0.15
"""Chosen on the four 2026-09-13/14 clips (0.0, 0.15 and 0.30 tried): 0.15
recovered people lying or sitting on the bed that the bare mattress outline
missed, while 0.30 started to take in a chair beside the bed."""
GRID_WIDTH = 320
"""Masks are voted on a grid this wide (height follows the frame's aspect
ratio). A zone polygon needs nowhere near full resolution."""

Point = tuple[float, float]
Segmenter = Callable[[bytes], "np.ndarray | None"]
"""Maps one JPEG to a boolean bed mask on the voting grid, or `None` when no
bed was found in that frame."""


def vote(masks: Sequence[np.ndarray], min_fraction: float) -> np.ndarray:
    """Pixels marked bed in at least `min_fraction` of `masks`."""
    stacked = np.stack([m.astype(bool) for m in masks])
    return stacked.mean(axis=0) >= min_fraction


def mask_to_polygon(mask: np.ndarray, rows: int = 16) -> list[Point]:
    """Outline `mask` as a normalised polygon: the leftmost and rightmost bed
    pixel on `rows` evenly spaced rows, walked down the left edge and back up
    the right. A bed seen from a wall-mounted camera has no holes or
    overhangs along a row, so this row scan keeps its shape without pulling in
    OpenCV. Returns `[]` for an empty mask."""
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return []
    height, width = mask.shape
    left: list[Point] = []
    right: list[Point] = []
    for y in np.unique(np.linspace(ys[0], ys[-1], num=max(2, rows)).round().astype(int)):
        xs = np.flatnonzero(mask[y])
        if xs.size == 0:
            continue
        left.append((xs[0] / width, y / height))
        right.append(((xs[-1] + 1) / width, y / height))
    if left:
        # Close the bottom edge on the last bed row rather than one row short.
        left[-1] = (left[-1][0], (ys[-1] + 1) / height)
        right[-1] = (right[-1][0], (ys[-1] + 1) / height)
    return _simplify(left + right[::-1])


def _simplify(points: list[Point], tolerance: float = 0.01) -> list[Point]:
    """Ramer-Douglas-Peucker on the closed outline: keep only the points that
    sit more than `tolerance` (frame fraction) off the line through the
    points kept around them. Judging each point against its immediate
    neighbours instead collapses a gently curving bed edge entirely."""
    if len(points) <= 3:
        kept = points
    else:
        start = points[0]
        far = max(range(len(points)), key=lambda i: math.dist(start, points[i]))
        kept = (
            _rdp(points[: far + 1], tolerance)[:-1] + _rdp(points[far:] + [start], tolerance)[:-1]
        )
    # Plain floats: numpy scalars from the mask indices are not YAML-safe.
    return [(round(float(x), 4), round(float(y), 4)) for x, y in kept]


def _rdp(points: list[Point], tolerance: float) -> list[Point]:
    (x0, y0), (x1, y1) = points[0], points[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    best, best_distance = 0, 0.0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        if length:
            distance = abs((x1 - x0) * (y0 - py) - (x0 - px) * (y1 - y0)) / length
        else:
            distance = math.hypot(px - x0, py - y0)
        if distance > best_distance:
            best, best_distance = i, distance
    if best_distance <= tolerance:
        return [points[0], points[-1]]
    return _rdp(points[: best + 1], tolerance)[:-1] + _rdp(points[best:], tolerance)


def grow_upward(polygon: Sequence[Point], fraction: float) -> list[Point]:
    """Stretch `polygon` up the frame by `fraction` of its own height, keeping
    its bottom edge in place. The mask covers the mattress surface, but a
    person lying or sitting on it rises above that surface in the image, so
    their centroid and box centre sit above the mask's top edge."""
    if not polygon or fraction <= 0.0:
        return list(polygon)
    bottom = max(y for _, y in polygon)
    return [(x, round(max(0.0, bottom - (bottom - y) * (1.0 + fraction)), 4)) for x, y in polygon]


def calibrate(
    jpegs: Sequence[bytes],
    segmenter: Segmenter,
    *,
    min_frame_fraction: float = 0.5,
    grow_up: float = DEFAULT_GROW_UP,
) -> list[Point]:
    """The bed polygon agreed on by the frames in which `segmenter` found a
    bed, stretched upward by `grow_up`, or `[]` if no frame had a bed."""
    masks = [mask for jpeg in jpegs if (mask := segmenter(jpeg)) is not None]
    if not masks:
        return []
    return grow_upward(mask_to_polygon(vote(masks, min_frame_fraction)), grow_up)


def yolo_segmenter(model_path: str = DEFAULT_MODEL, conf: float = 0.25) -> Segmenter:
    """A `Segmenter` backed by an ultralytics `-seg` model, taking the most
    confident bed in each frame. The model downloads on first use."""
    from PIL import Image, ImageDraw
    from ultralytics import YOLO

    model = YOLO(model_path)

    def segment(jpeg: bytes) -> np.ndarray | None:
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        result = model.predict(image, classes=[BED_CLASS], conf=conf, verbose=False)[0]
        if result.masks is None or len(result.masks) == 0:
            return None
        best = int(result.boxes.conf.argmax())
        outline = result.masks.xyn[best]
        grid_height = max(1, round(GRID_WIDTH * image.height / image.width))
        canvas = Image.new("1", (GRID_WIDTH, grid_height), 0)
        ImageDraw.Draw(canvas).polygon(
            [(float(x) * GRID_WIDTH, float(y) * grid_height) for x, y in outline], fill=1
        )
        return np.array(canvas, dtype=bool)

    return segment


def write_bed_zone(path: Path, polygon: Sequence[Point]) -> None:
    """Set `bed` in the zones file at `path`, keeping every other zone, and
    replace the file atomically the way the dashboard's zone editor does."""
    zones = {}
    if path.exists():
        zones = yaml.safe_load(path.read_text()) or {}
        if not isinstance(zones, dict):
            raise ValueError(f"{path} does not contain a mapping of zone names to polygons")
    zones["bed"] = [[x, y] for x, y in polygon]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(zones, handle, default_flow_style=None, sort_keys=False)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw the bed zone in zones.yaml from a bed segmentation of camera frames."
    )
    parser.add_argument("--zones", required=True, help="zones YAML to write the bed polygon into")
    parser.add_argument(
        "--frames-dir", help="directory of JPEG/PNG frames (offline mode, e.g. video_eval)"
    )
    parser.add_argument("--redis", help="Redis URL, e.g. redis://bus:6379 (live mode)")
    parser.add_argument("--count", type=int, default=40, help="frames to read in live mode")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"ultralytics -seg model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--min-frame-fraction",
        type=float,
        default=0.5,
        help="fraction of bed-finding frames that must agree on a pixel (default: 0.5)",
    )
    parser.add_argument(
        "--grow-up",
        type=float,
        default=DEFAULT_GROW_UP,
        help=f"stretch the polygon up by this fraction of its height (default: {DEFAULT_GROW_UP})",
    )
    args = parser.parse_args(argv)

    if bool(args.frames_dir) == bool(args.redis):
        parser.error("pass exactly one of --frames-dir or --redis")

    jpegs = (
        _read_frames_dir(Path(args.frames_dir))
        if args.frames_dir
        else _read_redis_frames(args.redis, args.count)
    )
    if not jpegs:
        print("no frames read; nothing to calibrate", file=sys.stderr)
        return 1

    polygon = calibrate(
        jpegs,
        yolo_segmenter(args.model),
        min_frame_fraction=args.min_frame_fraction,
        grow_up=args.grow_up,
    )
    if not polygon:
        print(f"no bed found in {len(jpegs)} frames; {args.zones} left unchanged", file=sys.stderr)
        return 1

    write_bed_zone(Path(args.zones), polygon)
    print(f"read {len(jpegs)} frames, bed polygon with {len(polygon)} points:")
    print(f"  bed={[list(p) for p in polygon]}")
    print(f"wrote {args.zones}; restart perceive to use it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
