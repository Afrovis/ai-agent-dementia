"""Render private, annotated evaluation videos from prepared clip artifacts.

The renderer deliberately distinguishes the three available sources of truth:
the recorder's coarse scenario card, the real-time pose pipeline output, and
the local vision-language model's whole-frame semantic labels.  It never adds
audio to derived videos.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image, ImageDraw, ImageFont

from video_eval.common import read_jsonl, write_meta
from video_eval.paths import EvalPaths

VisualizationMode = Literal["manual", "pipeline", "vision"]

CANVAS = (1280, 720)
FPS = 2
INK = (230, 237, 246)
MUTED = (145, 158, 176)
PANEL = (15, 24, 39)
PANEL_2 = (22, 34, 52)
ACCENTS = {
    "manual": (255, 193, 92),
    "pipeline": (50, 213, 181),
    "vision": (139, 124, 246),
}
STATE_COLORS = {
    "in_bed": (74, 222, 128),
    "sitting_up": (250, 204, 82),
    "standing": (71, 170, 255),
    "walking": (30, 211, 238),
    "upright": (71, 170, 255),
    "on_floor": (255, 95, 95),
    "absent": (173, 184, 201),
}
COCO_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True)
class RenderInputs:
    root: Path
    clip_dir: Path
    clip: dict[str, Any]
    frames: list[dict[str, Any]]
    manual: list[dict[str, Any]]
    pipeline: list[dict[str, Any]]
    poses: list[dict[str, Any]]
    vision: list[dict[str, Any]]
    zones: dict[str, Any]
    pipeline_tag: str
    pipeline_backend: str


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/SFNSRounded.ttf" if bold else "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _load_inputs(root: Path, clip_id: str, pipeline_tag: str | None) -> RenderInputs:
    clip_dir = root / "clips" / clip_id
    clip = yaml.safe_load((clip_dir / "clip.yaml").read_text(encoding="utf-8"))
    frames = read_jsonl(clip_dir / "frames.jsonl")
    manual = sorted(clip.get("script") or [], key=lambda item: float(item["t_s"]))
    vision_path = clip_dir / "labels" / "local.jsonl"
    vision = read_jsonl(vision_path) if vision_path.exists() else []

    if pipeline_tag is None:
        candidates = sorted(
            (clip_dir / "predictions").glob("yolo_yolo11s-pose-letterbox-*-g.jsonl")
        )
        if not candidates:
            raise FileNotFoundError(f"no gated YOLO11s-pose letterbox prediction for {clip_id}")
        prediction_path = candidates[-1]
        pipeline_tag = prediction_path.stem
    else:
        prediction_path = clip_dir / "predictions" / f"{pipeline_tag}.jsonl"
    pipeline = read_jsonl(prediction_path)
    prediction_meta_path = prediction_path.with_suffix(".meta.json")
    prediction_meta = json.loads(prediction_meta_path.read_text(encoding="utf-8"))
    prediction_parameters = prediction_meta.get("parameters") or {}
    backend_name = str(prediction_parameters.get("backend") or "unknown")
    model_path = prediction_parameters.get("yolo_model")
    pipeline_backend = Path(model_path).stem if model_path else backend_name
    cache_name = f"{pipeline_backend}-letterbox.jsonl"
    cache_path = clip_dir / "cache" / cache_name
    # New predictions carry the canonical pose geometry needed by their own
    # visualization.  Keep the old raw-detection cache as a compatibility
    # source for predictions produced before that field existed.
    poses = read_jsonl(cache_path) if cache_path.exists() else pipeline
    zones_path = clip_dir / "zones.yaml"
    zones = yaml.safe_load(zones_path.read_text(encoding="utf-8")) if zones_path.exists() else {}

    count = len(frames)
    for name, records in (("pipeline", pipeline), ("pose data", poses)):
        if len(records) != count:
            raise ValueError(f"{name} has {len(records)} records; frames has {count}")
    return RenderInputs(
        root,
        clip_dir,
        clip,
        frames,
        manual,
        pipeline,
        poses,
        vision,
        zones,
        pipeline_tag,
        pipeline_backend,
    )


def _current_manual(events: list[dict[str, Any]], t_s: float) -> dict[str, Any] | None:
    current = None
    for event in events:
        if float(event["t_s"]) > t_s:
            break
        current = event
    return current


def _manual_state(action: str | None) -> str | None:
    if action is None:
        return None
    if action in {"in_bed", "in_bed_above_blanket", "under_blanket_in_bed", "lay_down_on_bed"}:
        return "in_bed"
    if action in {"sitting_up", "sitting", "sitting_on_chair", "sit_on_bed"}:
        return "sitting_up"
    if action in {
        "floor",
        "sitting_on_floor",
        "sit_on_floor",
        "lay_down_on_floor",
        "lay_down_on_ground",
    }:
        return "on_floor"
    if action in {"out_of_frame", "leave_room", "leave_frame"}:
        return "absent"
    return "upright"


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", size, (0, 0, 0))
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return result


def _runtime_to_source(point: tuple[float, float]) -> tuple[float, float]:
    """Map normalized 4:3 runtime coordinates through its 16:9 letterbox."""
    x, y = point
    return x, (y - 0.125) / 0.75


def _point(rect: tuple[int, int, int, int], normalized: tuple[float, float]) -> tuple[int, int]:
    x, y, width, height = rect
    return round(x + normalized[0] * width), round(y + normalized[1] * height)


def _draw_pose(
    draw: ImageDraw.ImageDraw,
    pose_record: dict[str, Any],
    rect: tuple[int, int, int, int],
    *,
    source_mapping: bool,
) -> None:
    detections = pose_record.get("dets") or []
    if detections:
        detection = max(detections, key=lambda item: float(item.get("conf", 0.0)))
        keypoints = detection.get("kp") or []
        edges = COCO_EDGES
    else:
        bbox = pose_record.get("bbox")
        landmarks = pose_record.get("landmarks") or {}
        if not bbox:
            return
        names = (
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )
        keypoints = [landmarks.get(name, [0.0, 0.0, 0.0]) for name in names]
        edges = (
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (5, 7),
            (4, 6),
            (6, 8),
        )
        detection = {"bbox": bbox}
    mapped: list[tuple[int, int] | None] = []
    for raw_x, raw_y, visibility in keypoints:
        normalized = (float(raw_x), float(raw_y))
        if source_mapping:
            normalized = _runtime_to_source(normalized)
        mapped.append(_point(rect, normalized) if visibility >= 0.3 else None)
    for first, second in edges:
        if first < len(mapped) and second < len(mapped) and mapped[first] and mapped[second]:
            draw.line((mapped[first], mapped[second]), fill=(39, 236, 208), width=4)
    for value in mapped:
        if value:
            draw.ellipse(
                (value[0] - 4, value[1] - 4, value[0] + 4, value[1] + 4),
                fill=(255, 255, 255),
                outline=(14, 116, 144),
                width=2,
            )

    x1, y1, x2, y2 = detection["bbox"]
    if source_mapping:
        x1, y1 = _runtime_to_source((x1, y1))
        x2, y2 = _runtime_to_source((x2, y2))
    left, top = _point(rect, (x1, y1))
    right, bottom = _point(rect, (x2, y2))
    draw.rectangle((left, top, right, bottom), outline=(39, 236, 208), width=3)


def _draw_zone(
    draw: ImageDraw.ImageDraw,
    points: list[list[float]],
    rect: tuple[int, int, int, int],
    *,
    source_mapping: bool,
) -> None:
    mapped = []
    for raw in points:
        normalized = (float(raw[0]), float(raw[1]))
        if source_mapping:
            normalized = _runtime_to_source(normalized)
        mapped.append(_point(rect, normalized))
    if len(mapped) >= 3:
        draw.line(mapped + [mapped[0]], fill=(255, 196, 79), width=3)
        draw.text(
            (mapped[0][0] + 7, mapped[0][1] + 5),
            "BED ZONE",
            font=_font(15, bold=True),
            fill=(255, 213, 111),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    *,
    color: tuple[int, int, int] = INK,
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=_font(size, bold=bold), fill=color)


def _wrap(draw: ImageDraw.ImageDraw, value: str, width: int, size: int) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    font = _font(size)
    for word in words:
        proposed = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), proposed, font=font)[2] > width:
            lines.append(current)
            current = word
        else:
            current = proposed
    if current:
        lines.append(current)
    return lines


def _timeline_values(inputs: RenderInputs, mode: VisualizationMode) -> list[str | None]:
    if mode == "manual":
        return [
            _manual_state((_current_manual(inputs.manual, float(frame["t_s"])) or {}).get("action"))
            for frame in inputs.frames
        ]
    records = inputs.pipeline if mode == "pipeline" else inputs.vision
    key = "state" if mode == "pipeline" else "posture"
    return [record.get(key) for record in records]


def _draw_timeline(
    draw: ImageDraw.ImageDraw, values: list[str | None], index: int, duration_s: float
) -> None:
    x, y, width, height = 24, 636, 1232, 54
    draw.rounded_rectangle((x, y, x + width, y + height), radius=9, fill=PANEL)
    bar_y, bar_h = y + 24, 12
    for pixel in range(width):
        record_index = min(len(values) - 1, math.floor(pixel / width * len(values)))
        state = values[record_index]
        draw.line(
            (x + pixel, bar_y, x + pixel, bar_y + bar_h),
            fill=STATE_COLORS.get(state or "", (73, 83, 101)),
        )
    marker_x = x + round(index / max(1, len(values) - 1) * width)
    draw.line((marker_x, bar_y - 5, marker_x, bar_y + bar_h + 5), fill=(255, 255, 255), width=3)
    _text(draw, (x + 10, y + 4), "0:00", 13, color=MUTED)
    end = f"{int(duration_s) // 60}:{int(duration_s) % 60:02d}"
    end_width = draw.textbbox((0, 0), end, font=_font(13))[2]
    _text(draw, (x + width - end_width - 10, y + 4), end, 13, color=MUTED)


def _render_frame(inputs: RenderInputs, mode: VisualizationMode, index: int) -> Image.Image:
    item = inputs.frames[index]
    t_s = float(item["t_s"])
    source_path = inputs.root / item["review_path"]
    runtime_path = inputs.root / item["bridge_letterbox_path"]
    source = _fit(Image.open(source_path).convert("RGB"), (900, 506))
    runtime = _fit(Image.open(runtime_path).convert("RGB"), (288, 216))
    canvas = Image.new("RGB", CANVAS, (7, 13, 24))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (948, 92, 1256, 598), radius=10, fill=PANEL, outline=(49, 64, 84), width=2
    )
    canvas.paste(source, (24, 92))
    canvas.paste(runtime, (958, 122))
    accent = ACCENTS[mode]

    titles = {
        "manual": "MANUAL ANNOTATION",
        "pipeline": "REAL-TIME POSE PIPELINE",
        "vision": "LOCAL VISION SYSTEM",
    }
    subtitles = {
        "manual": "Recorder scenario card • coarse action timestamps",
        "pipeline": f"{inputs.pipeline_backend} → zone → rules → temporal tracker",
        "vision": "Qwen3-VL:8b • semantic whole-frame classification",
    }
    _text(draw, (24, 20), titles[mode], 29, color=accent, bold=True)
    _text(draw, (24, 55), subtitles[mode], 17, color=MUTED)
    clock = f"{int(t_s) // 60:02d}:{t_s % 60:04.1f}"
    _text(draw, (1150, 27), clock, 22, bold=True)
    draw.rectangle((24, 92, 924, 598), outline=(49, 64, 84), width=2)
    _text(draw, (958, 99), "CAMERA INPUT • 320×240", 14, color=MUTED, bold=True)
    draw.rectangle((958, 122, 1246, 338), outline=(71, 86, 108), width=2)
    _text(draw, (966, 313), "letterboxed", 12, color=(186, 197, 213), bold=True)

    main_rect = (24, 92, 900, 506)
    inset_rect = (958, 122, 288, 216)
    if mode == "pipeline":
        if inputs.zones.get("bed"):
            _draw_zone(draw, inputs.zones["bed"], main_rect, source_mapping=True)
            _draw_zone(draw, inputs.zones["bed"], inset_rect, source_mapping=False)
        _draw_pose(draw, inputs.poses[index], main_rect, source_mapping=True)
        _draw_pose(draw, inputs.poses[index], inset_rect, source_mapping=False)
    elif mode == "vision":
        draw.rectangle((30, 98, 918, 592), outline=accent, width=4)
        _text(draw, (39, 106), "WHOLE FRAME → LOCAL VLM", 15, color=accent, bold=True)

    panel_x = 966
    if mode == "manual":
        event = _current_manual(inputs.manual, t_s)
        action = event.get("action") if event else None
        state = _manual_state(action)
        _text(draw, (panel_x, 357), "RECORDER MARKER", 14, color=MUTED, bold=True)
        _text(
            draw,
            (panel_x, 381),
            (action or "not yet marked").replace("_", " ").upper(),
            22,
            color=STATE_COLORS.get(state or "", accent),
            bold=True,
        )
        if event:
            _text(draw, (panel_x, 412), f"marker at {float(event['t_s']):.1f} s", 15, color=MUTED)
        _text(draw, (panel_x, 449), "COARSE STATE MAP", 14, color=MUTED, bold=True)
        _text(
            draw,
            (panel_x, 473),
            (state or "unlabelled").replace("_", " ").upper(),
            21,
            color=STATE_COLORS.get(state or "", MUTED),
            bold=True,
        )
        note = "Markers describe intended actions; they are not frame-exact pose labels."
    elif mode == "pipeline":
        record = inputs.pipeline[index]
        state = record.get("state")
        detected = bool(record.get("detected"))
        _text(draw, (panel_x, 357), "LIVE DECISION PATH", 14, color=MUTED, bold=True)
        _text(
            draw,
            (panel_x, 382),
            f"1  DETECT   {'person' if detected else 'no person'}",
            16,
            color=INK,
        )
        confidence = record.get("detect_confidence")
        confidence_text = f"{float(confidence):.2f}" if confidence is not None else "—"
        _text(draw, (panel_x, 407), f"2  POSE     confidence {confidence_text}", 16, color=INK)
        _text(draw, (panel_x, 432), f"3  ZONE     {record.get('zone') or '—'}", 16, color=INK)
        gate = "dropped" if record.get("gated") else "admitted"
        _text(draw, (panel_x, 457), f"4  GATE     {gate}", 16, color=INK)
        emitted = "yes" if record.get("published") else "no / held"
        _text(draw, (panel_x, 482), f"5  EMIT     {emitted}", 16, color=INK)
        state_label = (state or "tracker warming").replace("_", " ").upper()
        _text(
            draw,
            (panel_x, 521),
            state_label,
            23,
            color=STATE_COLORS.get(state or "", accent),
            bold=True,
        )
        note = f"Backend {record.get('backend_ms', 0):.1f} ms • tag {inputs.pipeline_tag}"
    else:
        record = inputs.vision[index]
        state = record.get("posture")
        _text(draw, (panel_x, 357), "SEMANTIC LABEL", 14, color=MUTED, bold=True)
        _text(
            draw,
            (panel_x, 382),
            (state or "invalid").replace("_", " ").upper(),
            23,
            color=STATE_COLORS.get(state or "", accent),
            bold=True,
        )
        _text(draw, (panel_x, 418), f"location   {record.get('location') or '—'}", 16, color=INK)
        _text(
            draw,
            (panel_x, 444),
            f"confidence {float(record.get('confidence') or 0):.2f}",
            16,
            color=INK,
        )
        _text(draw, (panel_x, 476), "MODEL NOTE", 13, color=MUTED, bold=True)
        note_lines = _wrap(draw, str(record.get("note") or "No note"), 264, 15)[:3]
        for line_index, line in enumerate(note_lines):
            _text(draw, (panel_x, 498 + line_index * 20), line, 15, color=INK)
        note = "No pose geometry is claimed: this model labels the complete frame."

    note_lines = _wrap(draw, note, 880 if mode != "pipeline" else 270, 13)
    if mode in {"manual", "vision"}:
        for line_index, line in enumerate(note_lines[:2]):
            _text(draw, (38, 605 + line_index * 16), line, 13, color=MUTED)
    else:
        for line_index, line in enumerate(note_lines[:2]):
            _text(draw, (panel_x, 561 + line_index * 16), line, 12, color=MUTED)

    duration_s = float(inputs.clip["video"]["duration_s"])
    _draw_timeline(draw, _timeline_values(inputs, mode), index, duration_s)
    return canvas


def render_visualization(
    clip_id: str,
    mode: VisualizationMode,
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
    pipeline_tag: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Render one silent H.264 visualization and return its provenance summary."""
    started_at = time.monotonic()
    root = EvalPaths.for_clip(clip_id, root).root
    inputs = _load_inputs(root, clip_id, pipeline_tag)
    if mode == "vision" and len(inputs.vision) != len(inputs.frames):
        raise FileNotFoundError(
            f"local vision labels are incomplete for {clip_id}; run label-local first"
        )
    output_dir = output_dir or root / "analysis"
    output = output_dir / f"{clip_id}__{mode}.mp4"
    meta_path = output.with_suffix(".meta.json")
    parameters = {
        "clip_id": clip_id,
        "mode": mode,
        "fps": FPS,
        "canvas": list(CANVAS),
        "pipeline_tag": inputs.pipeline_tag,
        "audio": False,
    }
    if output.exists() and not force:
        return {"output": str(output), "skipped": True, **parameters}

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1280x720",
        "-framerate",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index in range(len(inputs.frames)):
            process.stdin.write(_render_frame(inputs, mode, index).tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

    write_meta(
        meta_path,
        command="visualize",
        parameters=parameters,
        started_at=started_at,
        versions=("pillow", "pyyaml"),
    )
    return {
        "output": str(output),
        "frames": len(inputs.frames),
        "duration_s": len(inputs.frames) / FPS,
        "skipped": False,
        **parameters,
    }
