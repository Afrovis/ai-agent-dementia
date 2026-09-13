"""Fail-closed head blurring and privacy-safe contact sheets."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from video_eval.common import (
    git_sha,
    matching_meta,
    read_jsonl,
    update_index,
    write_jsonl,
    write_meta,
)
from video_eval.paths import EvalPaths

Box = tuple[int, int, int, int]
SHEET_TILE_SIZE = (426, 240)


@dataclass(frozen=True)
class PersonDetection:
    bbox: Box
    nose: tuple[float, float, float] | None = None
    left_shoulder: tuple[float, float, float] | None = None
    right_shoulder: tuple[float, float, float] | None = None


class PersonDetector(Protocol):
    def detect(self, image: Image.Image) -> list[PersonDetection]: ...


class FaceDetector(Protocol):
    def detect(self, image: Image.Image) -> list[Box]: ...


class YoloPersonDetector:
    """All-person YOLO pose adapter; unlike perceive, blur must cover everyone."""

    def __init__(self, model_path: str = "yolov8n-pose.pt") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "video_eval blur requires ultralytics (the perceive yolo extra)"
            ) from exc
        self._model = YOLO(model_path)

    def detect(self, image: Image.Image) -> list[PersonDetection]:
        results = self._model(image, verbose=False)
        if not results or results[0].boxes is None:
            return []
        result = results[0]
        boxes = result.boxes.xyxy.tolist()
        xy = result.keypoints.xy.tolist() if result.keypoints is not None else []
        conf = (
            result.keypoints.conf.tolist()
            if result.keypoints is not None and result.keypoints.conf is not None
            else []
        )
        detections: list[PersonDetection] = []
        for index, raw_box in enumerate(boxes):
            points = xy[index] if index < len(xy) else []
            scores = conf[index] if index < len(conf) else []

            def point(point_index: int) -> tuple[float, float, float] | None:
                if point_index >= len(points):
                    return None
                visibility = float(scores[point_index]) if point_index < len(scores) else 1.0
                return float(points[point_index][0]), float(points[point_index][1]), visibility

            detections.append(
                PersonDetection(
                    bbox=tuple(round(float(value)) for value in raw_box),
                    nose=point(0),
                    left_shoulder=point(5),
                    right_shoulder=point(6),
                )
            )
        return detections


class MediaPipeFaceDetector:
    """Full-range MediaPipe face detector used for both blur and verification."""

    def __init__(self, min_confidence: float = 0.3) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("video_eval blur requires mediapipe<1.0") from exc
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_confidence
        )

    def detect(self, image: Image.Image) -> list[Box]:
        import numpy as np

        width, height = image.size
        result = self._detector.process(np.asarray(image.convert("RGB")))
        boxes: list[Box] = []
        for detection in result.detections or []:
            raw = detection.location_data.relative_bounding_box
            boxes.append(
                _clip_box(
                    (
                        round(raw.xmin * width),
                        round(raw.ymin * height),
                        round((raw.xmin + raw.width) * width),
                        round((raw.ymin + raw.height) * height),
                    ),
                    image.size,
                )
            )
        return boxes

    def close(self) -> None:
        self._detector.close()


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _clip_box(box: Box, size: tuple[int, int]) -> Box:
    width, height = size
    left, top, right, bottom = box
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def _pad_box(box: Box, fraction: float, size: tuple[int, int]) -> Box:
    left, top, right, bottom = box
    x_pad = round((right - left) * fraction)
    y_pad = round((bottom - top) * fraction)
    return _clip_box((left - x_pad, top - y_pad, right + x_pad, bottom + y_pad), size)


def _union(boxes: list[Box], size: tuple[int, int]) -> Box:
    return _clip_box(
        (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ),
        size,
    )


def _head_region(person: PersonDetection, face_boxes: list[Box], size: tuple[int, int]) -> Box:
    bbox = _clip_box(person.bbox, size)
    points = (person.nose, person.left_shoulder, person.right_shoulder)
    if not any(point is not None and point[2] > 0.3 for point in points):
        return bbox

    left, top, right, bottom = bbox
    regions = [(left, top, right, top + max(1, round((bottom - top) * 0.25)))]
    nose = person.nose
    if nose is not None and nose[2] > 0.3:
        shoulders = (person.left_shoulder, person.right_shoulder)
        if all(point is not None and point[2] > 0.3 for point in shoulders):
            assert shoulders[0] is not None and shoulders[1] is not None
            shoulder_width = math.dist(shoulders[0][:2], shoulders[1][:2])
        else:
            shoulder_width = (right - left) * 0.25
        radius = max(8, round(1.2 * shoulder_width))
        regions.append(
            (
                round(nose[0] - radius),
                round(nose[1] - radius),
                round(nose[0] + radius),
                round(nose[1] + radius),
            )
        )
    for face in face_boxes:
        if face[2] >= left and face[0] <= right and face[3] >= top and face[1] <= bottom:
            regions.append(_pad_box(face, 0.6, size))
    return _union(regions, size)


def _obscure(image: Image.Image, box: Box, *, strength: int = 1) -> None:
    box = _clip_box(box, image.size)
    region = image.crop(box)
    sigma = max(region.size) * 0.1 * strength
    region = region.filter(ImageFilter.GaussianBlur(radius=sigma))
    cell = max(8, 8 * strength)
    tiny = (max(1, region.width // cell), max(1, region.height // cell))
    region = region.resize(tiny, Image.Resampling.BILINEAR).resize(
        region.size, Image.Resampling.NEAREST
    )
    image.paste(region, box)


def _faces_at_two_scales(image: Image.Image, detector: FaceDetector) -> list[Box]:
    faces = detector.detect(image)
    doubled = image.resize((image.width * 2, image.height * 2), Image.Resampling.BILINEAR)
    faces.extend(detector.detect(doubled))
    return faces


def blur_image(
    image: Image.Image,
    *,
    people: PersonDetector,
    faces: FaceDetector,
    local_person_visible: bool | None = None,
) -> tuple[Image.Image, str, bool]:
    """Blur a frame and independently verify it; uncertainty becomes full-frame blur."""
    result = image.convert("RGB").copy()
    detections = people.detect(result)
    face_boxes = faces.detect(result)
    if detections:
        full_person = False
        for person in detections:
            invisible = not any(
                point is not None and point[2] > 0.3
                for point in (person.nose, person.left_shoulder, person.right_shoulder)
            )
            full_person = full_person or invisible
            _obscure(result, _head_region(person, face_boxes, result.size))
        mode = "person" if full_person else "head"
    elif local_person_visible:
        _obscure(result, (0, 0, result.width, result.height))
        mode = "full"
    elif face_boxes:
        for face in face_boxes:
            _obscure(result, _pad_box(face, 0.6, result.size))
        mode = "head"
    elif local_person_visible is None:
        # With neither a pose detection nor a local VLM opinion, absence is
        # unproven. Keep the frame local instead of treating a model miss as
        # permission to upload an unblurred bedroom frame.
        return result, "unknown", False
    else:
        mode = "none"

    for attempt in range(3):
        if not _faces_at_two_scales(result, faces):
            return result, mode, True
        # A surviving face makes the entire frame sensitive. Escalating to a
        # full-frame blur is intentionally conservative and cannot miss a box.
        _obscure(result, (0, 0, result.width, result.height), strength=attempt + 2)
        mode = "full"
    return result, mode, False


def blur_clip(
    clip_id: str,
    *,
    root: Path | None = None,
    force: bool = False,
    person_factory: Callable[[], PersonDetector] = YoloPersonDetector,
    face_factory: Callable[[], FaceDetector] = MediaPipeFaceDetector,
) -> dict[str, object]:
    started = time.monotonic()
    paths = EvalPaths.for_clip(clip_id, root)
    output = paths.clip / "blur.jsonl"
    meta_path = paths.clip / "blur.meta.json"
    local_path = paths.labels / "local.jsonl"
    parameters = {
        "clip_id": clip_id,
        "face_detector": "mediapipe-full-range",
        "local_labels_sha256": _digest(local_path),
        "pose": "yolov8n",
    }
    if not force and output.exists() and matching_meta(meta_path, parameters):
        records = read_jsonl(output)
        return {
            "status": "skipped",
            "frames": len(records),
            "upload_ok": sum(bool(row["upload_ok"]) for row in records),
        }
    frames = read_jsonl(paths.frames)
    local = (
        {int(row["frame_index"]): row for row in read_jsonl(local_path)}
        if local_path.exists()
        else {}
    )
    if force and paths.blurred.exists():
        shutil.rmtree(paths.blurred)
    paths.blurred.mkdir(parents=True, exist_ok=True)
    people = person_factory()
    faces = face_factory()
    records: list[dict[str, object]] = []
    try:
        for frame in frames:
            source = paths.root / frame["review_path"]
            with Image.open(source) as image:
                blurred, mode, upload_ok = blur_image(
                    image,
                    people=people,
                    faces=faces,
                    local_person_visible=local.get(int(frame["frame_index"]), {}).get(
                        "person_visible"
                    ),
                )
                destination = paths.blurred / source.name
                blurred.save(destination, format="JPEG", quality=85)
            records.append(
                {
                    "blur_mode": mode,
                    "blurred_path": str(destination.relative_to(paths.root)),
                    "frame_index": int(frame["frame_index"]),
                    "t_s": float(frame["t_s"]),
                    "upload_ok": upload_ok,
                }
            )
    finally:
        for detector in (people, faces):
            close = getattr(detector, "close", None)
            if close is not None:
                close()
    write_jsonl(output, records)
    write_meta(
        meta_path,
        command="blur",
        parameters=parameters,
        started_at=started,
        versions=("pillow", "mediapipe", "ultralytics"),
    )
    upload_ok_count = sum(bool(row["upload_ok"]) for row in records)
    update_index(paths.root, clip_id, "blur", "complete")
    return {"status": "complete", "frames": len(records), "upload_ok": upload_ok_count}


def _sheet_tile(image: Image.Image, label: str) -> Image.Image:
    tile = Image.new("RGB", SHEET_TILE_SIZE, "black")
    fitted = ImageOps.contain(image.convert("RGB"), (SHEET_TILE_SIZE[0], SHEET_TILE_SIZE[1] - 24))
    tile.paste(fitted, ((SHEET_TILE_SIZE[0] - fitted.width) // 2, 0))
    ImageDraw.Draw(tile).text((8, SHEET_TILE_SIZE[1] - 20), label, fill="white")
    return tile


def build_sheets(
    clip_id: str, *, root: Path | None = None, force: bool = False, confirm_reviewed: bool = False
) -> dict[str, object]:
    started = time.monotonic()
    paths = EvalPaths.for_clip(clip_id, root)
    blur_records = read_jsonl(paths.clip / "blur.jsonl")
    parameters = {
        "blur_manifest_sha256": _digest(paths.clip / "blur.jsonl"),
        "clip_id": clip_id,
        "columns": 3,
        "rows": 3,
        "tile_size": list(SHEET_TILE_SIZE),
    }
    manifest_path = paths.clip / "sheets.jsonl"
    meta_path = paths.clip / "sheets.meta.json"
    skipped = not force and manifest_path.exists() and matching_meta(meta_path, parameters)
    if not skipped:
        (paths.clip / "sheets.reviewed.json").unlink(missing_ok=True)
        if force and paths.sheets.exists():
            shutil.rmtree(paths.sheets)
        paths.sheets.mkdir(parents=True, exist_ok=True)
        eligible = [row for row in blur_records if row["upload_ok"] is True]
        sheets: list[dict[str, object]] = []
        for sheet_index, start in enumerate(range(0, len(eligible), 9), start=1):
            rows = eligible[start : start + 9]
            sheet = Image.new("RGB", (SHEET_TILE_SIZE[0] * 3, SHEET_TILE_SIZE[1] * 3), "black")
            for tile_index, row in enumerate(rows):
                with Image.open(paths.root / str(row["blurred_path"])) as image:
                    tile = _sheet_tile(
                        image, f"frame {row['frame_index']}  t={float(row['t_s']):.1f}s"
                    )
                sheet.paste(
                    tile,
                    ((tile_index % 3) * SHEET_TILE_SIZE[0], (tile_index // 3) * SHEET_TILE_SIZE[1]),
                )
            destination = paths.sheets / f"s_{sheet_index:04d}.jpg"
            sheet.save(destination, format="JPEG", quality=85)
            sheets.append(
                {
                    "frames": [
                        {"frame_index": row["frame_index"], "t_s": row["t_s"]} for row in rows
                    ],
                    "sheet_path": str(destination.relative_to(paths.root)),
                }
            )
        write_jsonl(manifest_path, sheets)
        write_meta(meta_path, command="sheets", parameters=parameters, started_at=started)
        update_index(paths.root, clip_id, "sheets", "needs_human_review")
    sheets = read_jsonl(manifest_path)
    if confirm_reviewed:
        marker = {
            "git_sha": git_sha(),
            "manifest_sha256": _digest(manifest_path),
            "reviewed": True,
            "sheet_count": len(sheets),
            "sheet_sha256": {
                row["sheet_path"]: _digest(paths.root / row["sheet_path"]) for row in sheets
            },
        }
        (paths.clip / "sheets.reviewed.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        update_index(paths.root, clip_id, "sheets", "human_reviewed")
    return {
        "status": "skipped" if skipped else "complete",
        "sheets": len(sheets),
        "eligible_frames": sum(len(row["frames"]) for row in sheets),
        "human_reviewed": (paths.clip / "sheets.reviewed.json").exists(),
    }
