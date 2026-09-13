"""Reconcile two independent frame labellers into a human-confirmed timeline."""

from __future__ import annotations

import getpass
import hashlib
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml
from perceive.classify import ClassifyThresholds
from PIL import Image

from video_eval.common import matching_meta, read_jsonl, update_index, write_meta
from video_eval.paths import EvalPaths

FPS = 2.0
CODEX_PREFERENCE_CONFIDENCE = 0.7
SHORT_INTERVAL_SECONDS = 2.0
SCRIPT_TOLERANCE_SECONDS = 10.0
WALK_THRESHOLD = ClassifyThresholds().walk_displacement_threshold
REFERENCE_STATES = frozenset({"in_bed", "sitting_up", "standing", "walking", "on_floor", "absent"})
REFERENCE_ZONES = frozenset({"bed", "door", "bathroom_path", "other"})


class CentroidDetector(Protocol):
    def detect(self, image: Image.Image) -> list[Any]: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["frame_index"]): row for row in read_jsonl(path)}


def reconcile_value(local: dict[str, Any], codex: dict[str, Any], field: str) -> str:
    """Apply the documented agreement/Codex-confidence rule to one field."""
    local_value = local.get(field)
    codex_value = codex.get(field)
    if local_value is not None and local_value == codex_value:
        return str(local_value)
    if (
        codex_value is not None
        and float(codex.get("confidence", 0.0)) >= CODEX_PREFERENCE_CONFIDENCE
    ):
        return str(codex_value)
    return "disputed"


def _centroid_x_by_frame(
    frames: list[dict[str, Any]], root: Path, detector: CentroidDetector
) -> dict[int, float | None]:
    centroids: dict[int, float | None] = {}
    for frame in frames:
        with Image.open(root / frame["review_path"]) as image:
            detections = detector.detect(image)
            if not detections:
                centroids[int(frame["frame_index"])] = None
                continue
            # The evaluation is single-person. Prefer the largest box if a
            # detector produces a stray second person-shaped region.
            person = max(
                detections,
                key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
            )
            centroids[int(frame["frame_index"])] = (
                (person.bbox[0] + person.bbox[2]) / 2.0 / image.width
            )
    return centroids


def derive_walking(
    rows: list[dict[str, Any]], centroids: dict[int, float | None], threshold: float
) -> None:
    """Split upright into standing/walking using classify.py's five-frame x displacement."""
    history: deque[float] = deque(maxlen=5)
    for row in rows:
        if row["state"] != "upright":
            history.clear()
            continue
        centroid = centroids.get(int(row["frame_index"]))
        if centroid is None:
            history.clear()
            row["state"] = "standing"
            continue
        history.append(centroid)
        row["state"] = (
            "walking"
            if len(history) >= 2 and abs(history[-1] - history[0]) >= threshold
            else "standing"
        )


def _intervals(rows: list[dict[str, Any]], frame_interval_s: float = 0.5) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (row["state"], row["zone"])
        if result and (result[-1]["state"], result[-1]["zone"]) == key:
            result[-1]["to_s"] = float(row["t_s"]) + frame_interval_s
        else:
            result.append(
                {
                    "from_s": float(row["t_s"]),
                    "to_s": float(row["t_s"]) + frame_interval_s,
                    "state": row["state"],
                    "zone": row["zone"],
                }
            )
    return result


def smooth_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Absorb sub-two-second noise, preserving safety and unresolved intervals."""
    work = [dict(row) for row in intervals]
    changed = True
    while changed and len(work) > 1:
        changed = False
        for index, row in enumerate(work):
            duration = float(row["to_s"]) - float(row["from_s"])
            if (
                duration >= SHORT_INTERVAL_SECONDS
                or row["state"] in {"on_floor", "disputed"}
                or row["zone"] == "disputed"
            ):
                continue
            if 0 < index < len(work) - 1:
                previous, following = work[index - 1], work[index + 1]
                if (previous["state"], previous["zone"]) == (
                    following["state"],
                    following["zone"],
                ):
                    previous["to_s"] = following["to_s"]
                    del work[index : index + 2]
                else:
                    previous_duration = previous["to_s"] - previous["from_s"]
                    following_duration = following["to_s"] - following["from_s"]
                    if previous_duration >= following_duration:
                        previous["to_s"] = row["to_s"]
                        del work[index]
                    else:
                        following["from_s"] = row["from_s"]
                        del work[index]
            elif index == 0:
                work[1]["from_s"] = row["from_s"]
                del work[0]
            else:
                work[-2]["to_s"] = row["to_s"]
                del work[-1]
            changed = True
            break
    return work


def _matching_transition(action: str, previous: dict[str, Any] | None, row: dict[str, Any]) -> bool:
    state, zone = row["state"], row["zone"]
    previous_state = previous["state"] if previous else None
    if action == "bed_exit":
        return state in {"standing", "walking"} and previous_state not in {"standing", "walking"}
    if action == "floor":
        return state == "on_floor" and previous_state != "on_floor"
    if action == "room_exit":
        return state == "absent" and previous_state != "absent"
    if action == "return":
        return state != "absent" and previous_state == "absent"
    if action in {"door", "bathroom_path"}:
        return zone == action and (previous is None or previous["zone"] != action)
    if action in {"in_bed", "sitting_up", "upright", "standing", "walking", "on_floor", "absent"}:
        wanted = {"standing", "walking"} if action == "upright" else {action}
        return state in wanted and previous_state not in wanted
    return True  # descriptive actions such as walk_to_door have no exact state contract


def scenario_disagreements(script: Any, intervals: list[dict[str, Any]]) -> list[str]:
    if script == "unknown" or not script:
        return ["Scenario card has no usable script; event alignment was not verified."]
    messages: list[str] = []
    for event in script:
        action = str(event.get("action", ""))
        expected = float(event.get("t_s", 0.0))
        if action not in {"bed_exit", "floor", "room_exit", "return"}:
            continue
        candidates = [
            row
            for index, row in enumerate(intervals)
            if _matching_transition(action, intervals[index - 1] if index else None, row)
            and abs(float(row["from_s"]) - expected) <= SCRIPT_TOLERANCE_SECONDS
        ]
        if not candidates:
            messages.append(
                f"Script event `{action}` at {expected:.1f}s has no matching timeline "
                "transition within 10s."
            )
    return messages


def _disputed_runs(rows: list[dict[str, Any]], frames: dict[int, dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    start: int | None = None
    active: list[dict[str, Any]] = []
    for position, row in enumerate(rows + [{"state": None, "zone": None}]):
        disputed = row.get("state") == "disputed" or row.get("zone") == "disputed"
        if disputed and start is None:
            start = position
        if disputed:
            active.append(row)
        elif start is not None:
            first, last = active[0], active[-1]
            paths = [frames[int(item["frame_index"])]["review_path"] for item in active]
            messages.append(
                f"Frames {first['frame_index']}–{last['frame_index']} ({first['t_s']:.1f}–"
                f"{last['t_s']:.1f}s) are disputed; review: "
                + ", ".join(f"`{path}`" for path in paths)
            )
            start = None
            active = []
    return messages


def _write_disagreements(path: Path, messages: list[str]) -> None:
    body = "# Reconciliation disagreements\n\n"
    body += (
        "No disagreements.\n"
        if not messages
        else "\n".join(f"- {item}" for item in messages) + "\n"
    )
    path.write_text(body, encoding="utf-8")


def validate_timeline(timeline: Any) -> list[dict[str, Any]]:
    """Reject ambiguous human edits before they become scored ground truth."""
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("reference timeline must be a non-empty list")
    previous_to: float | None = None
    for index, row in enumerate(timeline):
        if not isinstance(row, dict):
            raise ValueError(f"timeline interval {index} must be a mapping")
        if row.get("state") not in REFERENCE_STATES:
            raise ValueError(f"timeline interval {index} has invalid state {row.get('state')!r}")
        if row.get("zone") not in REFERENCE_ZONES:
            raise ValueError(f"timeline interval {index} has invalid zone {row.get('zone')!r}")
        start, end = float(row["from_s"]), float(row["to_s"])
        if start < 0 or end <= start:
            raise ValueError(f"timeline interval {index} has invalid bounds")
        if previous_to is not None and abs(start - previous_to) > 1e-6:
            raise ValueError(f"timeline interval {index} is not contiguous")
        previous_to = end
    return timeline


def reconcile_clip(
    clip_id: str,
    *,
    root: Path | None = None,
    force: bool = False,
    confirm: bool = False,
    confirmer: str | None = None,
    walk_threshold: float = WALK_THRESHOLD,
    detector_factory: Callable[[], CentroidDetector] | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    paths = EvalPaths.for_clip(clip_id, root)
    draft = paths.labels / "reference.draft.yaml"
    reference = paths.labels / "reference.yaml"
    disagreements_path = paths.labels / "disagreements.md"
    meta_path = paths.labels / "reconcile.meta.json"

    if confirm:
        if not draft.exists():
            raise FileNotFoundError("reference draft does not exist; run reconcile first")
        raw = yaml.safe_load(draft.read_text(encoding="utf-8")) or {}
        timeline = raw.get("timeline") or []
        if any(row.get("state") == "disputed" or row.get("zone") == "disputed" for row in timeline):
            raise RuntimeError("reference draft still contains disputed values")
        validate_timeline(timeline)
        raw["confirmed_by"] = confirmer or getpass.getuser()
        raw["confirmed_at"] = datetime.now(UTC).date().isoformat()
        reference.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        update_index(paths.root, clip_id, "reconcile", "human_confirmed")
        return {
            "status": "confirmed",
            "intervals": len(timeline),
            "confirmed_by": raw["confirmed_by"],
        }

    local_path, codex_path = paths.labels / "local.jsonl", paths.labels / "codex.jsonl"
    parameters = {
        "clip_id": clip_id,
        "clip_card_sha256": _sha256(paths.clip / "clip.yaml"),
        "codex_sha256": _sha256(codex_path),
        "frames_sha256": _sha256(paths.frames),
        "local_sha256": _sha256(local_path),
        "walk_threshold": walk_threshold,
    }
    if (
        not force
        and draft.exists()
        and disagreements_path.exists()
        and matching_meta(meta_path, parameters)
    ):
        timeline = (yaml.safe_load(draft.read_text(encoding="utf-8")) or {}).get("timeline", [])
        return {"status": "skipped", "intervals": len(timeline)}

    frames_list = read_jsonl(paths.frames)
    frames = {int(row["frame_index"]): row for row in frames_list}
    local, codex = _load_rows(local_path), _load_rows(codex_path)
    if set(frames) != set(local):
        raise ValueError("frames and local labels must have identical frame indices")
    if not set(codex).issubset(frames):
        raise ValueError("Codex labels contain frame indices outside the frame manifest")
    rows = [
        {
            "frame_index": index,
            "t_s": float(frames[index]["t_s"]),
            "state": reconcile_value(local[index], codex.get(index, {}), "posture"),
            "zone": reconcile_value(local[index], codex.get(index, {}), "location"),
        }
        for index in sorted(frames)
    ]

    detector = detector_factory() if detector_factory else _default_detector()
    try:
        centroids = _centroid_x_by_frame(frames_list, paths.root, detector)
    finally:
        close = getattr(detector, "close", None)
        if close is not None:
            close()
    derive_walking(rows, centroids, walk_threshold)
    timeline = smooth_intervals(_intervals(rows))
    clip_card = yaml.safe_load((paths.clip / "clip.yaml").read_text(encoding="utf-8")) or {}
    messages = _disputed_runs(rows, frames) + scenario_disagreements(
        clip_card.get("script"), timeline
    )
    payload = {"clip_id": clip_id, "frame_interval_s": 1 / FPS, "timeline": timeline}
    paths.labels.mkdir(parents=True, exist_ok=True)
    draft.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_disagreements(disagreements_path, messages)
    write_meta(
        meta_path,
        command="reconcile",
        parameters=parameters,
        started_at=started,
        versions=("pillow", "pyyaml", "ultralytics"),
    )
    update_index(paths.root, clip_id, "reconcile", "needs_human_confirmation")
    return {"status": "complete", "intervals": len(timeline), "disagreements": len(messages)}


def _default_detector() -> CentroidDetector:
    from video_eval.blur import YoloPersonDetector

    return YoloPersonDetector()
