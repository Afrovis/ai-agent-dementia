"""Replay cached pose detections through the real perceive tracker and score them.

Reads `clips/<clip>/cache/<model>-<variant>.jsonl` from `detection_cache.py`,
rebuilds the `PoseResult` the live backend would have produced, drives
`perceive.classify.StateTracker` exactly as `video_eval.predict` does, and
scores the resulting state stream with `vlm_agreement.py`. No model runs, so
a whole model x variant x threshold matrix takes seconds.

Usage:
    python tools/video_eval/scripts/rule_replay.py --data-root ../data-ai-agent-dementia \
        --clip <clip-id> --model yolov8n-pose --variant letterbox [--set floor_head_y=0.55]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import yaml
from perceive.backends import LANDMARK_NAMES, Landmark, PoseResult
from perceive.classify import ClassifyThresholds, StateTracker, zone_for_pose
from perceive.zones import load_zones

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_agreement import (  # noqa: E402
    POSTURES,
    SCRIPT_STATE,
    agreement,
    event_latencies,
    read_jsonl,
)

SCRIPT_MARGIN_S = 2.0
"""Frames this close to a scripted transition are left out of the script reference."""

COCO = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
YOLO_KEYPOINT_FLOOR = 0.5
"""ultralytics zeroes keypoints under this confidence; the cache keeps them zeroed."""


def to_pose(det: dict, mediapipe: bool) -> PoseResult:
    landmarks = {}
    for name in LANDMARK_NAMES:
        x, y, c = det["kp"][COCO[name]]
        if not mediapipe and x == 0.0 and y == 0.0:
            continue
        landmarks[name] = Landmark(x=x, y=y, visibility=c)
    return PoseResult(landmarks=landmarks, bbox=tuple(det["bbox"]), confidence=det["conf"])


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def select_argmax(dets: list[dict], _last_box: list[float] | None, _gap: int) -> dict:
    """Today's rule_replay/live-backend default absent phantom filtering:
    the single highest-confidence candidate, no memory of prior frames."""
    return max(dets, key=lambda d: d["conf"])


def select_track(
    dets: list[dict],
    last_box: list[float] | None,
    gap: int,
    *,
    proximity_radius: float = 0.3,
    max_track_gap: int = 15,
) -> dict:
    """Prefer the candidate closest to the last selected box over the one
    with the highest raw confidence, as long as the track is still fresh --
    the same continuity idea as `KnownPhantoms._rescue_from_confident_phantom`
    in perceive/phantom.py, generalised to every frame rather than only the
    narrow case where the argmax pick sits on a known phantom box. A person
    moving through the room keeps winning because their box stays close to
    where they just were; an unrelated high-confidence object elsewhere only
    wins once the track has gone stale (gap > max_track_gap) or nothing is
    within `proximity_radius`, exactly like a real re-acquire after the
    person actually left frame."""
    if last_box is not None and gap <= max_track_gap:
        in_range = [d for d in dets if _center_distance(d["bbox"], last_box) <= proximity_radius]
        if in_range:
            return min(in_range, key=lambda d: _center_distance(d["bbox"], last_box))
    return max(dets, key=lambda d: d["conf"])


SELECTORS = {"argmax": select_argmax, "track": select_track}


def replay(
    cache_rows: list[dict],
    gate: dict[int, bool],
    zones,
    thresholds: ClassifyThresholds,
    *,
    mediapipe: bool,
    model_floor: float,
    gated: bool,
    confirm_frames: int = 3,
    select: str = "argmax",
) -> list[dict]:
    tracker = StateTracker(thresholds=thresholds, confirm_frames=confirm_frames)
    selector = SELECTORS[select]
    rows = []
    last_box: list[float] | None = None
    calls_since_confirmed = 0
    for row in cache_rows:
        admitted = gate[row["frame_index"]] or not gated
        pose = None
        if admitted and not row.get("skipped"):
            dets = [d for d in row["dets"] if d["conf"] >= model_floor]
            if dets:
                chosen = selector(dets, last_box, calls_since_confirmed)
                pose = to_pose(chosen, mediapipe)
                last_box = list(chosen["bbox"])
                calls_since_confirmed = 0
            else:
                calls_since_confirmed += 1
            zone = zone_for_pose(zones, pose) if pose is not None else "other"
            tracker.update(pose, zone, row["t_s"])
        snap = tracker.snapshot()
        rows.append(
            {
                "frame_index": row["frame_index"],
                "t_s": row["t_s"],
                "state": snap[0] if snap else None,
                "bbox": list(pose.bbox) if pose else None,
                "conf": pose.confidence if pose else None,
            }
        )
    return rows


def score(rows: list[dict], reference: dict[int, str], script: list[dict]) -> dict:
    agree = agreement(rows, reference)
    events = event_latencies(rows, script)
    predicted_floor = sum(
        1
        for r in rows
        if r["state"] == "on_floor" and reference.get(r["frame_index"]) not in (None, "on_floor")
    )
    # A floor episode is a run of consecutive `on_floor` frames; it is false
    # when none of its frames is labelled on_floor. The agent alerts on the
    # first on_floor frame, so episodes, not frames, are what a caregiver sees.
    floor_episodes = false_episodes = 0
    run: list[str | None] = []
    for r in [*rows, {"state": None, "frame_index": -1}]:
        if r["state"] == "on_floor":
            run.append(reference.get(r["frame_index"]))
        elif run:
            floor_episodes += 1
            false_episodes += "on_floor" not in run
            run = []
    recalls = [v for v in agree["recall"].values() if v is not None]
    return {
        "agree": agree["overall"],
        "macro": sum(recalls) / len(recalls),
        "recall": agree["recall"],
        "totals": agree["totals"],
        "confusion": agree["confusion"],
        "events": sum(1 for *_, lat in events if lat is not None),
        "n_events": len(events),
        "missed": [(t0, target) for t0, target, lat in events if lat is None],
        "floor_fp": predicted_floor,
        "floor_episodes": floor_episodes,
        "false_floor_episodes": false_episodes,
    }


def script_reference(cache_rows: list[dict], script: list[dict]) -> dict[int, str]:
    """A per-frame reference from `clip.yaml`, independent of any VLM: each
    scripted action holds until the next; actions perceive cannot express and
    frames within `SCRIPT_MARGIN_S` of a transition are left unscored."""
    times = [float(event["t_s"]) for event in script]
    reference = {}
    for row in cache_rows:
        t = float(row["t_s"])
        active = [event for event in script if float(event["t_s"]) <= t]
        if not active or any(abs(t - s) < SCRIPT_MARGIN_S for s in times):
            continue
        posture = SCRIPT_STATE.get(active[-1]["action"])
        if posture is not None:
            reference[int(row["frame_index"])] = posture
    return reference


def load_clip(root: Path, clip_id: str, labels_path: Path | None = None):
    clip = root / "clips" / clip_id
    labels = read_jsonl(labels_path or clip / "labels" / "local.jsonl")
    reference = {
        int(r["frame_index"]): r["posture"] for r in labels if r.get("posture") in POSTURES
    }
    script = (yaml.safe_load((clip / "clip.yaml").read_text()) or {}).get("script") or []
    return clip, reference, script


def parse_sets(pairs: list[str]) -> dict[str, float | bool]:
    defaults = ClassifyThresholds()
    names = {f.name for f in fields(ClassifyThresholds)}
    out: dict[str, float | bool] = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        if key not in names:
            raise SystemExit(f"unknown threshold {key}; known: {sorted(names)}")
        if isinstance(getattr(defaults, key), bool):
            out[key] = value.lower() in ("1", "true", "yes")
        else:
            out[key] = float(value)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip", action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--model-floor", type=float, default=0.25)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--set", action="append", default=[], help="threshold=value")
    parser.add_argument("--confusion", action="store_true")
    parser.add_argument("--dump", type=Path, help="write replayed rows as JSONL here")
    parser.add_argument(
        "--reference",
        choices=("vlm", "script"),
        default="vlm",
        help="score against VLM labels or the clip.yaml script timeline",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        help="score against clips/<clip>/labels/local.jsonl under this root instead",
    )
    parser.add_argument(
        "--select",
        choices=sorted(SELECTORS),
        default="argmax",
        help="candidate-selection strategy: argmax (today's default) or track (prefer "
        "proximity to the last selected box)",
    )
    args = parser.parse_args(argv)

    thresholds = replace(ClassifyThresholds(), **parse_sets(args.set))
    root = args.data_root.resolve()
    header = f"{'clip':>4s} {'model':16s} {'variant':12s} {'agree':>5s} {'macro':>5s} "
    header += " ".join(f"{p:>8s}" for p in POSTURES) + "  events floorFP falseEp"
    print(header)
    for clip_id in args.clip:
        labels_path = (
            args.labels_root.resolve() / "clips" / clip_id / "labels" / "local.jsonl"
            if args.labels_root
            else None
        )
        clip, reference, script = load_clip(root, clip_id, labels_path)
        zones = load_zones(clip / "zones.yaml")
        for variant in args.variant:
            gate = {
                int(k): v
                for k, v in json.loads(
                    (clip / "cache" / f"gate-{variant}.json").read_text()
                ).items()
            }
            for model in args.model:
                path = clip / "cache" / f"{model}-{variant}.jsonl"
                if not path.exists():
                    print(f"missing {path.name}")
                    continue
                cache_rows = read_jsonl(path)
                frame_reference = (
                    script_reference(cache_rows, script)
                    if args.reference == "script"
                    else reference
                )
                rows = replay(
                    cache_rows,
                    gate,
                    zones,
                    thresholds,
                    mediapipe=model.startswith("mediapipe"),
                    model_floor=0.0 if model.startswith("mediapipe") else args.model_floor,
                    gated=not args.no_gate,
                    select=args.select,
                )
                result = score(rows, frame_reference, script)
                cells = " ".join(
                    f"{result['recall'][p]:8.2f}"
                    if result["recall"][p] is not None
                    else f"{'-':>8s}"
                    for p in POSTURES
                )
                print(
                    f"{clip_id[-2:]:>4s} {model:16s} {variant:12s} {result['agree']:5.2f} "
                    f"{result['macro']:5.2f} {cells}  "
                    f"{result['events']:2d}/{result['n_events']:<2d}  "
                    f"{result['floor_fp']:5d} {result['false_floor_episodes']:3d}"
                    f"/{result['floor_episodes']}"
                )
                if args.confusion:
                    for truth in POSTURES:
                        counts = result["confusion"].get(truth)
                        if counts:
                            print(
                                f"      {truth:>10s} -> "
                                + ", ".join(f"{k} {v}" for k, v in counts.most_common())
                            )
                    if result["missed"]:
                        print(
                            "      missed: " + ", ".join(f"{t:g}s {s}" for t, s in result["missed"])
                        )
                if args.dump:
                    args.dump.mkdir(parents=True, exist_ok=True)
                    with (args.dump / f"{clip_id}-{model}-{variant}.jsonl").open("w") as handle:
                        for r in rows:
                            handle.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
