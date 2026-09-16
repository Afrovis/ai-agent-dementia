"""Find recurring bounding-box "jump" locations in recorded prediction runs.

Some rooms have no genuinely empty footage to calibrate config/phantoms.yaml
against (`perceive.calibrate_phantoms` needs one). With exactly one person in
frame, the selected bbox should trace a single continuous track. Any
tracking-loss / re-lock event that lands the box far from where it just was
is either legitimate fast motion or a phantom lock-on -- and unlike a moving
person, a static phantom object gets landed on again and again at the same
spot. This walks one or more `predict.py` prediction JSONL files, finds those
teleports, and clusters the destinations. A cluster that recurs -- especially
across separate videos of the same room, and especially when confidence was
dropping in the frames just before the jump -- is a phantom candidate worth
a human look before adding it to config/phantoms.yaml.

Usage:
    python tools/video_eval/scripts/phantom_jump_finder.py \
        --predictions ../data-ai-agent-dementia/clips/<clip1>/predictions/<tag>.jsonl \
        --predictions ../data-ai-agent-dementia/clips/<clip2>/predictions/<tag>.jsonl \
        --out /tmp/phantom_candidates.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b
    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
    area_b = max(0.0, bx_max - bx_min) * max(0.0, by_max - by_min)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def center(box: Box) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = box
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def center_distance(a: Box, b: Box) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _median_box(boxes: list[Box]) -> Box:
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0

    return (
        _median([b[0] for b in boxes]),
        _median([b[1] for b in boxes]),
        _median([b[2] for b in boxes]),
        _median([b[3] for b in boxes]),
    )


@dataclass
class JumpEvent:
    clip: str
    frame_index: int
    t_s: float
    box: Box
    confidence: float | None
    gap_frames: int
    pre_conf_trend: list[float]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_jumps(
    rows: list[dict],
    clip: str,
    *,
    jump_threshold: float,
    pre_window: int,
) -> list[JumpEvent]:
    """Walk one clip's prediction rows in order and record every frame where
    the selected box's center moved >= jump_threshold from the last frame
    that had a detection. `gap_frames` counts frames in between with no
    detection (gated or lost), and `pre_conf_trend` is the confidence of the
    last `pre_window` detected frames before the jump, so a caller can see
    whether confidence was already sliding before the box teleported."""
    events: list[JumpEvent] = []
    prev_box: Box | None = None
    prev_frame_index: int | None = None
    recent_conf: list[float] = []

    for row in rows:
        bbox = row.get("bbox")
        if bbox is None:
            continue
        box = tuple(bbox)
        conf = row.get("detect_confidence")
        frame_index = row["frame_index"]

        if prev_box is not None:
            dist = center_distance(prev_box, box)
            if dist >= jump_threshold:
                gap = frame_index - prev_frame_index - 1
                events.append(
                    JumpEvent(
                        clip=clip,
                        frame_index=frame_index,
                        t_s=row.get("t_s", 0.0),
                        box=box,
                        confidence=conf,
                        gap_frames=gap,
                        pre_conf_trend=list(recent_conf[-pre_window:]),
                    )
                )

        prev_box = box
        prev_frame_index = frame_index
        if conf is not None:
            recent_conf.append(conf)
            if len(recent_conf) > pre_window:
                recent_conf.pop(0)

    return events


def cluster_jumps(events: list[JumpEvent], *, cluster_iou: float) -> list[dict]:
    """Greedy IoU clustering of jump *destinations* (same style as
    perceive.phantom.cluster_static_boxes, but keyed on recurrence across
    jump events rather than frame-fraction of a single empty-room run)."""
    clusters: list[list[JumpEvent]] = []
    for event in events:
        matched = False
        for cluster in clusters:
            if iou(event.box, cluster[0].box) >= cluster_iou:
                cluster.append(event)
                matched = True
                break
        if not matched:
            clusters.append([event])

    summaries = []
    for cluster in clusters:
        clips = sorted({e.clip for e in cluster})
        confidences = [e.confidence for e in cluster if e.confidence is not None]
        pre_drops = [
            e.pre_conf_trend[0] - e.pre_conf_trend[-1]
            for e in cluster
            if len(e.pre_conf_trend) >= 2
        ]
        summaries.append(
            {
                "box": list(_median_box([e.box for e in cluster])),
                "occurrences": len(cluster),
                "clips": clips,
                "n_clips": len(clips),
                "mean_confidence": round(sum(confidences) / len(confidences), 3)
                if confidences
                else None,
                "mean_gap_frames": round(sum(e.gap_frames for e in cluster) / len(cluster), 1),
                "mean_pre_jump_confidence_drop": round(sum(pre_drops) / len(pre_drops), 3)
                if pre_drops
                else None,
                "example_frames": [
                    {"clip": e.clip, "frame_index": e.frame_index, "t_s": round(e.t_s, 1)}
                    for e in cluster[:5]
                ],
            }
        )

    summaries.sort(key=lambda s: (s["n_clips"], s["occurrences"]), reverse=True)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", action="append", required=True, dest="predictions")
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=0.22,
        help="Min normalized center-distance between consecutive detected frames to call a jump",
    )
    parser.add_argument(
        "--cluster-iou",
        type=float,
        default=0.5,
        help="IoU to merge two jump-destination boxes into the same cluster",
    )
    parser.add_argument("--pre-window", type=int, default=3)
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Only report clusters landed on at least this many times",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    all_events: list[JumpEvent] = []
    for pred_path in args.predictions:
        path = Path(pred_path)
        rows = read_jsonl(path)
        clip = path.parent.parent.name  # .../<clip>/predictions/<tag>.jsonl
        events = find_jumps(
            rows, clip, jump_threshold=args.jump_threshold, pre_window=args.pre_window
        )
        print(
            f"{clip} ({path.name}): {len(rows)} rows, "
            f"{len(events)} jump events (>= {args.jump_threshold})"
        )
        all_events.extend(events)

    clusters = cluster_jumps(all_events, cluster_iou=args.cluster_iou)
    clusters = [c for c in clusters if c["occurrences"] >= args.min_occurrences]

    print(
        f"\n{len(clusters)} recurring jump-destination clusters "
        f"(>= {args.min_occurrences} occurrences), most-supported first:\n"
    )
    for c in clusters:
        print(
            f"  box={[round(v, 3) for v in c['box']]} "
            f"occurrences={c['occurrences']} clips={c['n_clips']} "
            f"mean_conf={c['mean_confidence']} mean_gap_frames={c['mean_gap_frames']} "
            f"mean_pre_jump_conf_drop={c['mean_pre_jump_confidence_drop']}"
        )
        for ex in c["example_frames"]:
            print(f"      {ex['clip']} frame={ex['frame_index']} t_s={ex['t_s']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as handle:
            yaml.safe_dump(
                {
                    "phantom_candidates": [
                        {
                            "box": c["box"],
                            "occurrences": c["occurrences"],
                            "clips": c["clips"],
                            "mean_confidence": c["mean_confidence"],
                            "mean_pre_jump_confidence_drop": c["mean_pre_jump_confidence_drop"],
                        }
                        for c in clusters
                    ]
                },
                handle,
                sort_keys=False,
            )
        print(f"\nWrote {len(clusters)} candidates to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
