"""Compare offline predictions with the local VLM labels and the scripted timeline.

A stop-gap scorer for clips that have no confirmed `labels/reference.yaml`
yet (the `reconcile` step in docs/VIDEO_EVAL.md has not run). It answers two
questions per prediction tag:

1. Per-frame agreement with `labels/local.jsonl`: predicted `standing` and
   `walking` collapse to the VLM's `upright`; frames the VLM failed on are
   skipped. Recall is reported per VLM posture as `recall/frames`.
2. Latency from each scripted event in `clip.yaml` to the first prediction
   that reaches the matching state, within a 30 s window, allowing 2 s of
   slack before the scripted time because the script times are approximate.

Usage:
    python tools/video_eval/scripts/vlm_agreement.py --clip <clip-id> [--tag <glob>]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path

import yaml

COLLAPSE = {"standing": "upright", "walking": "upright"}
POSTURES = ("in_bed", "sitting_up", "upright", "on_floor", "absent")

# Scripted actions that map onto a perceive state. Transitions and actions the
# vocabulary cannot express (drawers, dressing, getting up) are left out.
SCRIPT_STATE = {
    "in_bed": "in_bed",
    "in_bed_above_blanket": "in_bed",
    "sitting_up": "sitting_up",
    "sitting": "sitting_up",
    "sitting_on_chair": "sitting_up",
    "bed_exit": "upright",
    "walking": "upright",
    "standing_and_walking": "upright",
    "in_frame": "upright",
    "floor": "on_floor",
    "sitting_on_floor": "on_floor",
    "out_of_frame": "absent",
}
EVENT_WINDOW_S = 30.0
EARLY_SLACK_S = 2.0


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def agreement(rows: list[dict], reference: dict[int, str]) -> dict:
    hits: Counter = Counter()
    totals: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        truth = reference.get(row["frame_index"])
        state = row.get("state")
        if truth is None or state is None:
            continue
        predicted = COLLAPSE.get(state, state)
        totals[truth] += 1
        confusion[truth][predicted] += 1
        if predicted == truth:
            hits[truth] += 1
    overall = sum(hits.values()) / sum(totals.values()) if totals else 0.0
    recall = {p: (hits[p] / totals[p] if totals[p] else None) for p in POSTURES}
    return {"overall": overall, "recall": recall, "totals": totals, "confusion": confusion}


def event_latencies(rows: list[dict], script: list[dict]) -> list[tuple[float, str, float | None]]:
    results = []
    for event in script:
        target = SCRIPT_STATE.get(event["action"])
        if target is None:
            continue
        t0 = float(event["t_s"])
        hit = None
        for row in rows:
            t = float(row["t_s"])
            if t < t0 - EARLY_SLACK_S or t > t0 + EVENT_WINDOW_S:
                continue
            state = row.get("state")
            if state is not None and COLLAPSE.get(state, state) == target:
                hit = max(0.0, t - t0)
                break
        results.append((t0, target, hit))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--tag", default="*", help="glob over prediction tags")
    parser.add_argument("--data-root", type=Path, default=Path("../data-ai-agent-dementia"))
    parser.add_argument("--confusion", action="store_true", help="print confusion per tag")
    args = parser.parse_args(argv)

    clip = args.data_root.resolve() / "clips" / args.clip
    labels = read_jsonl(clip / "labels" / "local.jsonl")
    reference = {
        int(r["frame_index"]): r["posture"] for r in labels if r.get("posture") in POSTURES
    }
    script = (yaml.safe_load((clip / "clip.yaml").read_text()) or {}).get("script") or []

    print(f"clip {args.clip}: {len(reference)} VLM-labelled frames, {len(script)} script events")
    header = f"{'tag':46s} {'agree':>6s} "
    header += " ".join(f"{p:>10s}" for p in POSTURES) + "   events  median_s"
    print(header)
    for path in sorted(clip.glob("predictions/*.jsonl")):
        tag = path.stem
        if not fnmatch(tag, args.tag):
            continue
        rows = read_jsonl(path)
        agree = agreement(rows, reference)
        events = event_latencies(rows, script)
        matched = [lat for _, _, lat in events if lat is not None]
        cells = []
        for posture in POSTURES:
            recall = agree["recall"][posture]
            text = f"{recall:.2f}" if recall is not None else "-"
            cells.append(f"{text:>6s}/{agree['totals'][posture]:<3d}")
        median = f"{statistics.median(matched):.1f}" if matched else "-"
        line = f"{tag:46s} {agree['overall']:6.2f} {' '.join(cells)}"
        line += f"   {len(matched):2d}/{len(events):<2d}   {median:>6s}"
        print(line)
        misses = [(t0, target) for t0, target, lat in events if lat is None]
        if misses:
            print("    missed: " + ", ".join(f"{t0:g}s {target}" for t0, target in misses))
        if args.confusion:
            for truth in POSTURES:
                row = agree["confusion"].get(truth)
                if row:
                    cells = ", ".join(f"{k} {v}" for k, v in row.most_common())
                    print(f"    {truth:>10s} -> {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
