"""Grid-search perceive rules on cached detections, scored against confirmed references.

`rule_replay.py` scores against per-frame VLM labels, which cannot tell
`walking` from `standing` (one frame shows no motion). This script scores the
same cached replay against each clip's human-confirmed `labels/reference.yaml`
instead, adds bed-occupancy and bed-exit metrics, and sweeps threshold grids
with a leave-one-clip-out check so a setting is judged on a clip it was not
picked on.

Usage:
    python tools/video_eval/scripts/state_sweep.py --data-root ../data-ai-agent-dementia
    python tools/video_eval/scripts/state_sweep.py --data-root ../data-ai-agent-dementia \
        --set walk_mode=ground --grid walk_motion_threshold=0.3,0.4,0.5 \
        --grid walk_window_seconds=1.5,2.5 --loco

`--set`/`--grid` take any `ClassifyThresholds` field plus the tracker's own
`confirm_frames` and `bed_hold_seconds`. No model runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from collections import Counter
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml
from perceive.classify import (
    ClassifyThresholds,
    StateTracker,
    ground_xy,
    ground_zone_for_pose,
    zone_for_pose,
)
from perceive.zones import load_zones

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_replay import to_pose  # noqa: E402
from vlm_agreement import read_jsonl  # noqa: E402

TRACKER_KEYS = {"confirm_frames": int, "bed_hold_seconds": float}
EXIT_TO = {"standing", "walking", "on_floor"}
EXIT_WINDOW_S = (-3.0, 30.0)
"""A predicted bed exit this close to a reference exit (early, late) catches it."""
FALSE_EXIT_TOLERANCE_S = 10.0
UPRIGHT = {"standing", "walking"}


def occupied(state: str | None, zone: str | None) -> bool:
    """Bed occupancy as the agent sees it: lying in bed, or sitting up on it."""
    return state == "in_bed" or (state == "sitting_up" and zone == "bed")


def reference_frames(timeline: list[dict], cache_rows: list[dict]) -> dict[int, tuple[str, str]]:
    out = {}
    for row in cache_rows:
        t = float(row["t_s"])
        for span in timeline:
            if float(span["from_s"]) <= t < float(span["to_s"]):
                out[int(row["frame_index"])] = (span["state"], span["zone"])
                break
    return out


def reference_exits(timeline: list[dict]) -> list[float]:
    exits = []
    for before, after in itertools.pairwise(timeline):
        if occupied(before["state"], before["zone"]) and after["state"] in EXIT_TO:
            exits.append(float(after["from_s"]))
    return exits


def replay(
    cache_rows: list[dict],
    gate: dict[int, bool],
    zones: Any,
    thresholds: ClassifyThresholds,
    tracker_kwargs: dict[str, Any],
    *,
    model_floor: float,
    mediapipe: bool,
) -> list[dict]:
    tracker = StateTracker(thresholds=thresholds, **tracker_kwargs)
    rows = []
    for row in cache_rows:
        index = int(row["frame_index"])
        admitted = gate.get(index, True) and not row.get("skipped")
        chosen = pose = ground_zone = None
        published = False
        if admitted:
            dets = [d for d in row["dets"] if d["conf"] >= model_floor]
            chosen = max(dets, key=lambda d: d["conf"]) if dets else None
            pose = to_pose(chosen, mediapipe) if chosen else None
            zone = zone_for_pose(zones, pose) if pose is not None else "other"
            ground_zone = ground_zone_for_pose(zones, pose) if pose is not None else None
            published = (
                tracker.update(pose, zone, float(row["t_s"]), ground_zone=ground_zone) is not None
            )
        snap = tracker.snapshot()
        rows.append(
            {
                "frame_index": index,
                "t_s": float(row["t_s"]),
                "state": snap[0] if snap else None,
                "state_confidence": snap[1] if snap else None,
                "zone": snap[2] if snap else None,
                # The rest is what `video_eval visualize --mode pipeline` draws.
                "gated": not admitted,
                "published": published,
                "detected": pose is not None,
                "detect_confidence": pose.confidence if pose is not None else None,
                "backend_ms": float(row.get("ms") or 0.0),
                "dets": [chosen] if chosen else [],
                "ground": list(ground_xy(pose)) if pose is not None else None,
                "ground_zone": ground_zone,
            }
        )
    return rows


def write_predictions(clip: dict, rows: list[dict], tag: str, args, config: dict) -> Path:
    """Write `rows` where `video_eval visualize --pipeline-tag <tag>` reads them."""
    predictions = args.data_root.resolve() / "clips" / clip["id"] / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    path = predictions / f"{tag}.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    meta = {
        "command": "state_sweep",
        "parameters": {
            "backend": "mediapipe" if args.model.startswith("mediapipe") else "yolo",
            "yolo_model": args.model,
            "variant": args.variant,
            "gated": not args.no_gate,
            "zones_dir": str(args.zones_dir) if args.zones_dir else None,
            "replayed_from": f"cache/{args.model}-{args.variant}.jsonl",
            "config": config,
        },
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return path


def predicted_exits(rows: list[dict]) -> list[float]:
    exits, was_occupied = [], False
    for row in rows:
        if row["state"] is None:
            continue
        now_occupied = occupied(row["state"], row["zone"])
        if was_occupied and row["state"] in EXIT_TO:
            exits.append(row["t_s"])
        if now_occupied or row["state"] in EXIT_TO:
            # Absent/undetected frames neither start nor end an occupancy run.
            was_occupied = now_occupied
    return exits


def clip_counts(rows: list[dict], reference: dict[int, tuple[str, str]], timeline: list[dict]):
    """Additive counts for one clip, so clips can be pooled before ratios."""
    confusion: Counter[tuple[str, str]] = Counter()
    bed = Counter()
    upright_in_bed = 0
    for row in rows:
        ref = reference.get(row["frame_index"])
        if ref is None:
            continue
        predicted = row["state"] or "undetected"
        confusion[(ref[0], predicted)] += 1
        truth, guess = occupied(*ref), occupied(row["state"], row["zone"])
        bed[(truth, guess)] += 1
        if row["state"] in UPRIGHT and row["zone"] == "bed" and ref[1] != "bed":
            upright_in_bed += 1

    ref_exits = reference_exits(timeline)
    pred_exits = predicted_exits(rows)
    latencies, caught = [], 0
    for t in ref_exits:
        hits = [p - t for p in pred_exits if EXIT_WINDOW_S[0] <= p - t <= EXIT_WINDOW_S[1]]
        if hits:
            caught += 1
            latencies.append(max(0.0, min(hits, key=abs)))
    false_exits = sum(
        1 for p in pred_exits if not any(abs(p - t) <= FALSE_EXIT_TOLERANCE_S for t in ref_exits)
    )

    floor_episodes = false_floor = 0
    run: list[str | None] = []
    for row in [*rows, {"state": None, "frame_index": -1}]:
        if row["state"] == "on_floor":
            ref = reference.get(row["frame_index"])
            run.append(ref[0] if ref else None)
        elif run:
            floor_episodes += 1
            false_floor += "on_floor" not in run
            run = []
    return {
        "confusion": confusion,
        "bed": bed,
        "exits": len(ref_exits),
        "exits_caught": caught,
        "exit_latencies": latencies,
        "false_exits": false_exits,
        "floor_episodes": floor_episodes,
        "false_floor_episodes": false_floor,
        "upright_in_bed": upright_in_bed,
    }


def pool(parts: list[dict]) -> dict:
    total: dict[str, Any] = {
        "confusion": Counter(),
        "bed": Counter(),
        "exit_latencies": [],
    }
    for part in parts:
        total["confusion"].update(part["confusion"])
        total["bed"].update(part["bed"])
        total["exit_latencies"] += part["exit_latencies"]
        for key in ("exits", "exits_caught", "false_exits", "floor_episodes", "upright_in_bed"):
            total[key] = total.get(key, 0) + part[key]
        total["false_floor_episodes"] = (
            total.get("false_floor_episodes", 0) + part["false_floor_episodes"]
        )
    return total


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def metrics(counts: dict) -> dict:
    confusion = counts["confusion"]
    n = sum(confusion.values())
    out: dict[str, Any] = {
        "frames": n,
        "acc": _ratio(sum(v for (t, p), v in confusion.items() if t == p), n),
        "upright_acc": _ratio(
            sum(v for (t, p), v in confusion.items() if t == p or (t in UPRIGHT and p in UPRIGHT)),
            n,
        ),
    }
    for state in ("standing", "walking", "in_bed", "sitting_up", "on_floor", "absent"):
        tp = confusion[(state, state)]
        support = sum(v for (t, _), v in confusion.items() if t == state)
        predicted = sum(v for (_, p), v in confusion.items() if p == state)
        recall, precision = _ratio(tp, support), _ratio(tp, predicted)
        f1 = (
            2 * recall * precision / (recall + precision)
            if recall and precision
            else (0.0 if support else None)
        )
        out[state] = {"recall": recall, "precision": precision, "f1": f1, "support": support}
    bed = counts["bed"]
    out["bed_recall"] = _ratio(bed[(True, True)], bed[(True, True)] + bed[(True, False)])
    out["bed_precision"] = _ratio(bed[(True, True)], bed[(True, True)] + bed[(False, True)])
    latencies = counts["exit_latencies"]
    out["exits"] = f"{counts['exits_caught']}/{counts['exits']}"
    out["exits_missed"] = counts["exits"] - counts["exits_caught"]
    out["exit_latency_mean"] = _ratio(sum(latencies), len(latencies))
    out["exit_latency_max"] = max(latencies) if latencies else None
    out["false_exits"] = counts["false_exits"]
    out["false_floor_episodes"] = counts["false_floor_episodes"]
    out["upright_in_bed"] = counts["upright_in_bed"]
    walk, stand = out["walking"]["f1"] or 0.0, out["standing"]["f1"] or 0.0
    out["motion_f1"] = (walk + stand) / 2
    return out


def objective(m: dict, key: str) -> float:
    """Higher is better. Missed bed exits and false floor episodes are safety
    regressions, so they dominate whatever `key` measures."""
    value = m[key]
    if isinstance(value, dict):
        value = value["f1"]
    return (value or 0.0) - 1.0 * m["exits_missed"] - 0.5 * m["false_floor_episodes"]


def parse_value(key: str, raw: str, defaults: ClassifyThresholds) -> Any:
    if key in TRACKER_KEYS:
        return TRACKER_KEYS[key](raw)
    current = getattr(defaults, key)
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, str):
        return raw
    return float(raw)


def split_config(config: dict[str, Any]) -> tuple[ClassifyThresholds, dict[str, Any]]:
    tracker = {k: v for k, v in config.items() if k in TRACKER_KEYS}
    thresholds = replace(
        ClassifyThresholds(), **{k: v for k, v in config.items() if k not in TRACKER_KEYS}
    )
    return thresholds, tracker


def load_clips(
    root: Path,
    clip_ids: list[str] | None,
    model: str,
    variant: str,
    zones_dir: Path | None = None,
) -> list[dict]:
    clips = []
    candidates = clip_ids or sorted(p.name for p in (root / "clips").iterdir() if p.is_dir())
    for clip_id in candidates:
        clip = root / "clips" / clip_id
        reference_path = clip / "labels" / "reference.yaml"
        cache_path = clip / "cache" / f"{model}-{variant}.jsonl"
        if not reference_path.exists() or not cache_path.exists():
            if clip_ids:
                raise SystemExit(f"{clip_id}: needs labels/reference.yaml and {cache_path.name}")
            continue
        raw = yaml.safe_load(reference_path.read_text()) or {}
        if not raw.get("confirmed_by"):
            continue
        cache_rows = read_jsonl(cache_path)
        gate_path = clip / "cache" / f"gate-{variant}.json"
        gate = (
            {int(k): v for k, v in json.loads(gate_path.read_text()).items()}
            if gate_path.exists()
            else {}
        )
        clips.append(
            {
                "id": clip_id,
                "cache": cache_rows,
                "gate": gate,
                "zones": load_zones(
                    zones_dir / f"{clip_id}.yaml" if zones_dir else clip / "zones.yaml"
                ),
                "timeline": raw["timeline"],
                "reference": reference_frames(raw["timeline"], cache_rows),
            }
        )
    return clips


def run_config(clips: list[dict], config: dict[str, Any], args: argparse.Namespace) -> dict:
    thresholds, tracker_kwargs = split_config(config)
    per_clip = {}
    for clip in clips:
        rows = replay(
            clip["cache"],
            {} if args.no_gate else clip["gate"],
            clip["zones"],
            thresholds,
            tracker_kwargs,
            model_floor=0.0 if args.model.startswith("mediapipe") else args.model_floor,
            mediapipe=args.model.startswith("mediapipe"),
        )
        per_clip[clip["id"]] = clip_counts(rows, clip["reference"], clip["timeline"])
        if args.predictions_tag:
            write_predictions(clip, rows, args.predictions_tag, args, config)
        if args.dump:
            args.dump.mkdir(parents=True, exist_ok=True)
            with (args.dump / f"{clip['id']}.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
    return per_clip


def _fmt(value: Any) -> str:
    if value is None:
        return "   -"
    if isinstance(value, float):
        return f"{value:4.2f}"
    return str(value)


COLUMNS = [
    ("acc", lambda m: m["acc"]),
    ("upr", lambda m: m["upright_acc"]),
    ("stF1", lambda m: m["standing"]["f1"]),
    ("stR", lambda m: m["standing"]["recall"]),
    ("stP", lambda m: m["standing"]["precision"]),
    ("wkF1", lambda m: m["walking"]["f1"]),
    ("wkR", lambda m: m["walking"]["recall"]),
    ("wkP", lambda m: m["walking"]["precision"]),
    ("bedR", lambda m: m["bed_recall"]),
    ("bedP", lambda m: m["bed_precision"]),
    ("flrR", lambda m: m["on_floor"]["recall"]),
    ("exits", lambda m: m["exits"]),
    ("lat", lambda m: m["exit_latency_mean"]),
    ("fExit", lambda m: m["false_exits"]),
    ("fFlr", lambda m: m["false_floor_episodes"]),
    ("upBed", lambda m: m["upright_in_bed"]),
]


def header(label_width: int) -> str:
    return f"{'config':{label_width}s} " + " ".join(f"{name:>5s}" for name, _ in COLUMNS)


def line(label: str, m: dict, label_width: int) -> str:
    return f"{label:{label_width}s} " + " ".join(f"{_fmt(get(m)):>5s}" for _, get in COLUMNS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip", action="append", help="default: every confirmed clip")
    parser.add_argument("--model", default="yolo11s-pose")
    parser.add_argument("--variant", default="letterbox640")
    parser.add_argument("--model-floor", type=float, default=0.25)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument(
        "--zones-dir",
        type=Path,
        help="read <dir>/<clip>.yaml (e.g. perceive.calibrate_bed output) instead of zones.yaml",
    )
    parser.add_argument("--set", action="append", default=[], help="key=value, fixed")
    parser.add_argument("--grid", action="append", default=[], help="key=v1,v2,... swept")
    parser.add_argument(
        "--rank",
        default="motion_f1",
        help="metric to maximise: motion_f1, acc, upright_acc, bed_recall, or a state name (F1)",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--per-clip", action="store_true", help="print each clip for the best")
    parser.add_argument("--loco", action="store_true", help="leave-one-clip-out selection")
    parser.add_argument("--confusion", action="store_true", help="standing/walking confusion")
    parser.add_argument("--dump", type=Path, help="write replayed rows (single config only)")
    parser.add_argument("--json", type=Path, help="write every config's metrics here")
    parser.add_argument(
        "--predictions-tag",
        help="write clips/<clip>/predictions/<tag>.jsonl for `video_eval visualize` "
        "(single config only)",
    )
    args = parser.parse_args(argv)
    # The tracker logs every fall drop; across a grid that drowns the table.
    logging.getLogger("perceive").setLevel(logging.WARNING)

    defaults = ClassifyThresholds()
    known = {f.name for f in fields(ClassifyThresholds)} | set(TRACKER_KEYS)

    def parse_pair(pair: str) -> tuple[str, str]:
        key, value = pair.split("=", 1)
        if key not in known:
            raise SystemExit(f"unknown key {key}; known: {sorted(known)}")
        return key, value

    fixed = {k: parse_value(k, v, defaults) for k, v in map(parse_pair, args.set)}
    axes = [
        (k, [parse_value(k, v, defaults) for v in values.split(",")])
        for k, values in map(parse_pair, args.grid)
    ]
    configs = [
        {**fixed, **dict(zip([k for k, _ in axes], combo, strict=True))}
        for combo in itertools.product(*[values for _, values in axes])
    ]
    if (args.dump or args.predictions_tag) and len(configs) > 1:
        raise SystemExit("--dump and --predictions-tag need a single config (no --grid)")

    clips = load_clips(
        args.data_root.resolve(), args.clip, args.model, args.variant, args.zones_dir
    )
    if not clips:
        raise SystemExit("no clip has both a confirmed reference and the requested cache")
    print(f"clips: {', '.join(c['id'] for c in clips)} | {args.model}-{args.variant}")

    results = []
    for config in configs:
        per_clip = run_config(clips, config, args)
        pooled = metrics(pool(list(per_clip.values())))
        results.append({"config": config, "per_clip": per_clip, "metrics": pooled})

    def label(config: dict) -> str:
        swept = {k: config[k] for k, _ in axes}
        return (
            " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in swept.items())
            or "defaults"
        )

    ranked = sorted(results, key=lambda r: objective(r["metrics"], args.rank), reverse=True)
    width = max(12, *(len(label(r["config"])) for r in ranked[: args.top]))
    print(header(width))
    for result in ranked[: args.top]:
        print(line(label(result["config"]), result["metrics"], width))

    best = ranked[0]
    if args.per_clip:
        print("\nper clip, best config:")
        for clip_id, counts in best["per_clip"].items():
            print(line(clip_id[-9:], metrics(counts), width))
    if args.confusion:
        confusion = pool(list(best["per_clip"].values()))["confusion"]
        for truth in ("standing", "walking", "sitting_up", "in_bed"):
            row = {p: v for (t, p), v in confusion.items() if t == truth}
            print(
                f"  {truth:>10s} -> "
                + ", ".join(f"{p} {v}" for p, v in sorted(row.items(), key=lambda kv: -kv[1]))
            )

    if args.loco and len(clips) > 1 and len(configs) > 1:
        print(
            "\nleave-one-clip-out (config picked on the other clips, scored on the held-out one):"
        )
        held_parts = []
        for clip in clips:
            others = [c["id"] for c in clips if c["id"] != clip["id"]]
            pick = max(
                results,
                key=lambda r: objective(
                    metrics(pool([r["per_clip"][o] for o in others])), args.rank
                ),
            )
            held = pick["per_clip"][clip["id"]]
            held_parts.append(held)
            print(line(f"{clip['id'][-9:]} <- {label(pick['config'])}", metrics(held), width + 13))
        print(line("pooled held-out", metrics(pool(held_parts)), width + 13))

    if args.json:
        args.json.write_text(
            json.dumps(
                [{"config": r["config"], "metrics": r["metrics"]} for r in results],
                indent=1,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
