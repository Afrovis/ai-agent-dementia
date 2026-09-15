"""Score offline perception predictions against a confirmed reference timeline."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from perception_bench.scoring import ACCURACY_TARGET, LATENCY_TARGET_S, score_predictions

from video_eval.common import matching_meta, read_jsonl, update_index, write_meta
from video_eval.paths import EvalPaths
from video_eval.reconcile import validate_timeline

TARGET_RECALL = ACCURACY_TARGET
TARGET_FLOOR_DELAY_S = LATENCY_TARGET_S
EVENT_WINDOW_S = 30.0
EARLY_EVENT_TOLERANCE_S = 3.0
FALSE_TRANSITION_TOLERANCE_S = 10.0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_at(t_s: float, timeline: list[dict[str, Any]]) -> tuple[str, str] | None:
    for row in timeline:
        if float(row["from_s"]) <= t_s < float(row["to_s"]):
            return str(row["state"]), str(row["zone"])
    return None


def expand_reference(
    frames: list[dict[str, Any]], timeline: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    expanded: dict[int, dict[str, Any]] = {}
    for frame in frames:
        value = _reference_at(float(frame["t_s"]), timeline)
        if value is not None:
            expanded[int(frame["frame_index"])] = {
                "state": value[0],
                "zone": value[1],
                "t_s": float(frame["t_s"]),
            }
    return expanded


def _accuracy_dict(pairs: list[tuple[str, str]], *, collapse_upright: bool) -> dict[str, Any]:
    if collapse_upright:
        pairs = [
            (
                "standing" if actual in {"standing", "walking"} else actual,
                "standing" if predicted in {"standing", "walking"} else predicted,
            )
            for actual, predicted in pairs
        ]
    result = score_predictions(pairs)
    per_state: dict[str, Any] = {}
    for state, stats in result.per_state.items():
        if collapse_upright and state == "walking":
            continue
        label = "upright" if collapse_upright and state == "standing" else state
        per_state[label] = {
            **asdict(stats),
            "state": label,
            "precision": stats.precision,
            "recall": stats.recall,
        }
    confusion = result.confusion
    if collapse_upright:
        confusion = {
            ("upright" if actual == "standing" else actual): {
                ("upright" if predicted == "standing" else predicted): count
                for predicted, count in values.items()
                if predicted != "walking"
            }
            for actual, values in result.confusion.items()
            if actual != "walking"
        }
    return {
        "frame_count": result.frame_count,
        "overall_accuracy": result.overall_accuracy,
        "per_state": per_state,
        "confusion": confusion,
    }


def _frame_metrics(
    predictions: list[dict[str, Any]], reference: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    available = [row for row in predictions if int(row["frame_index"]) in reference]
    admitted = [row for row in available if not row.get("gated", False)]

    def pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [
            (reference[int(row["frame_index"])]["state"], row.get("state") or "undetected")
            for row in rows
        ]

    detection: dict[str, Any] = {}
    states = sorted({row["state"] for row in reference.values()})
    for state in states:
        all_state = [
            row for row in available if reference[int(row["frame_index"])]["state"] == state
        ]
        admitted_state = [
            row for row in admitted if reference[int(row["frame_index"])]["state"] == state
        ]

        def rate(rows: list[dict[str, Any]]) -> float | None:
            return (
                None if not rows else sum(row.get("detected") is True for row in rows) / len(rows)
            )

        detection[state] = {
            "all": rate(all_state),
            "all_support": len(all_state),
            "admitted": rate(admitted_state),
            "admitted_support": len(admitted_state),
        }
    return {
        "all": {
            "exact": _accuracy_dict(pairs(available), collapse_upright=False),
            "upright_collapsed": _accuracy_dict(pairs(available), collapse_upright=True),
        },
        "admitted": {
            "exact": _accuracy_dict(pairs(admitted), collapse_upright=False),
            "upright_collapsed": _accuracy_dict(pairs(admitted), collapse_upright=True),
        },
        "detection_rate": detection,
    }


def _reference_events(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for row in timeline:
        state = "upright" if row["state"] in {"standing", "walking"} else row["state"]
        previous_state = None
        if previous:
            previous_state = (
                "upright" if previous["state"] in {"standing", "walking"} else previous["state"]
            )
        if state in {"sitting_up", "upright", "on_floor", "absent"} and state != previous_state:
            events.append({"kind": "state", "value": state, "t_s": float(row["from_s"])})
        if row["zone"] in {"door", "bathroom_path"} and (
            previous is None or previous["zone"] != row["zone"]
        ):
            events.append({"kind": "zone", "value": row["zone"], "t_s": float(row["from_s"])})
        previous = row
    return events


def _prediction_events(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    prior_state = prior_zone = None
    for row in predictions:
        if row.get("state") is None:
            continue
        state = "upright" if row["state"] in {"standing", "walking"} else row["state"]
        if state in {"sitting_up", "upright", "on_floor", "absent"} and state != prior_state:
            events.append({"kind": "state", "value": state, "t_s": float(row["t_s"])})
        zone = row.get("zone")
        if zone in {"door", "bathroom_path"} and zone != prior_zone:
            events.append({"kind": "zone", "value": zone, "t_s": float(row["t_s"])})
        prior_state, prior_zone = state, zone
    return events


def event_metrics(
    predictions: list[dict[str, Any]], timeline: list[dict[str, Any]]
) -> dict[str, Any]:
    references = _reference_events(timeline)
    predicted = _prediction_events(predictions)
    delays: list[dict[str, Any]] = []
    for expected in references:
        before = [row for row in predictions if float(row["t_s"]) <= expected["t_s"]]
        current = (
            max(before, key=lambda row: float(row["t_s"]))
            if before
            else min(predictions, key=lambda row: float(row["t_s"]), default=None)
        )
        current_value = None
        if current is not None:
            if expected["kind"] == "state" and current.get("state") is not None:
                current_value = (
                    "upright" if current["state"] in {"standing", "walking"} else current["state"]
                )
            elif expected["kind"] == "zone":
                current_value = current.get("zone")
        run_onset = max(
            (
                event["t_s"]
                for event in predicted
                if current is not None
                and event["kind"] == expected["kind"]
                and event["value"] == expected["value"]
                and event["t_s"] <= float(current["t_s"])
            ),
            default=None,
        )
        early_onset = (
            run_onset
            if current_value == expected["value"]
            and run_onset is not None
            and expected["t_s"] - EARLY_EVENT_TOLERANCE_S <= run_onset < expected["t_s"]
            else None
        )
        hits = [
            event
            for event in predicted
            if event["kind"] == expected["kind"]
            and event["value"] == expected["value"]
            and expected["t_s"] <= event["t_s"] <= expected["t_s"] + EVENT_WINDOW_S
        ]
        if early_onset is not None:
            onset_offset = early_onset - expected["t_s"]
            delay = 0.0
        else:
            delay = None if not hits else min(event["t_s"] for event in hits) - expected["t_s"]
            onset_offset = delay
        delays.append(
            {
                **expected,
                "delay_s": delay,
                "onset_offset_s": onset_offset,
                "missed": delay is None,
            }
        )
    false = [
        event
        for event in predicted
        if not any(
            ref["kind"] == event["kind"]
            and ref["value"] == event["value"]
            and abs(ref["t_s"] - event["t_s"]) <= FALSE_TRANSITION_TOLERANCE_S
            for ref in references
        )
    ]
    duration_s = max((float(row["t_s"]) for row in predictions), default=0.0) + 0.5
    return {
        "delays": delays,
        "false_transition_count": len(false),
        "false_transitions": false,
        "false_transitions_per_minute": None if duration_s <= 0 else len(false) / (duration_s / 60),
    }


def _gate_verdicts(frame_metrics: dict[str, Any], events: dict[str, Any]) -> dict[str, Any]:
    exact = frame_metrics["all"]["exact"]["per_state"]
    gates: dict[str, Any] = {}
    for state in ("standing", "on_floor"):
        recall = exact[state]["recall"]
        gates[f"{state}_recall"] = {
            "measured": recall is not None,
            "met": None if recall is None else recall >= TARGET_RECALL,
            "value": recall,
            "target": TARGET_RECALL,
        }
    floor_delays: list[float | None] = [
        row["delay_s"]
        for row in events["delays"]
        if row["kind"] == "state" and row["value"] == "on_floor"
    ]
    measured = bool(floor_delays)
    detected_delays = [value for value in floor_delays if value is not None]
    maximum = max(detected_delays) if detected_delays else None
    gates["on_floor_delay"] = {
        "measured": measured,
        "met": (
            None
            if not measured
            else len(detected_delays) == len(floor_delays)
            and maximum is not None
            and maximum <= TARGET_FLOOR_DELAY_S
        ),
        "missed": len(floor_delays) - len(detected_delays),
        "value_s": maximum,
        "target_s": TARGET_FLOOR_DELAY_S,
    }
    return gates


def _percent(value: float | None) -> str:
    return "not measurable" if value is None else f"{value:.1%}"


def _confusion_markdown(confusion: dict[str, dict[str, int]]) -> list[str]:
    preferred = ["in_bed", "sitting_up", "standing", "walking", "on_floor", "absent"]
    columns = preferred + sorted(
        {value for row in confusion.values() for value in row if value not in preferred}
    )
    lines = [
        "| Actual \\ predicted | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for actual in preferred:
        row = confusion.get(actual, {})
        lines.append(
            f"| {actual} | " + " | ".join(str(row.get(value, 0)) for value in columns) + " |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    all_exact = report["frames"]["all"]["exact"]
    collapsed = report["frames"]["all"]["upright_collapsed"]
    admitted = report["frames"]["admitted"]["exact"]
    lines = [
        f"# Video evaluation: {report['clip_id']} / {report['tag']}",
        "",
        "## Frame metrics",
        "",
        f"- Overall accuracy (all): {_percent(all_exact['overall_accuracy'])}",
        f"- Overall accuracy (admitted): {_percent(admitted['overall_accuracy'])}",
        f"- Upright-collapsed accuracy (all): {_percent(collapsed['overall_accuracy'])}",
        f"- Upright recall (all): {_percent(collapsed['per_state']['upright']['recall'])}",
        "",
        "| State | Recall | Precision | Support |",
        "| --- | ---: | ---: | ---: |",
    ]
    for state, row in all_exact["per_state"].items():
        lines.append(
            f"| {state} | {_percent(row['recall'])} | "
            f"{_percent(row['precision'])} | {row['support']} |"
        )
    lines.extend(["", "### All-frame confusion matrix", ""])
    lines.extend(_confusion_markdown(all_exact["confusion"]))
    lines.extend(
        [
            "",
            "### Detection rate by reference state",
            "",
            "| State | All frames | Admitted frames |",
            "| --- | ---: | ---: |",
        ]
    )
    for state, row in report["frames"]["detection_rate"].items():
        lines.append(f"| {state} | {_percent(row['all'])} | {_percent(row['admitted'])} |")
    lines.extend(["", "## Event metrics", ""])
    for event in report["events"]["delays"]:
        value = "missed"
        if not event["missed"]:
            onset_offset = event.get("onset_offset_s", event["delay_s"])
            value = f"{event['delay_s']:.1f}s (onset offset {onset_offset:+.1f}s)"
        lines.append(f"- {event['kind']} `{event['value']}` at {event['t_s']:.1f}s: {value}")
    false_rate = report["events"]["false_transitions_per_minute"]
    rendered_rate = "not measurable" if false_rate is None else f"{false_rate:.3f}"
    lines.append(f"- False transitions per minute: {rendered_rate}")
    lines.extend(["", "## PLAN.md gates", ""])
    for name, gate in report["gates"].items():
        verdict = (
            "not measurable" if not gate["measured"] else ("MET" if gate["met"] else "NOT MET")
        )
        lines.append(f"- `{name}`: {verdict}")
    return "\n".join(lines) + "\n"


def score_clip(
    clip_id: str,
    *,
    root: Path | None = None,
    tag: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    paths = EvalPaths.for_clip(clip_id, root)
    reference_path = paths.labels / "reference.yaml"
    if not reference_path.exists():
        raise RuntimeError("confirmed labels/reference.yaml is required; drafts are never scored")
    reference_raw = yaml.safe_load(reference_path.read_text(encoding="utf-8")) or {}
    if not reference_raw.get("confirmed_by") or not reference_raw.get("confirmed_at"):
        raise RuntimeError("reference.yaml is not stamped as human-confirmed")
    validate_timeline(reference_raw.get("timeline"))
    prediction_paths = sorted(paths.predictions.glob("*.jsonl"))
    if tag:
        prediction_paths = [paths.predictions / f"{tag}.jsonl"]
    if not prediction_paths:
        raise FileNotFoundError("no prediction JSONL files found")
    frames = read_jsonl(paths.frames)
    expanded = expand_reference(frames, reference_raw.get("timeline") or [])
    results: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        started = time.monotonic()
        current_tag = prediction_path.stem
        report_path = paths.reports / f"{current_tag}.json"
        markdown_path = paths.reports / f"{current_tag}.md"
        meta_path = paths.reports / f"{current_tag}.meta.json"
        parameters = {
            "clip_id": clip_id,
            "prediction_sha256": _digest(prediction_path),
            "reference_sha256": _digest(reference_path),
            "tag": current_tag,
        }
        if (
            not force
            and report_path.exists()
            and markdown_path.exists()
            and matching_meta(meta_path, parameters)
        ):
            results.append({"tag": current_tag, "status": "skipped"})
            continue
        predictions = read_jsonl(prediction_path)
        frames_result = _frame_metrics(predictions, expanded)
        events = event_metrics(predictions, reference_raw["timeline"])
        report = {
            "clip_id": clip_id,
            "tag": current_tag,
            "reference_confirmed_by": reference_raw["confirmed_by"],
            "reference_confirmed_at": str(reference_raw["confirmed_at"]),
            "frames": frames_result,
            "events": events,
            "gates": _gate_verdicts(frames_result, events),
        }
        paths.reports.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        write_meta(meta_path, command="score", parameters=parameters, started_at=started)
        update_index(paths.root, clip_id, f"score_{current_tag}", "complete")
        results.append({"tag": current_tag, "status": "complete", "gates": report["gates"]})
    return {"status": "complete", "reports": results}
