"""Human-readable and JSON reports for the decision benchmark."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from decision_bench.runner import Trace
from decision_bench.scoring import ModelResult


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def result_to_dict(result: ModelResult) -> dict[str, object]:
    """Convert one model result to a stable JSON-compatible mapping."""
    scenarios = []
    for scenario in result.scenarios:
        checkpoints = [
            {
                "id": item.checkpoint_id,
                "status": item.status,
                "reasons": list(item.reasons),
                "observations": {key: list(value) for key, value in item.observations.items()},
                "escalation_latency": item.escalation_latency,
            }
            for item in scenario.checkpoints
        ]
        scenarios.append(
            {
                "id": scenario.scenario_id,
                "category": scenario.category,
                "noise_of": scenario.noise_of,
                "checkpoints": checkpoints,
                "wording_failures": scenario.wording_failures,
                "llm_errors": scenario.llm_errors,
                "llm_none": scenario.llm_none,
                "wall_time_seconds": scenario.wall_time_seconds,
            }
        )
    return {
        "model": result.model,
        "labelled_checkpoints": result.labelled_count,
        "passed_checkpoints": result.passed_count,
        "pass_rate": result.pass_rate,
        "pass_rate_by_category": result.pass_rate_by_category,
        "pass_rate_clean_noisy": result.pass_rate_clean_noisy,
        "critical_violations": [list(item) for item in result.critical_violations],
        "escalation_latency": result.escalation_summary,
        "wording_failures": result.wording_failures,
        "review_checkpoints": result.review_count,
        "doubt_checkpoints": result.doubt_count,
        "points": result.points,
        "unlabelled_checkpoints": result.unlabelled_count,
        "llm_errors": result.llm_errors,
        "llm_none": result.llm_none,
        "wall_time_seconds": result.wall_time_seconds,
        "scenarios": scenarios,
    }


def json_payload(
    results: list[ModelResult], citation_warnings: list[str] | None = None
) -> dict[str, object]:
    return {
        "citation_warnings": citation_warnings or [],
        "results": [result_to_dict(result) for result in results],
    }


def print_json_report(
    results: list[ModelResult], citation_warnings: list[str] | None = None
) -> None:
    print(json.dumps(json_payload(results, citation_warnings), indent=2))


def write_json_report(
    path: Path, results: list[ModelResult], citation_warnings: list[str] | None = None
) -> None:
    path.write_text(json.dumps(json_payload(results, citation_warnings), indent=2) + "\n")


def _print_table(rows: list[tuple[str, list[str]]], models: list[str]) -> None:
    label_width = max([len(row[0]) for row in rows] + [6])
    widths = [max(12, len(model)) for model in models]
    print(
        f"{'measure':<{label_width}} "
        + " ".join(f"{model:>{width}}" for model, width in zip(models, widths, strict=True))
    )
    print("-" * (label_width + 1 + sum(widths) + max(0, len(widths) - 1)))
    for label, values in rows:
        print(
            f"{label:<{label_width}} "
            + " ".join(f"{value:>{width}}" for value, width in zip(values, widths, strict=True))
        )


def _summary(observations: dict[str, tuple[str, ...]]) -> str:
    parts = []
    for key in ("phases", "goals", "strategies", "notifies"):
        values = observations[key]
        if values:
            parts.append(f"{key}={','.join(values)}")
    if observations["says"]:
        parts.append("says=" + " | ".join(repr(value) for value in observations["says"]))
    return "; ".join(parts) or "no recorded actions"


def print_report(results: list[ModelResult], citation_warnings: list[str] | None = None) -> None:
    """Print aggregate comparisons and review-oriented checkpoint details."""
    print("Decision benchmark")
    print("=" * 72)
    for warning in citation_warnings or []:
        print(f"WARNING: {warning}")
    models = [item.model for item in results]
    categories = sorted({category for item in results for category in item.pass_rate_by_category})
    rows = [("overall", [_rate(item.pass_rate) for item in results])]
    rows.extend(
        (
            f"category/{category}",
            [_rate(item.pass_rate_by_category.get(category)) for item in results],
        )
        for category in categories
    )
    for clean in ("clean", "noisy"):
        rows.append((clean, [_rate(item.pass_rate_clean_noisy.get(clean)) for item in results]))
    _print_table(rows, models)

    for result in results:
        print(f"\n{result.model}")
        print(f"  critical violations: {len(result.critical_violations)}")
        for scenario, checkpoint, reason in result.critical_violations:
            print(f"    {scenario}/{checkpoint}: {reason}")
        latency = result.escalation_summary
        if latency:
            print(
                "  escalation latency: "
                f"min {latency['min']:g}s, median {latency['median']:g}s, max {latency['max']:g}s"
            )
        else:
            print("  escalation latency: n/a")
        wording = ", ".join(
            f"{pattern}={count}" for pattern, count in sorted(result.wording_failures.items())
        )
        print(f"  wording failures: {wording or 'none'}")
        print(
            f"  checkpoints: {result.review_count} review, {result.doubt_count} doubt, "
            f"{result.unlabelled_count} unlabelled"
        )
        print(f"  LLM calls: {result.llm_errors} errors, {result.llm_none} None returns")
        for scenario in result.scenarios:
            for checkpoint in scenario.checkpoints:
                if checkpoint.status == "unlabelled":
                    print(
                        f"  {scenario.scenario_id}/{checkpoint.checkpoint_id}: "
                        f"{_summary(checkpoint.observations)}"
                    )


def _entry_text(kind: str, data: dict[str, object]) -> str:
    if kind == "PersonState":
        repeat = " heartbeat" if data.get("repeated") else ""
        return f"IN person{repeat} {data.get('state')}@{data.get('zone')}"
    if kind == "Utterance":
        return f"IN utterance {data.get('text')!r}"
    if kind == "State":
        return (
            f"STATE {data.get('phase')} goal={data.get('goal')} "
            f"strategy={data.get('strategy') or '-'}"
        )
    if kind == "Say":
        return f"Say[{data.get('strategy')}] {data.get('text')!r}"
    if kind == "Notify":
        return f"Notify[{data.get('level')}] {data.get('title')!r}"
    if kind in {"SessionState", "GoalChanged", "LightCommand", "Show"}:
        fields = {
            key: value
            for key, value in data.items()
            if key
            in {
                "phase",
                "goal",
                "strategy_index",
                "from_goal",
                "to_goal",
                "state",
                "headline",
            }
        }
        return f"{kind} {fields}"
    if kind.startswith("LLM"):
        return f"{kind} {data}"
    return kind


def print_trace(trace: Trace, *, file=None) -> None:
    """Print a compact timeline, grouping all entries emitted in the same second."""
    grouped: dict[float, list[str]] = defaultdict(list)
    for entry in trace.entries:
        grouped[entry.t].append(_entry_text(entry.kind, entry.data))
    print(f"\nTRACE {trace.scenario_id} ({trace.start:%H:%M}, end {trace.end_t:g}s)", file=file)
    for second, items in grouped.items():
        print(f"  {second:7.1f}s  " + " | ".join(items), file=file)
