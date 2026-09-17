"""Human-readable and JSON reports for dialogue model comparisons."""

from __future__ import annotations

import json

from dialogue_bench.scoring import ModelResult


def result_to_dict(result: ModelResult, *, include_text: bool = False) -> dict[str, object]:
    scenarios: list[dict[str, object]] = []
    for item in result.scenarios:
        row: dict[str, object] = {
            "id": item.scenario_id,
            "expected_intent": item.expected_intent.value,
            "actual_intent": item.actual_intent.value if item.actual_intent else None,
            "intent_correct": item.intent_correct,
            "composition_safe": item.composition_safe,
            "composition_failure": item.composition_failure,
            "composition_copies_template": item.composition_copies_template,
            "interpret_latency_seconds": item.interpret_latency_seconds,
            "compose_latency_seconds": item.compose_latency_seconds,
        }
        if include_text:
            row["composition_text"] = item.composition_text
        scenarios.append(row)
    return {
        "model": result.model,
        "scenario_count": result.scenario_count,
        "intent_correct": result.intent_correct,
        "intent_accuracy": result.intent_accuracy,
        "intent_accuracy_by_class": result.intent_accuracy_by_class,
        "intent_confusion_matrix": result.confusion,
        "mean_interpret_latency_seconds": result.mean_interpret_latency_seconds,
        "max_interpret_latency_seconds": result.max_interpret_latency_seconds,
        "safe_compositions": result.safe_compositions,
        "composition_pass_rate": result.composition_pass_rate,
        "mean_compose_latency_seconds": result.mean_compose_latency_seconds,
        "max_compose_latency_seconds": result.max_compose_latency_seconds,
        "composition_failures_by_reason": result.failures_by_reason,
        "template_copies": result.template_copies,
        "distinct_compositions": result.distinct_compositions,
        "scenarios": scenarios,
    }


def print_report(results: list[ModelResult]) -> None:
    print("Dialogue regression bench (issue #17)")
    print("=" * 72)
    print(f"{'model':28} {'intent':>14} {'safe composition':>20}")
    print("-" * 72)
    for result in results:
        intent = f"{result.intent_correct}/{result.scenario_count}"
        safe = f"{result.safe_compositions}/{result.scenario_count}"
        print(f"{result.model:28} {intent:>14} {safe:>20}")
        per_class = ", ".join(
            f"{label}={'n/a' if score is None else f'{score:.0%}'}"
            for label, score in result.intent_accuracy_by_class.items()
        )
        print(f"  intent by class: {per_class}")
        print(
            "  end-to-end latency: "
            f"interpret mean {result.mean_interpret_latency_seconds:.3f}s, "
            f"max {result.max_interpret_latency_seconds:.3f}s; "
            f"compose mean {result.mean_compose_latency_seconds:.3f}s, "
            f"max {result.max_compose_latency_seconds:.3f}s"
        )
        print(
            f"  composition variety: {result.distinct_compositions} distinct, "
            f"{result.template_copies} copied the caregiver phrase"
        )
        for reason, count in sorted(result.failures_by_reason.items()):
            print(f"  composition failures ({count}): {reason}")


def print_json_report(results: list[ModelResult], *, include_text: bool = False) -> None:
    payload = {"results": [result_to_dict(r, include_text=include_text) for r in results]}
    print(json.dumps(payload, indent=2))
