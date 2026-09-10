"""Turns tier results into a readable stdout report or a `--json` one.

Kept separate from `run.py`'s orchestration and each tier's own module, the
same reasoning as `scoring.py`: presentation should not know how a number
was produced, only how to show it -- and how to show "not measured"
distinctly from "measured and failed", which is the one thing every part
of this report has to get right (this issue's hard constraint: missing
data is not a failing test).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from perception_bench.daylight import Tier2Result
from perception_bench.infrared import Tier3Result
from perception_bench.scoring import (
    ACCURACY_TARGET,
    LATENCY_TARGET_S,
    TARGET_STATES,
    AccuracyResult,
    LatencyResult,
    target_states_meet_accuracy,
)
from perception_bench.synthetic import ClipRun


@dataclass(frozen=True)
class Verdict:
    """One pass/fail/not-measured line item the exit code is built from.

    `advisory` marks a verdict that is reported honestly -- including a
    real FAIL, never a softened one -- but never contributes to the exit
    code. Tier 1 is advisory: it runs against synthetic, generated-in-
    process clips as a smoke test, not the real recorded footage the
    >95%/<2s targets in PLAN.md section 12 are defined against."""

    label: str
    passed: bool | None  # None means "not measured"
    detail: str
    advisory: bool = False


@dataclass(frozen=True)
class BenchReport:
    tier1_runs: list[ClipRun]
    tier1_accuracy: AccuracyResult
    tier1_latency: LatencyResult
    tier2: Tier2Result
    tier3: Tier3Result
    verdicts: list[Verdict]

    @property
    def exit_code(self) -> int:
        """Non-zero if any measured, non-advisory verdict failed. A tier
        with nothing measured (`passed is None`) never contributes to a
        non-zero exit, per this issue's "missing data is not a failing
        test". Advisory verdicts (tier 1) are reported with their real
        numbers but never gate the exit code either -- see `Verdict`."""
        return 1 if any(v.passed is False and not v.advisory for v in self.verdicts) else 0


def _tier1_verdicts(accuracy: AccuracyResult, latency: LatencyResult) -> list[Verdict]:
    verdicts: list[Verdict] = []
    target_hits = target_states_meet_accuracy(accuracy)
    for state in TARGET_STATES:
        stats = accuracy.per_state[state]
        passed = target_hits[state]
        recall_str = "not measured (no frames)" if stats.recall is None else f"{stats.recall:.1%}"
        verdicts.append(
            Verdict(
                label=f"tier1 {state} recall >= {ACCURACY_TARGET:.0%}",
                passed=passed,
                detail=recall_str,
                advisory=True,
            )
        )

    within = latency.within_target(LATENCY_TARGET_S)
    if within is None:
        detail = "not measured (no transitions scripted)"
    else:
        detail = f"max {latency.max_latency_s:.2f}s, mean {latency.mean_latency_s:.2f}s"
    verdicts.append(
        Verdict(
            label=f"tier1 latency <= {LATENCY_TARGET_S:.0f}s",
            passed=within,
            detail=detail,
            advisory=True,
        )
    )
    return verdicts


def _tier2_verdicts(tier2: Tier2Result) -> list[Verdict]:
    if tier2.accuracy is None:
        return [
            Verdict(
                label="tier2 daylight accuracy",
                passed=None,
                detail=tier2.skipped_reason or "skipped",
            )
        ]
    verdicts: list[Verdict] = []
    target_hits = target_states_meet_accuracy(tier2.accuracy)
    label_suffix = " (night-degraded)" if tier2.degraded else ""
    for state in TARGET_STATES:
        stats = tier2.accuracy.per_state[state]
        passed = target_hits[state]
        recall_str = "not measured (no frames)" if stats.recall is None else f"{stats.recall:.1%}"
        verdicts.append(
            Verdict(
                label=f"tier2{label_suffix} {state} recall >= {ACCURACY_TARGET:.0%}",
                passed=passed,
                detail=recall_str,
            )
        )
    return verdicts


def _tier3_verdicts(tier3: Tier3Result) -> list[Verdict]:
    return [
        Verdict(
            label="tier3 infrared bench",
            passed=None,
            detail=tier3.missing_reason or f"{tier3.clip_count} clip(s) scored",
        )
    ]


def build_report(
    tier1_runs: list[ClipRun],
    tier1_accuracy: AccuracyResult,
    tier1_latency: LatencyResult,
    tier2: Tier2Result,
    tier3: Tier3Result,
) -> BenchReport:
    verdicts = (
        _tier1_verdicts(tier1_accuracy, tier1_latency)
        + _tier2_verdicts(tier2)
        + _tier3_verdicts(tier3)
    )
    return BenchReport(
        tier1_runs=tier1_runs,
        tier1_accuracy=tier1_accuracy,
        tier1_latency=tier1_latency,
        tier2=tier2,
        tier3=tier3,
        verdicts=verdicts,
    )


def _format_confusion_matrix(accuracy: AccuracyResult) -> str:
    from perception_bench.scoring import STATE_NAMES

    header = "actual \\ predicted".ljust(20) + "".join(s[:10].rjust(11) for s in STATE_NAMES)
    lines = [header]
    for actual in STATE_NAMES:
        row = accuracy.confusion.get(actual, {})
        cells = "".join(str(row.get(pred, 0)).rjust(11) for pred in STATE_NAMES)
        lines.append(actual.ljust(20) + cells)
    return "\n".join(lines)


def _format_verdict(v: Verdict) -> str:
    marker = "PASS" if v.passed else ("FAIL" if v.passed is False else "n/a ")
    tag = " (advisory, smoke-test, does not gate exit code)" if v.advisory else ""
    return f"  [{marker}] {v.label}{tag}: {v.detail}"


def print_report(report: BenchReport) -> None:
    """Print the human-readable report to stdout."""
    print("Perception bench (issue #11)")
    print("=" * 60)

    print("\nTier 1: synthetic scripted clips (advisory / smoke-test, does not gate exit code)")
    print("-" * 60)
    print(f"clips run: {len(report.tier1_runs)}")
    for run in report.tier1_runs:
        if run.undetected_transitions:
            print(f"  {run.clip_name}: undetected transitions: {run.undetected_transitions}")
    overall = report.tier1_accuracy.overall_accuracy
    if overall is not None:
        print(f"overall frame accuracy: {overall:.1%}")
    else:
        print("overall frame accuracy: n/a")
    print("confusion matrix (frame counts):")
    print(_format_confusion_matrix(report.tier1_accuracy))

    print("\nTier 2: daylight footage (IndoorActionDataset)")
    print("-" * 60)
    if report.tier2.accuracy is None:
        print(f"SKIPPED: {report.tier2.skipped_reason}")
    else:
        print(f"clips scored: {report.tier2.clip_count}, degraded={report.tier2.degraded}")
        overall = report.tier2.accuracy.overall_accuracy
        print(f"overall frame accuracy: {overall:.1%}" if overall is not None else "n/a")
        print("confusion matrix (frame counts):")
        print(_format_confusion_matrix(report.tier2.accuracy))
        print("note: cannot measure latency or in_bed (see daylight.py docstring)")

    print("\nTier 3: infrared footage from the actual room")
    print("-" * 60)
    if report.tier3.missing_reason:
        print(f"NOT MEASURED: {report.tier3.missing_reason}")
    else:
        print(f"clips scored: {report.tier3.clip_count}")

    print("\nVerdicts")
    print("-" * 60)
    for verdict in report.verdicts:
        print(_format_verdict(verdict))

    advisory_miss = any(v.passed is False and v.advisory for v in report.verdicts)
    print()
    if report.exit_code != 0:
        print("Result: at least one measured target was missed.")
    elif advisory_miss:
        print(
            "Result: no measured target was missed. "
            "(Advisory tier 1 targets missed, see above -- does not gate.)"
        )
    else:
        print("Result: no measured target missed. (Tiers marked n/a were not measured.)")


def _per_state_dict(accuracy: AccuracyResult) -> dict:
    """`asdict` only serialises dataclass fields, and `PerStateStats.recall`
    /`.precision` are computed properties, not fields -- add them by hand
    so `--json` output actually carries the numbers the stdout report
    shows."""
    out = {}
    for state, stats in accuracy.per_state.items():
        row = asdict(stats)
        row["recall"] = stats.recall
        row["precision"] = stats.precision
        out[state] = row
    return out


def report_to_dict(report: BenchReport) -> dict:
    """A JSON-serialisable dict, for `--json`."""
    return {
        "tier1": {
            "clips": [
                {
                    "name": run.clip_name,
                    "undetected_transitions": run.undetected_transitions,
                    "latency_samples_s": run.latency.samples,
                }
                for run in report.tier1_runs
            ],
            "overall_accuracy": report.tier1_accuracy.overall_accuracy,
            "confusion_matrix": report.tier1_accuracy.confusion,
            "per_state": _per_state_dict(report.tier1_accuracy),
        },
        "tier2": {
            "measured": report.tier2.accuracy is not None,
            "skipped_reason": report.tier2.skipped_reason,
            "degraded": report.tier2.degraded,
            "clip_count": report.tier2.clip_count,
            "overall_accuracy": (
                report.tier2.accuracy.overall_accuracy if report.tier2.accuracy else None
            ),
            "confusion_matrix": (
                report.tier2.accuracy.confusion if report.tier2.accuracy else None
            ),
            "per_state": (
                _per_state_dict(report.tier2.accuracy) if report.tier2.accuracy else None
            ),
        },
        "tier3": {
            "measured": report.tier3.missing_reason is None,
            "missing_reason": report.tier3.missing_reason,
            "clip_count": report.tier3.clip_count,
        },
        "verdicts": [asdict(v) for v in report.verdicts],
        "exit_code": report.exit_code,
    }


def print_json_report(report: BenchReport) -> None:
    print(json.dumps(report_to_dict(report), indent=2))
