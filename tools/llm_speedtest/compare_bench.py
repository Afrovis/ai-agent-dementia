"""Side-by-side comparison of dialogue-bench JSON results from run_bench.sh.

  .venv-mlx/bin/python tools/llm_speedtest/compare_bench.py tools/llm_speedtest/results/bench-*/

Writes ``comparison.md`` into the results directory and prints the summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dialogue_bench.scenarios import load_scenarios


def main() -> None:
    out_dir = Path(sys.argv[1])
    results = [
        r
        for f in sorted(out_dir.glob("*.json"))
        for r in json.loads(f.read_text())["results"]
    ]
    names = [r["model"].split("/")[-1] for r in results]
    by_id = [{s["id"]: s for s in r["scenarios"]} for r in results]

    lines = [
        (
            "| model | intent acc | wants_to_leave | safe replies | interpret s (mean/max) "
            "| compose s (mean/max) | turn s |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in zip(names, results):
        lines.append(
            f"| {name} | {r['intent_accuracy']:.0%} "
            f"| {r['intent_accuracy_by_class']['wants_to_leave']:.0%} "
            f"| {r['safe_compositions']}/{r['scenario_count']} "
            f"| {r['mean_interpret_latency_seconds']:.2f} / {r['max_interpret_latency_seconds']:.2f} "
            f"| {r['mean_compose_latency_seconds']:.2f} / {r['max_compose_latency_seconds']:.2f} "
            f"| {r['mean_interpret_latency_seconds'] + r['mean_compose_latency_seconds']:.2f} |"
        )
    summary = "\n".join(lines)

    misses = ["| scenario | utterance | expected | " + " | ".join(names) + " |"]
    misses.append("|---" * (3 + len(names)) + "|")
    replies = ["| scenario | utterance | " + " | ".join(names) + " |"]
    replies.append("|---" * (2 + len(names)) + "|")
    for scenario in load_scenarios():
        rows = [ids.get(scenario.id, {}) for ids in by_id]
        utterance = scenario.utterance.replace("|", "/")
        if not all(row.get("intent_correct") for row in rows):
            cells = [
                row.get("actual_intent") or "—"
                if not row.get("intent_correct")
                else "ok"
                for row in rows
            ]
            misses.append(
                f"| {scenario.id} | {utterance} | {scenario.expected_intent.value} | "
                + " | ".join(cells)
                + " |"
            )
        cells = []
        for row in rows:
            text = (row.get("composition_text") or "(none)").replace("|", "/")
            if not row.get("composition_safe"):
                text = f"**REJECTED ({row.get('composition_failure')})** {text}"
            cells.append(text)
        replies.append(f"| {scenario.id} | {utterance} | " + " | ".join(cells) + " |")

    report = "\n\n".join(
        [
            "# Dialogue bench comparison",
            summary,
            "## Intent misses",
            "\n".join(misses),
            "## Composed replies",
            "\n".join(replies),
        ]
    )
    (out_dir / "comparison.md").write_text(report + "\n")
    print(summary)
    print("\n" + "\n".join(misses))
    print(f"\nfull report: {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
