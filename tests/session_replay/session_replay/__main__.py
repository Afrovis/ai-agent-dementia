"""CLI for extracting and replaying bedside sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from session_replay.core import (
    check_expectations,
    extract,
    format_row,
    load_expect,
    parse_ts,
    run_scenario,
)


def main(argv: list[str] | None = None) -> int:
    """Run the requested extract or simulation."""
    parser = argparse.ArgumentParser(prog="python -m session_replay")
    commands = parser.add_subparsers(dest="command", required=True)
    extracting = commands.add_parser("extract", help="keep text and person-state inputs")
    extracting.add_argument("source", type=Path)
    extracting.add_argument("target", type=Path)
    extracting.add_argument("--since")
    extracting.add_argument("--until")
    running = commands.add_parser("run", help="simulate the agent against a scenario")
    running.add_argument("scenario", type=Path, nargs="+")
    running.add_argument("--expect", type=Path)
    running.add_argument("--llm", choices=("live", "recorded", "none"))
    running.add_argument("--llm-latency", default="none", help="none, recorded, or fixed:<seconds>")
    running.add_argument("--llm-latency-fallback", type=float, default=2.5)
    running.add_argument("--model")
    running.add_argument("--backend", choices=("ollama", "openai"))
    running.add_argument("--base-url")
    running.add_argument("--strategies")
    running.add_argument("--person")
    running.add_argument("--tz", default="America/New_York")
    running.add_argument("--tail-s", type=float, default=60)
    running.add_argument("--out", type=Path)
    running.add_argument("--invariants", action="store_true", help="write scene_lab reports")
    running.add_argument("--thresholds", type=Path)
    running.add_argument("--runs-root", type=Path)
    running.add_argument("--tt2-judge", action="store_true", help="rate questions with Claude")
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            count = extract(
                args.source,
                args.target,
                since=parse_ts(args.since) if args.since else None,
                until=parse_ts(args.until) if args.until else None,
            )
            print(f"extracted {count} events")
            return 0
        if len(args.scenario) > 1 and not args.invariants:
            parser.error("multiple scenarios require --invariants")
        if len(args.scenario) > 1 and args.expect:
            parser.error("--expect requires one scenario")
        if len(args.scenario) > 1 and args.out:
            parser.error("--out requires one scenario")
        if not args.invariants and (args.thresholds or args.runs_root or args.tt2_judge):
            parser.error("--thresholds, --runs-root and --tt2-judge require --invariants")
        invariant_run = None
        if args.invariants:
            try:
                from scene_lab.offline import InvariantRun
                from scene_lab.trace import from_session_replay

                if args.tt2_judge:
                    from scene_lab.judge import claude_judge
            except ImportError as exc:
                parser.error(f"--invariants requires scene_lab; install tests/scene_lab ({exc})")
            invariant_run = InvariantRun(
                "session_replay",
                args.model or args.llm or "recorded/none",
                args.thresholds,
                args.runs_root,
                claude_judge() if args.tt2_judge else None,
            )
        passed_all = True
        for scenario in args.scenario:
            expect_path = args.expect
            if args.invariants and expect_path is None:
                sibling = scenario.with_suffix(".expect.yaml")
                if sibling.exists():
                    expect_path = sibling
            spec = load_expect(expect_path) if expect_path else None
            mode = args.llm or (spec or {}).get("llm", "none")
            timeline = run_scenario(
                scenario,
                expect=spec,
                llm_mode=mode,
                model=args.model,
                backend=args.backend,
                base_url=args.base_url,
                strategies=args.strategies,
                person=args.person,
                tz=args.tz,
                tail_s=args.tail_s,
                llm_latency=args.llm_latency,
                llm_latency_fallback=args.llm_latency_fallback,
            )
            if len(args.scenario) > 1:
                print(f"\nSCENARIO {scenario.stem}")
            for row in timeline:
                print(format_row(row))
            if args.out:
                with args.out.open("w") as handle:
                    for row in timeline:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if invariant_run is not None:
                from agent.config import AgentConfig
                from agent.profile import load_profile

                normalized = from_session_replay(timeline, scenario.stem)
                normalized.meta["profile"] = load_profile(
                    args.person or AgentConfig.from_env().person_path
                ).prompt_data()
                normalized.meta["llm_latency"] = args.llm_latency
                # run_scenario forces an always-on night window so any capture can start a
                # session; the veto context the agent saw was therefore always "night".
                normalized.meta["in_night_window"] = True
                invariant_run.add(scenario.stem, normalized)
            if spec is not None:
                passed, reports = check_expectations(timeline, spec)
                print("\nExpectations:")
                if not passed and spec.get("known_bug"):
                    print(f"Known bug: {spec['known_bug']}")
                print("\n".join(reports))
                passed_all &= passed
        if invariant_run is not None:
            _, summary_text = invariant_run.close()
            print(summary_text)
        return 0 if passed_all else 1
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
