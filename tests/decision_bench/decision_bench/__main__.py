"""CLI entry point for replaying and scoring decision scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from agent.llm import local_llm

from decision_bench.report import print_json_report, print_report, print_trace, write_json_report
from decision_bench.runner import run_scenario
from decision_bench.schema import (
    CATEGORIES,
    DEFAULT_PROFILE_PATH,
    SCENARIOS_DIR,
    citation_problems,
    guideline_clauses,
    load_default_profile,
    load_scenarios,
)
from decision_bench.scoring import ModelResult, score_scenario
from decision_bench.stub_llm import StubLLM
from decision_bench.verdicts import VERDICTS_PATH, add_pending, load_verdicts

DEFAULT_MODELS = ("llama3.1:8b", "qwen2.5:7b", "mistral:7b")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay scripted nights through the real Night Companion agent."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="model to test; repeat to compare models",
    )
    parser.add_argument("--backend", choices=("ollama", "openai", "stub"), default="ollama")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--timeout", type=float, default=10.0, help="timeout per model call")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--strategies", type=Path)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--scenario", action="append", dest="scenario_ids")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", type=Path, help="also write the JSON report to this path")
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=VERDICTS_PATH,
        help="human verdicts on sentences for the review-only wording patterns",
    )
    parser.add_argument(
        "--collect-verdicts",
        action="store_true",
        help="append sentences still needing a human verdict to --verdicts",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the untimed model call that loads the model before the first scenario",
    )
    return parser


def _client(backend: str, model: str, args: argparse.Namespace):
    if backend == "stub":
        return StubLLM()
    url = args.ollama_url if backend == "ollama" else args.base_url
    return local_llm(backend, url=url, model=model, timeout_seconds=args.timeout)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "annotate":
        from decision_bench.annotate import main as annotate_main

        return annotate_main(argv[1:])
    if argv and argv[0] == "apply":
        from decision_bench.review import main as review_main

        return review_main(argv[1:])
    args = build_arg_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    scenarios = load_scenarios(args.scenarios)
    if args.category:
        scenarios = [item for item in scenarios if item.category == args.category]
    if args.scenario_ids:
        wanted = set(args.scenario_ids)
        known = {item.id for item in scenarios}
        missing = sorted(wanted - known)
        if missing:
            raise SystemExit(f"unknown or filtered scenario(s): {', '.join(missing)}")
        scenarios = [item for item in scenarios if item.id in wanted]
    warnings = citation_problems(scenarios, guideline_clauses())
    verdicts = load_verdicts(args.verdicts)
    models = args.models or (["stub"] if args.backend == "stub" else list(DEFAULT_MODELS))
    results: list[ModelResult] = []
    traces = []

    for model in models:
        started = perf_counter()
        scenario_results = []
        client = _client(args.backend, model, args)
        if args.backend != "stub" and not args.no_warmup:
            # A cold model can exceed --timeout on its first calls, which would
            # read as the first scenario's LLM failures rather than the model's.
            client.interpret("Hello.", (), load_default_profile(args.profile))
        for scenario in scenarios:
            scenario_started = perf_counter()
            trace = run_scenario(
                scenario,
                llm=client,
                profile_path=args.profile,
                strategies_path=args.strategies,
            )
            elapsed = perf_counter() - scenario_started
            profile = load_default_profile(args.profile) | scenario.profile
            scenario_results.append(
                score_scenario(
                    scenario,
                    trace,
                    profile=profile,
                    wall_time_seconds=elapsed,
                    verdicts=verdicts,
                )
            )
            traces.append((model, trace))
        results.append(
            ModelResult(
                model=model,
                scenarios=tuple(scenario_results),
                wall_time_seconds=perf_counter() - started,
            )
        )

    if args.json:
        print_json_report(results, warnings)
    else:
        print_report(results, warnings)
    if args.trace:
        destination = sys.stderr if args.json else None
        for model, trace in traces:
            if len(models) > 1:
                print(f"\nMODEL {model}", file=destination)
            print_trace(trace, file=destination)
    if args.out:
        write_json_report(args.out, results, warnings)
    if args.collect_verdicts:
        pending = [
            text
            for result in results
            for scenario_result in result.scenarios
            for checkpoint in scenario_result.checkpoints
            if any("requires review" in reason for reason in checkpoint.reasons)
            for text in checkpoint.observations.get("says", ())
        ]
        added = add_pending(pending, args.verdicts)
        print(f"{added} new sentence(s) need a verdict in {args.verdicts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
