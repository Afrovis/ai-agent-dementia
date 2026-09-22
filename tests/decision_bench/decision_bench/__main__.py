"""CLI entry point for replaying and scoring decision scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from agent.llm import local_llm

from decision_bench.annotate import AnnotatorError
from decision_bench.judge import first_occurrences, judge_sentences, sentences_from_trace
from decision_bench.report import print_json_report, print_report, print_trace, write_json_report
from decision_bench.runner import Trace, run_scenario
from decision_bench.schema import (
    CATEGORIES,
    DEFAULT_PROFILE_PATH,
    SCENARIOS_DIR,
    Scenario,
    citation_problems,
    guideline_clauses,
    load_default_profile,
    load_scenarios,
)
from decision_bench.scoring import ModelResult, score_scenario
from decision_bench.stub_llm import StubLLM
from decision_bench.verdicts import (
    VERDICTS_PATH,
    add_pending,
    conflicts,
    load_verdicts,
    record_judgements,
    unjudged,
)

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
        "--judge",
        action="store_true",
        help="check new sentences against the profile with an isolated Claude judge "
        "(Claude Code on the claude.ai subscription, not the API)",
    )
    parser.add_argument("--judge-model", default="opus")
    parser.add_argument("--judge-effort", default="medium")
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


def _judge(runs, args: argparse.Namespace) -> None:
    sentences = [
        sentence
        for model_runs in runs.values()
        for _, trace, _, profile in model_runs
        for sentence in sentences_from_trace(trace, profile)
    ]
    pending = first_occurrences(sentences, unjudged((s.text for s in sentences), args.verdicts))
    if not pending:
        print("judge: every sentence already checked", file=sys.stderr)
        return
    try:
        judged, used_model = judge_sentences(
            pending, model=args.judge_model, effort=args.judge_effort
        )
    except AnnotatorError as exc:
        raise SystemExit(str(exc)) from exc
    recorded = record_judgements(
        ((s.text, verdict) for s, verdict in zip(pending, judged, strict=True)),
        used_model,
        args.verdicts,
    )
    print(f"judge ({used_model}): checked {recorded} sentence(s)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "annotate":
        from decision_bench.annotate import main as annotate_main

        return annotate_main(argv[1:])
    if argv and argv[0] == "calibrate":
        from decision_bench.calibrate import main as calibrate_main

        return calibrate_main(argv[1:])
    if argv and argv[0] == "noise":
        from decision_bench.noise import main as noise_main

        return noise_main(argv[1:])
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
    models = args.models or (["stub"] if args.backend == "stub" else list(DEFAULT_MODELS))
    runs: dict[str, list[tuple[Scenario, Trace, float, dict[str, object]]]] = {}
    wall_times: dict[str, float] = {}
    traces = []

    for model in models:
        started = perf_counter()
        runs[model] = []
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
            runs[model].append((scenario, trace, elapsed, profile))
            traces.append((model, trace))
        wall_times[model] = perf_counter() - started

    if args.collect_verdicts or args.judge:
        texts = [
            str(entry.data.get("text", ""))
            for _, trace in traces
            for entry in trace.entries
            if entry.kind == "Say"
        ]
        added = add_pending(texts, args.verdicts)
        print(f"{added} new sentence(s) added to {args.verdicts}", file=sys.stderr)
    if args.judge:
        _judge(runs, args)

    verdicts = load_verdicts(args.verdicts)
    results = [
        ModelResult(
            model=model,
            scenarios=tuple(
                score_scenario(
                    scenario, trace, profile=profile, wall_time_seconds=elapsed, verdicts=verdicts
                )
                for scenario, trace, elapsed, profile in model_runs
            ),
            wall_time_seconds=wall_times[model],
        )
        for model, model_runs in runs.items()
    ]

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
    disputed = conflicts(args.verdicts)
    if disputed:
        destination = sys.stderr if args.json else None
        print(f"\n{len(disputed)} human verdict(s) the judge disagrees with:", file=destination)
        for text, pattern, human, judged, evidence in disputed:
            print(f"  {pattern}: human {human}, judge {judged}: {text!r}", file=destination)
            print(f"    judge's evidence: {evidence}", file=destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
