"""Command-line interface for classifier_bench."""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

from decision_bench.annotate import ANNOTATIONS_DIR
from decision_bench.schema import GUIDELINES_PATH, SCENARIOS_DIR

from classifier_bench.ask import ask_questions, read_jsonl, run_ollama, run_stub
from classifier_bench.build import build_questions, build_summary, write_questions
from classifier_bench.score import format_report, score_records, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe one decision action at a time.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="derive questions from paired annotations")
    build.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR / "model")
    build.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR)
    build.add_argument("--out", type=Path, required=True)

    ask = subparsers.add_parser("ask", help="ask a candidate about each action")
    ask.add_argument("questions", type=Path)
    ask.add_argument("--out", type=Path, required=True)
    ask.add_argument("--backend", choices=("ollama", "stub"), default="ollama")
    ask.add_argument("--model", default="gemma4:e4b-mlx")
    ask.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ask.add_argument("--guidelines", type=Path, default=GUIDELINES_PATH)
    ask.add_argument("--limit", type=int)
    ask.add_argument("--set", choices=("agree", "disputed", "all"), default="all")
    ask.add_argument("--force", action="store_true")

    score = subparsers.add_parser("score", help="score candidate answers")
    score.add_argument("answers", type=Path)
    score.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        questions = build_questions(args.annotations, args.scenarios)
        write_questions(args.out, questions)
        print(build_summary(questions))
        print(f"wrote {args.out}")
        return 0
    if args.command == "ask":
        if args.limit is not None and args.limit < 0:
            raise SystemExit("--limit must be non-negative")
        questions = read_jsonl(args.questions)
        runner = (
            functools.partial(run_ollama, base_url=args.ollama_url)
            if args.backend == "ollama"
            else run_stub
        )
        written = ask_questions(
            questions,
            args.out,
            guidelines_text=args.guidelines.read_text(encoding="utf-8"),
            runner=runner,
            model=args.model,
            selected_set=args.set,
            limit=args.limit,
            force=args.force,
        )
        failures = sum("error" in item for item in written)
        print(f"wrote {len(written)} answer(s) to {args.out}; failures={failures}")
        return 0
    if args.command == "score":
        report = score_records(read_jsonl(args.answers))
        print(format_report(report))
        if args.out:
            write_report(args.out, report)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
