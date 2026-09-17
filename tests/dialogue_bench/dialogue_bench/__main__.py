"""CLI entry point for running the dialogue suite against local Ollama models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.llm import LOCAL_BACKENDS, local_llm

from dialogue_bench.report import print_json_report, print_report
from dialogue_bench.scenarios import DEFAULT_SCENARIOS_PATH, load_scenarios
from dialogue_bench.scoring import run_model

DEFAULT_MODELS = ("llama3.1:8b", "qwen2.5:7b", "mistral:7b")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local text models on 50 Night Companion dialogue scenarios."
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="model to test; repeat to compare models (defaults to three documented models)",
    )
    parser.add_argument(
        "--backend",
        choices=LOCAL_BACKENDS,
        default="ollama",
        help="ollama, or openai for an OpenAI-compatible server such as mlx_lm.server",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--base-url", default="http://localhost:8080", help="server URL for --backend openai"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="timeout per model call")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="include generated text in JSON output (off by default for safer sharing)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    scenarios = load_scenarios(args.scenarios)
    models = args.models or list(DEFAULT_MODELS)
    results = [
        run_model(
            model,
            local_llm(
                args.backend,
                url=args.ollama_url if args.backend == "ollama" else args.base_url,
                model=model,
                timeout_seconds=args.timeout,
            ),
            scenarios,
        )
        for model in models
    ]
    if args.json:
        print_json_report(results, include_text=args.include_text)
    else:
        print_report(results)
    # This is a comparison/evaluation command, not a fixed quality gate. It
    # exits successfully after completing; measured failures remain explicit.
    return 0


if __name__ == "__main__":
    sys.exit(main())
