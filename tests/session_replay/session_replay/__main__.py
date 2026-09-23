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
    running.add_argument("scenario", type=Path)
    running.add_argument("--expect", type=Path)
    running.add_argument("--llm", choices=("live", "recorded", "none"))
    running.add_argument("--model")
    running.add_argument("--backend", choices=("ollama", "openai"))
    running.add_argument("--base-url")
    running.add_argument("--strategies")
    running.add_argument("--person")
    running.add_argument("--tz", default="America/New_York")
    running.add_argument("--tail-s", type=float, default=60)
    running.add_argument("--out", type=Path)
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
        spec = load_expect(args.expect) if args.expect else None
        mode = args.llm or (spec or {}).get("llm", "none")
        timeline = run_scenario(
            args.scenario,
            expect=spec,
            llm_mode=mode,
            model=args.model,
            backend=args.backend,
            base_url=args.base_url,
            strategies=args.strategies,
            person=args.person,
            tz=args.tz,
            tail_s=args.tail_s,
        )
        for row in timeline:
            print(format_row(row))
        if args.out:
            with args.out.open("w") as handle:
                for row in timeline:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if spec is not None:
            passed, reports = check_expectations(timeline, spec)
            print("\nExpectations:")
            if not passed and spec.get("known_bug"):
                print(f"Known bug: {spec['known_bug']}")
            print("\n".join(reports))
            return 0 if passed else 1
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
