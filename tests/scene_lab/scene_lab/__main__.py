"""Scene lab command line tools."""

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from .bugs import RunDir, _default_root, merge, render_merge_md, results_to_entries
from .invariants import check_trace
from .thresholds import load
from .trace import Trace, from_agent_log, from_export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scene_lab")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("file", type=Path)
    check.add_argument("--format", choices=("export", "trace", "agent_log"), default="export")
    check.add_argument("--thresholds", type=Path)
    check.add_argument("--run-kind", default="check")
    bugs = sub.add_parser("bugs")
    bugs.add_argument("runs", nargs="+")
    live = sub.add_parser("run")
    live.add_argument("scene", type=Path, nargs="?")
    live.add_argument("--hours", type=float)
    live.add_argument("--max-scenes", type=int)
    live.add_argument("--no-stack-up", action="store_true")
    live.add_argument("--keep-stack", action="store_true")
    live.add_argument("--runs-root", type=Path)
    live.add_argument("--run-dir", type=Path)
    frombench = sub.add_parser("from-bench")
    frombench.add_argument("scenario")
    frombench.add_argument("--out", type=Path, default=Path("tests/scene_lab/scenes"))
    difference = sub.add_parser("diff")
    difference.add_argument("live", type=Path)
    difference.add_argument("inprocess", type=Path)
    difference.add_argument("--tolerance-s", type=float, default=3)
    offline = sub.add_parser("inprocess")
    offline.add_argument("scene", type=Path)
    offline.add_argument("--out", type=Path)
    try:
        from .promote import register

        register(sub)
    except ImportError:
        sub.add_parser("promote", help="reserved for phase 5")
    args = parser.parse_args(argv)
    if args.command == "bugs":
        root = _default_root()
        paths = [Path(p) if Path(p).exists() else root / p for p in args.runs]
        print(render_merge_md(merge(paths)), end="")
        return 0
    if args.command == "run":
        from .run import run_hours, run_live

        if (args.scene is None) == (args.hours is None):
            parser.error("run needs either a scene card or --hours N")
        if args.hours is not None:
            if args.no_stack_up or args.run_dir:
                parser.error("--no-stack-up and --run-dir require a scene card")
            print(
                asyncio.run(
                    run_hours(
                        args.hours,
                        runs_root=args.runs_root,
                        keep_stack=args.keep_stack,
                        max_scenes=args.max_scenes,
                    )
                )
            )
            return 0

        print(
            asyncio.run(
                run_live(
                    args.scene,
                    no_stack_up=args.no_stack_up,
                    keep_stack=args.keep_stack,
                    runs_root=args.runs_root,
                    run_dir=args.run_dir,
                )
            )
        )
        return 0
    if args.command == "from-bench":
        from .frombench import convert

        print(convert(args.scenario, args.out))
        return 0
    if args.command == "diff":
        from .diff import compare, render

        print(
            render(
                compare(
                    Trace.read_jsonl(args.live), Trace.read_jsonl(args.inprocess), args.tolerance_s
                )
            )
        )
        return 0
    if args.command == "inprocess":
        from .inprocess import run_card

        print(run_card(args.scene, args.out))
        return 0
    if args.command == "promote":
        if not hasattr(args, "run_dir"):
            parser.error("promote is planned for phase 5")
        from .promote import promote

        promote(
            args.run_dir,
            scene=args.scene,
            at=args.at,
            to=args.to,
            before=args.before,
            after=args.after,
            out_dir=args.out_dir,
            name=args.name,
            claude=args.claude,
        )
        return 0
    thresholds = load(args.thresholds)
    if args.format == "trace":
        trace = Trace.read_jsonl(args.file)
    else:
        lines = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
        trace = (
            from_export(lines, id=args.file.stem)
            if args.format == "export"
            else from_agent_log(lines, id=args.file.stem)
        )
    results = check_trace(trace, thresholds)
    run = RunDir(args.run_kind, thresholds=thresholds)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "-"
    run.write_scene(trace.id, trace, results, {"commit": commit, "model": "-"})
    entries = results_to_entries(results, run.id, trace.id, commit, "-")
    run.append(entries)
    run.finish(args.run_kind, commit, "-", 0, 1)
    print(f"{run.path}: {len(entries)} issues, {len(results)} results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
