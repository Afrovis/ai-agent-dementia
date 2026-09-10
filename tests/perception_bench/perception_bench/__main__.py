"""CLI entry point: `python -m perception_bench` (see `README.md` for how to
get it on `sys.path`/installed).

Runs all three tiers -- tier 1 always, tier 2 and 3 opt-in and self-skipping
when their data is absent -- prints a report, and exits non-zero only if a
*measured* target was missed (this issue's hard requirement: missing data
is not a failing test).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from perception_bench.daylight import DEFAULT_DATA_DIR, Tier2Result, run_tier2
from perception_bench.infrared import DEFAULT_MANIFEST_PATH, run_tier3
from perception_bench.report import build_report, print_json_report, print_report
from perception_bench.scoring import LatencyResult, score_predictions
from perception_bench.synthetic import run_tier1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perception_bench",
        description=(
            "Night Companion perception bench (issue #11): measures perceive's "
            "pose pipeline on synthetic, daylight, and infrared fixtures."
        ),
    )
    parser.add_argument("--json", action="store_true", help="print a machine-readable JSON report")
    parser.add_argument(
        "--daylight-data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory fetch_daylight.sh populated (default: %(default)s)",
    )
    parser.add_argument(
        "--daylight-pose-backend",
        default="mediapipe",
        choices=("mediapipe", "yolo"),
        help="pose backend to run tier 2 with, if installed (default: %(default)s)",
    )
    parser.add_argument(
        "--night-degrade",
        action="store_true",
        help="apply the night-degradation transform to tier 2 frames before scoring",
    )
    parser.add_argument(
        "--ir-manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="tier 3 manifest path (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-daylight",
        action="store_true",
        help="skip tier 2 entirely without even checking for data (useful in CI)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    tier1_runs = run_tier1()
    tier1_pairs = [pair for run in tier1_runs for pair in run.pairs]
    tier1_accuracy = score_predictions(tier1_pairs)
    tier1_latency = LatencyResult(
        samples=[sample for run in tier1_runs for sample in run.latency.samples]
    )

    if args.skip_daylight:
        tier2 = Tier2Result(
            accuracy=None,
            skipped_reason="--skip-daylight given",
            degraded=args.night_degrade,
            clip_count=0,
            pose_backend_name=args.daylight_pose_backend,
        )
    else:
        tier2 = run_tier2(
            args.daylight_data_dir,
            pose_backend_name=args.daylight_pose_backend,
            degrade=args.night_degrade,
        )

    tier3 = run_tier3(args.ir_manifest)

    report = build_report(tier1_runs, tier1_accuracy, tier1_latency, tier2, tier3)

    if args.json:
        print_json_report(report)
    else:
        print_report(report)

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
