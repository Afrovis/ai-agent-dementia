"""CLI entry point for ``python -m video_eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from video_eval.prepare import prepare_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="video_eval")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="private evaluation root (default: VIDEO_EVAL_DATA or ../data-ai-agent-dementia)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="extract review and bridge frames")
    prepare.add_argument("--video", required=True, type=Path)
    prepare.add_argument("--clip", required=True)
    prepare.add_argument("--variant", choices=("squash", "letterbox"), default="squash")
    prepare.add_argument("--force", action="store_true")

    predict = commands.add_parser("predict", help="run capture/perception offline")
    predict.add_argument("--clip", required=True)
    predict.add_argument("--backend", choices=("mediapipe", "yolo"), default="mediapipe")
    predict.add_argument("--variant", choices=("squash", "letterbox"), default="squash")
    predict.add_argument("--no-gate", action="store_true")
    predict.add_argument("--force", action="store_true")
    predict.add_argument("--confirm-frames", type=int)
    predict.add_argument("--min-confidence", type=float)
    predict.add_argument("--walk-threshold", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_video(
            args.video,
            args.clip,
            root=args.data_root,
            variant=args.variant,
            force=args.force,
        )
    else:
        # Keep `prepare` usable in a lightweight host environment with only
        # Pillow/PyYAML/ffmpeg. The real capture/perceive packages are needed
        # only by `predict`, so importing them eagerly would be needless coupling.
        from video_eval.predict import predict_clip

        result = predict_clip(
            args.clip,
            root=args.data_root,
            backend_name=args.backend,
            variant=args.variant,
            no_gate=args.no_gate,
            force=args.force,
            confirm_frames=args.confirm_frames,
            min_confidence=args.min_confidence,
            walk_threshold=args.walk_threshold,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
