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

    zones = commands.add_parser("zones", help="print a bridge frame path for drawing zones")
    zones.add_argument("--clip", required=True)
    zones.add_argument("--variant", choices=("squash", "letterbox"), default="squash")

    blur = commands.add_parser("blur", help="fail-closed head blur of review frames")
    blur.add_argument("--clip", required=True)
    blur.add_argument("--force", action="store_true")

    sheets = commands.add_parser("sheets", help="build privacy-safe 3x3 contact sheets")
    sheets.add_argument("--clip", required=True)
    sheets.add_argument("--force", action="store_true")
    sheets.add_argument(
        "--confirm-reviewed",
        action="store_true",
        help="record that a human spot-checked the generated sheets",
    )

    local = commands.add_parser("label-local", help="label review frames with Ollama")
    local.add_argument("--clip", required=True)
    local.add_argument("--model", default="qwen3-vl:8b")
    local.add_argument("--fast", action="store_true", help="use gemma4:e4b-mlx")
    local.add_argument("--ollama-url", default="http://localhost:11434")
    adaptive = local.add_mutually_exclusive_group()
    adaptive.add_argument("--adaptive", action="store_true", dest="adaptive", default=True)
    adaptive.add_argument("--all-frames", action="store_false", dest="adaptive")
    local.add_argument("--motion-threshold", type=float, default=0.02)
    local.add_argument("--force", action="store_true")

    codex = commands.add_parser("label-codex", help="label reviewed contact sheets with Codex")
    codex.add_argument("--clip", required=True)
    codex.add_argument("--model")
    codex.add_argument("--force", action="store_true")
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
    elif args.command == "predict":
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
    elif args.command == "zones":
        from video_eval.zones import zone_reference_frame

        print(zone_reference_frame(args.clip, root=args.data_root, variant=args.variant))
        return 0
    elif args.command == "blur":
        from video_eval.blur import blur_clip

        result = blur_clip(args.clip, root=args.data_root, force=args.force)
    elif args.command == "sheets":
        from video_eval.blur import build_sheets

        result = build_sheets(
            args.clip,
            root=args.data_root,
            force=args.force,
            confirm_reviewed=args.confirm_reviewed,
        )
    elif args.command == "label-local":
        from video_eval.label_local import FAST_MODEL, label_local

        result = label_local(
            args.clip,
            root=args.data_root,
            model=FAST_MODEL if args.fast else args.model,
            ollama_url=args.ollama_url,
            adaptive=args.adaptive,
            motion_threshold=args.motion_threshold,
            force=args.force,
        )
    else:
        from video_eval.label_codex import label_codex

        result = label_codex(args.clip, root=args.data_root, model=args.model, force=args.force)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
