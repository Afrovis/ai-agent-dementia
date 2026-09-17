#!/usr/bin/env python3
"""Render each `script.json` step's prompt text to a WAV (HANDOFF.md V10).

Runs inside the root `embodiment` image, which already bundles Piper and the
`en_US-lessac-medium` voice (services/embodiment/Dockerfile) -- this script
has no dependencies of its own beyond that image's Python environment.

    docker compose -f ../docker-compose.yml run --rm --no-deps \\
        -v "$PWD/scripts:/scripts:ro" \\
        -v "$PWD/web/volunteer_web/static:/static:ro" \\
        -v "$VOLUNTEER_DATA_DIR/prompts:/out" \\
        embodiment python /scripts/render_prompts.py /static/script.json /out
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

DEFAULT_VOICE_MODEL = Path("/app/models/piper/en_US-lessac-medium.onnx")


def render_prompts(
    script_path: Path, output_dir: Path, *, voice_model: Path, speed: float, force: bool
) -> list[str]:
    from piper import PiperVoice, SynthesisConfig

    steps = json.loads(script_path.read_text(encoding="utf-8"))
    voice = PiperVoice.load(voice_model)
    config = SynthesisConfig(length_scale=1.0 / speed)

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for step in steps:
        destination = output_dir / f"{step['step_id']}.wav"
        if destination.exists() and not force:
            continue
        temporary = destination.with_suffix(".wav.tmp")
        with wave.open(str(temporary), "wb") as wav_file:
            voice.synthesize_wav(step["text"], wav_file, syn_config=config)
        temporary.replace(destination)
        rendered.append(step["step_id"])
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--voice-model", type=Path, default=DEFAULT_VOICE_MODEL)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Piper speed multiplier; 1.0 is normal pace (unlike the bedside voice's 0.85x).",
    )
    parser.add_argument("--force", action="store_true", help="re-render WAVs that already exist")
    args = parser.parse_args(argv)

    rendered = render_prompts(
        args.script_path,
        args.output_dir,
        voice_model=args.voice_model,
        speed=args.speed,
        force=args.force,
    )
    print(f"rendered {len(rendered)} prompt(s) into {args.output_dir}: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
