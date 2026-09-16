"""Ask the local MLX VLM a free-form question about one recorded frame.

`video_eval label-local` always runs the fixed posture-labelling prompt over
a whole clip. This is the ad-hoc counterpart for debugging a single frame:
point it at one `frame_index` in one clip, optionally draw a detection box
on the image first, and ask any question. Everything stays local -- the
frame never leaves the machine -- which is what makes this usable on raw,
unblurred bridge frames that must not go to an external labeller.

Runs the model natively through `mlx-vlm` on Apple Silicon rather than
through Ollama's HTTP API -- same model family, ~6x faster per call, no
request timeouts (see label_local.py's MlxLabeller on the
worktree-mlx-vlm-backend branch, which this borrows the load/generate
pattern from). Install with `pip install 'tools/video_eval[mlx]'`.

Usage:
    python tools/video_eval/scripts/ask_frame.py \
        --data-root ../data-ai-agent-dementia --clip <clip-id> \
        --frame 101 --box 0.02932 0.58586 0.54988 0.8657 \
        --question "Is the region in the red box the bed, or a separate object? One sentence."
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "video_eval"))

DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"
"""Same model family as the Ollama qwen3-vl:8b default, 8-bit native MLX."""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def draw_box(jpeg: bytes, box: tuple[float, float, float, float]) -> bytes:
    import io

    with Image.open(io.BytesIO(jpeg)) as image:
        image = image.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = box
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [x1 * width, y1 * height, x2 * width, y2 * height], outline=(255, 0, 0), width=3
        )
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def ask(model_id: str, jpeg: bytes, question: str) -> str:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(model_id)
    config = load_config(model_id)
    prompt = apply_chat_template(processor, config, question, num_images=1)

    with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
        handle.write(jpeg)
        handle.flush()
        result = generate(
            model, processor, prompt, [handle.name], max_tokens=300, temperature=0.0, verbose=False
        )
    return result.text if hasattr(result, "text") else str(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frame", type=int, required=True, help="frame_index")
    parser.add_argument(
        "--variant-key",
        default="review_path",
        help="key into frames.jsonl for the image path",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="normalized box to draw in red before asking",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    root = args.data_root.resolve()
    clip = root / "clips" / args.clip
    frames = read_jsonl(clip / "frames.jsonl")
    match = next((f for f in frames if int(f["frame_index"]) == args.frame), None)
    if match is None:
        raise SystemExit(f"frame_index {args.frame} not found in {args.clip}")

    jpeg = (root / match[args.variant_key]).read_bytes()
    if args.box:
        jpeg = draw_box(jpeg, tuple(args.box))

    answer = ask(args.model, jpeg, args.question)
    print(f"[{args.clip} frame={args.frame} t_s={match['t_s']}]")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
