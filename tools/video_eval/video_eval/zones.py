"""Locate a prepared bridge frame for hand-authored camera zones."""

from __future__ import annotations

from pathlib import Path

from video_eval.common import read_jsonl
from video_eval.paths import EvalPaths


def zone_reference_frame(
    clip_id: str,
    *,
    root: Path | None = None,
    variant: str = "squash",
) -> Path:
    """Return a prepared bridge frame that can be used to draw ``zones.yaml``."""
    paths = EvalPaths.for_clip(clip_id, root)
    if not paths.frames.is_file():
        raise RuntimeError(f"no frame manifest for {clip_id}; run prepare first")

    path_key = "bridge_path" if variant == "squash" else "bridge_letterbox_path"
    for frame in read_jsonl(paths.frames):
        relative_path = frame.get(path_key)
        if relative_path:
            reference = paths.root / relative_path
            if reference.is_file():
                return reference.resolve()

    raise RuntimeError(
        f"no prepared {variant} bridge frames for {clip_id}; "
        f"run prepare --variant {variant} first"
    )
