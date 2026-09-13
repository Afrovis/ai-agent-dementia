"""Filesystem layout shared by all video-evaluation commands."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalPaths:
    """Resolved paths for one private clip under ``VIDEO_EVAL_DATA``."""

    root: Path
    clip_id: str

    @classmethod
    def for_clip(cls, clip_id: str, root: Path | None = None) -> EvalPaths:
        if not clip_id or clip_id in {".", ".."} or Path(clip_id).name != clip_id:
            raise ValueError("clip id must be one non-empty path component")
        configured = root or Path(os.environ.get("VIDEO_EVAL_DATA", "../data-ai-agent-dementia"))
        return cls(root=configured.resolve(), clip_id=clip_id)

    @property
    def clip(self) -> Path:
        return self.root / "clips" / self.clip_id

    @property
    def review(self) -> Path:
        return self.clip / "review"

    def bridge(self, variant: str) -> Path:
        if variant == "squash":
            return self.clip / "bridge"
        if variant == "letterbox":
            return self.clip / "bridge-letterbox"
        raise ValueError(f"unknown bridge variant: {variant}")

    @property
    def frames(self) -> Path:
        return self.clip / "frames.jsonl"

    @property
    def predictions(self) -> Path:
        return self.clip / "predictions"

    @property
    def blurred(self) -> Path:
        return self.clip / "blurred"

    @property
    def sheets(self) -> Path:
        return self.clip / "sheets"

    @property
    def labels(self) -> Path:
        return self.clip / "labels"

    @property
    def reports(self) -> Path:
        return self.clip / "reports"

    @property
    def e2e(self) -> Path:
        return self.clip / "e2e"
