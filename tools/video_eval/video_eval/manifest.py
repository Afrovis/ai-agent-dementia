"""Maintains `<data root>/manifest.yaml`, the tier-3 manifest
`perception_bench.infrared` reads (see `docs/VIDEO_EVAL.md` step A8).

Kept dependency-free of `perception_bench` on purpose -- `video_eval.reconcile`
needs to update this file at confirm time, and `perception_bench` is the one
side of that relationship allowed to depend on `video_eval`, not the other
way around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MANIFEST_FILENAME = "manifest.yaml"


def manifest_path(root: Path) -> Path:
    return root / MANIFEST_FILENAME


def load_raw_manifest(root: Path) -> dict[str, Any]:
    """Load `<root>/manifest.yaml`, or an empty `{"clips": []}` if absent.

    Raises `ValueError` if the file exists but is not a mapping, matching
    `perception_bench.infrared.load_manifest`'s "fail loud on malformed"
    rule.
    """
    path = manifest_path(root)
    if not path.exists():
        return {"clips": []}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping with a 'clips' key")
    clips = raw.get("clips")
    if clips is None:
        raw["clips"] = []
    elif not isinstance(clips, list):
        raise ValueError(f"{path}: 'clips' must be a list")
    return raw


def upsert_clip(
    root: Path,
    clip_id: str,
    *,
    confirmed_by: str,
    confirmed_at: str,
) -> Path:
    """Add or update `clip_id`'s entry in `<root>/manifest.yaml`.

    Idempotent: re-confirming the same clip updates its entry in place
    rather than appending a duplicate. Preserves every other entry and
    their order. Creates the file if it does not exist yet. Writes
    atomically (temp file + `Path.replace`).
    """
    raw = load_raw_manifest(root)
    clips: list[dict[str, Any]] = raw["clips"]
    entry = {"clip_id": clip_id, "confirmed_by": confirmed_by, "confirmed_at": confirmed_at}
    for index, existing in enumerate(clips):
        if isinstance(existing, dict) and existing.get("clip_id") == clip_id:
            clips[index] = entry
            break
    else:
        clips.append(entry)
    raw["clips"] = clips

    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    temporary.replace(path)
    return path
