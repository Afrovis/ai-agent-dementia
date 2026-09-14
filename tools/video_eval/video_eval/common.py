"""Small provenance, JSONL, and index helpers for evaluation commands."""

from __future__ import annotations

import json
import os
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


def git_sha() -> str:
    """Return the current full git SHA, or an explicit fallback outside a checkout."""
    configured = os.environ.get("VIDEO_EVAL_GIT_SHA")
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_meta(
    path: Path,
    *,
    command: str,
    parameters: dict[str, Any],
    started_at: float,
    versions: tuple[str, ...] = ("pillow", "pyyaml"),
) -> dict[str, Any]:
    meta = {
        "command": command,
        "git_sha": git_sha(),
        "package_versions": package_versions(versions),
        "parameters": parameters,
        "wall_seconds": round(time.monotonic() - started_at, 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def matching_meta(path: Path, parameters: dict[str, Any]) -> bool:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return stored.get("parameters") == parameters


def update_index(root: Path, clip_id: str, stage: str, value: str) -> None:
    """Atomically update one clip stage in the private data-root index."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "index.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        raw = {}
    clips = raw.setdefault("clips", {})
    clip = clips.setdefault(clip_id, {})
    clip[stage] = value
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
