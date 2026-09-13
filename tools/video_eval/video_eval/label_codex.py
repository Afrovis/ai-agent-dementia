"""Privacy-gated contact-sheet labelling through the local Codex CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from video_eval.common import matching_meta, read_jsonl, update_index, write_jsonl, write_meta
from video_eval.labels import LABEL_PROMPT, label_record, validate_label
from video_eval.paths import EvalPaths


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_from_output(output: str) -> list[Any]:
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex returned invalid JSON") from exc
    if not isinstance(raw, list):
        raise ValueError("Codex response must be a JSON array")
    return raw


def codex_prompt(frames: list[dict[str, Any]]) -> str:
    labels = ", ".join(f"frame {row['frame_index']}" for row in frames)
    return (
        f"{LABEL_PROMPT}\n\nThis is a 3 by 3 contact sheet (last row may be partial). "
        f"Its tiles are labelled {labels}, in reading order. Return a JSON array with exactly "
        f"{len(frames)} objects, one per tile in that order. Do not wrap it in markdown."
    )


def run_codex(sheet: Path, prompt: str, model: str | None) -> str:
    command = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check", "-i", str(sheet)]
    if model:
        command.extend(["--model", model])
    command.append("-")
    try:
        completed = subprocess.run(
            command, input=prompt, check=True, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError("codex CLI is required for label-codex") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Codex labelling failed: {exc.stderr.strip()}") from exc
    return completed.stdout


def label_codex(
    clip_id: str,
    *,
    root: Path | None = None,
    model: str | None = None,
    force: bool = False,
    runner: Callable[[Path, str, str | None], str] = run_codex,
) -> dict[str, object]:
    started = time.monotonic()
    paths = EvalPaths.for_clip(clip_id, root)
    reviewed = paths.clip / "sheets.reviewed.json"
    if not reviewed.exists():
        raise RuntimeError(
            "contact sheets require a human privacy spot-check; inspect them, then run "
            "`video_eval sheets --clip ... --confirm-reviewed`"
        )
    sheet_manifest = read_jsonl(paths.clip / "sheets.jsonl")
    review_marker = json.loads(reviewed.read_text(encoding="utf-8"))
    if review_marker.get("reviewed") is not True or review_marker.get("sheet_count") != len(
        sheet_manifest
    ):
        raise RuntimeError("contact-sheet human review marker is invalid or stale")
    if review_marker.get("manifest_sha256") != _digest(paths.clip / "sheets.jsonl"):
        raise RuntimeError("contact-sheet manifest changed after human review")
    output = paths.labels / "codex.jsonl"
    meta_path = paths.labels / "codex.meta.json"
    parameters = {
        "clip_id": clip_id,
        "model": model or "codex-config-default",
        "review_marker": review_marker,
    }
    if not force and output.exists() and matching_meta(meta_path, parameters):
        return {
            "status": "skipped",
            "frames": len(read_jsonl(output)),
            "model": model or "codex-config-default",
        }
    records: list[dict[str, Any]] = []
    sheets_root = paths.sheets.resolve()
    for sheet_entry in sheet_manifest:
        sheet = (paths.root / sheet_entry["sheet_path"]).resolve()
        if sheet.parent != sheets_root or not sheet.is_file():
            raise RuntimeError(f"refusing non-sheet Codex input: {sheet}")
        expected_digest = review_marker.get("sheet_sha256", {}).get(sheet_entry["sheet_path"])
        if expected_digest != _digest(sheet):
            raise RuntimeError(f"contact sheet changed after human review: {sheet.name}")
        frames = sheet_entry["frames"]
        raw_labels = _array_from_output(runner(sheet, codex_prompt(frames), model))
        if len(raw_labels) != len(frames):
            raise ValueError(
                f"Codex returned {len(raw_labels)} labels for {len(frames)} sheet tiles"
            )
        for frame, raw in zip(frames, raw_labels, strict=True):
            label = validate_label(raw)
            records.append(
                label_record(
                    label,
                    frame_index=int(frame["frame_index"]),
                    t_s=float(frame["t_s"]),
                    labeller=f"codex:{model or 'config-default'}",
                )
            )
        write_jsonl(output, records)
    write_meta(
        meta_path,
        command="label-codex",
        parameters=parameters,
        started_at=started,
        versions=("codex",),
    )
    update_index(paths.root, clip_id, "label_codex", "complete")
    return {
        "status": "complete",
        "frames": len(records),
        "model": model or "codex-config-default",
        "sheets": len(sheet_manifest),
    }
