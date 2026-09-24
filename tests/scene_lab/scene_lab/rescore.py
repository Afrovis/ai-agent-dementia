"""Re-check a finished run's saved traces with the current invariants.

Checks improve after a run has been recorded; re-running an hour of live scenes to
see their effect is wasteful. `rescore` reads each scene's `trace.jsonl`, restores the
context the live recorder supplied (always night, the scene card's profile), and writes
a sibling run `<run>-rescored` with fresh reports, `bugs.jsonl` and `bugs.md`. Harness
entries from the original run are carried over unchanged. The original run is not touched.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from .bugs import BugEntry, RunDir, _read, results_to_entries
from .invariants import check_trace
from .thresholds import load
from .trace import Trace

REPO = Path(__file__).resolve().parents[3]


def _profile(card: dict) -> dict:
    path = card.get("profile") or "config/person.example.yaml"
    if path == "default":
        path = "config/person.example.yaml"
    try:
        from agent.profile import load_profile

        return load_profile(REPO / path).prompt_data()
    except Exception:
        return {}


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "-"


def rescore(source: Path, thresholds_path: Path | None = None) -> Path:
    source = Path(source)
    thresholds = load(thresholds_path)
    target = RunDir(
        "rescored",
        root=source.parent,
        path=source.with_name(source.name + "-rescored"),
        thresholds=thresholds,
    )
    if (target.path / "bugs.jsonl").exists():
        (target.path / "bugs.jsonl").unlink()
    commit = _commit()
    scenes = sorted(p for p in source.iterdir() if (p / "trace.jsonl").is_file())
    model = "-"
    for folder in scenes:
        trace = Trace.read_jsonl(folder / "trace.jsonl", id=folder.name, source="live")
        card = {}
        if (folder / "scene.yaml").is_file():
            card = yaml.safe_load((folder / "scene.yaml").read_text()) or {}
        report = {}
        if (folder / "report.json").is_file():
            report = json.loads((folder / "report.json").read_text())
        model = report.get("model", model)
        # Live scenes run on the always-night nightsim stack.
        trace.meta.setdefault("in_night_window", True)
        trace.meta.setdefault("profile", _profile(card))
        results = check_trace(trace, thresholds)
        target.write_scene(
            folder.name,
            trace,
            results,
            {"commit": commit, "model": model, "rescored_from": report.get("commit", "-")},
        )
        entries = results_to_entries(results, target.id, folder.name, commit, model)
        harness = [
            BugEntry(**{**row, "run": target.id})
            for row in _read(source / "bugs.jsonl")
            if row.get("origin") == "harness" and row.get("scene") == folder.name
        ]
        target.append(entries + harness)
    target.finish("rescored", commit, model, 0, len(scenes))
    return target.path


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "rescore", help="re-check a finished run's traces with the current invariants"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--thresholds", type=Path)
