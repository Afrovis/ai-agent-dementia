"""Persist text-only live evidence and score a scene."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import redis
from nc_shared.replay import export_history

from .bugs import BugEntry, RunDir, harness_error, results_to_entries
from .invariants import InvariantResult, check_trace
from .trace import from_export


def agent_model(repo_root: Path, env: dict | None = None) -> str:
    values = dict(env or os.environ)
    if "AGENT_LLM_MODEL" in values:
        return values["AGENT_LLM_MODEL"]
    dotenv = repo_root / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            if line.startswith("AGENT_LLM_MODEL="):
                return line.partition("=")[2].strip().strip("\"'")
    return "-"


def record_scene(
    run: RunDir,
    scene_id: str,
    source: Path,
    start: datetime,
    stack,
    *,
    preflight: dict,
    mind_lines: list[dict],
    mind_calls: list[dict] | None = None,
    playback_source: str = "virtual_page",
    capped: bool = False,
    errors: list[str] | None = None,
    export_file: Path | None = None,
) -> tuple[Path, list[BugEntry]]:
    folder = run.path / scene_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scene.yaml").write_bytes(source.read_bytes())
    calls = mind_calls or []
    (folder / "mind.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in sorted([*mind_lines, *calls], key=lambda row: row["t"])
        )
    )
    errors = list(errors or [])
    try:
        (folder / "agent.log").write_text(stack.logs("agent", start.isoformat()))
    except Exception as exc:
        errors.append(f"agent logs: {type(exc).__name__}: {exc}")
        (folder / "agent.log").write_text("")
    target = folder / "export.jsonl"
    if export_file:
        target.write_bytes(export_file.read_bytes())
    else:
        try:
            with target.open("w") as out:
                export_history(
                    redis.Redis(host="localhost", port=16379),
                    out,
                    since=start,
                    include_media=False,
                )
        except Exception as exc:
            errors.append(f"bus export: {type(exc).__name__}: {exc}")
            target.write_text("")
    lines = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
    trace = from_export(
        [{key: value for key, value in row.items() if key != "recorded_at"} for row in lines],
        id=scene_id,
    )
    trace.source = "live"
    # The nightsim stack is always night; the profile feeds wording and veto checks.
    trace.meta["in_night_window"] = True
    person = (getattr(stack, "env", None) or {}).get("NIGHTSIM_PERSON")
    if person:
        try:
            from agent.profile import load_profile

            path = Path(person)
            if not path.is_absolute():
                path = Path(stack.repo_root) / path
            trace.meta["profile"] = load_profile(path).prompt_data()
        except Exception as exc:
            errors.append(f"profile: {type(exc).__name__}: {exc}")
    results = check_trace(trace, run.thresholds)
    if capped:
        results.append(
            InvariantResult(
                id="SM-3",
                severity="major",
                passed=False,
                window=(run.thresholds.max_scene_s, run.thresholds.max_scene_s),
                t=run.thresholds.max_scene_s,
                reason="scene hit 600 s cap",
                evidence=["scene hit 600 s cap"],
            )
        )
    model = agent_model(stack.repo_root, stack.env)
    extra = {
        **preflight,
        "model": model,
        "scene": scene_id,
        "playback_source": playback_source,
        "scene_start": start.isoformat(),
        "capped": capped,
    }
    latencies = sorted(row["latency_s"] for row in calls)
    if latencies:

        def percentile(p: float) -> float:
            index = (len(latencies) - 1) * p
            lo, hi = math.floor(index), math.ceil(index)
            return round(latencies[lo] + (latencies[hi] - latencies[lo]) * (index - lo), 3)

        extra["mind_latency_p50_s"] = percentile(0.5)
        extra["mind_latency_p95_s"] = percentile(0.95)
    run.write_scene(scene_id, trace, results, extra)
    entries = results_to_entries(results, run.id, scene_id, preflight.get("commit", "-"), model)
    for message in errors:
        item = harness_error(run.id, scene_id, message)
        item.commit = preflight.get("commit", "-")
        item.model = model
        entries.append(item)
    run.append(entries)
    return folder, entries
