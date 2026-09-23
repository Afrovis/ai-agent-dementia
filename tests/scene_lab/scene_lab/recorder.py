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
from .invariants import InvariantResult, _state, check_trace
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


class StreamTap:
    """Follow every non-media stream during a scene with XRANGE from the last seen id.

    Capped streams (activity keeps about 200 entries) lose early playback reports and
    decision records in a busy scene, so an end-of-scene export is not enough. No consumer
    groups are created and nothing is acked, so services are unaffected. Media streams are
    never read: synthesised audio must not reach disk (HANDOFF rule 2).
    """

    def __init__(self, client, streams=None, poll_s: float = 0.5, now_fn=None) -> None:
        from nc_shared.replay import ALL_STREAMS, MEDIA_STREAMS

        self.client = client
        self.streams = [s for s in (streams or ALL_STREAMS) if s not in MEDIA_STREAMS]
        self.poll_s = poll_s
        self.now_fn = now_fn or __import__("time").time
        self.last: dict[str, str] = {}
        self.rows: list[tuple[str, dict]] = []
        self._stop = None
        self._thread = None

    def poll(self) -> int:
        from nc_shared.replay import _event_from_fields

        added = 0
        for stream in self.streams:
            low = f"({self.last[stream]}" if stream in self.last else "-"
            for msg_id, fields in self.client.xrange(stream, min=low, max="+") or []:
                msg_id = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                self.last[stream] = msg_id
                try:
                    event = _event_from_fields(fields)
                except Exception:
                    continue
                self.rows.append(
                    (
                        event.ts.isoformat(),
                        {
                            "stream": stream,
                            "event_type": type(event).__name__,
                            "ts": event.ts.isoformat(),
                            "recorded_at": self.now_fn(),
                            "payload": event.model_dump(mode="json"),
                        },
                    )
                )
                added += 1
        return added

    def start(self) -> None:
        import threading

        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                try:
                    self.poll()
                except Exception:
                    pass
                self._stop.wait(self.poll_s)

        self._thread = threading.Thread(target=loop, name="scene-lab-tap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                print("scene_lab: bus tap thread still running after 5 s; export may be partial")
        try:
            self.poll()
        except Exception:
            pass

    def write(self, path: Path, since: datetime | None = None) -> Path:
        rows = sorted(self.rows, key=lambda item: item[0])
        with path.open("w") as out:
            for ts, row in rows:
                if since is not None and datetime.fromisoformat(ts) < since:
                    continue
                out.write(json.dumps(row) + "\n")
        return path


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
        if export_file.name.endswith(".tap.jsonl"):
            export_file.unlink(missing_ok=True)
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
                context=_state(trace, run.thresholds.max_scene_s),
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
