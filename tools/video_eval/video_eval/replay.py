"""Build a bridge replay, run it through Docker Compose, and assess the bus trace."""

from __future__ import annotations

import base64
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from video_eval.common import matching_meta, read_jsonl, update_index, write_jsonl, write_meta
from video_eval.paths import EvalPaths
from video_eval.reconcile import validate_timeline

RETAINED_STREAMS = frozenset({"person", "session", "say", "notify", "light"})


def build_raw_replay(
    frames: list[dict[str, Any]],
    root: Path,
    output: Path,
    *,
    variant: str,
    start: datetime,
    source_width: int = 320,
    source_height: int = 240,
) -> int:
    path_key = "bridge_path" if variant == "squash" else "bridge_letterbox_path"
    records: list[dict[str, Any]] = []
    for frame in frames:
        if path_key not in frame:
            raise RuntimeError(f"manifest has no {path_key}; prepare that variant first")
        ts = start + timedelta(seconds=float(frame["t_s"]))
        payload = {
            "jpeg": base64.b64encode((root / frame[path_key]).read_bytes()).decode("ascii"),
            "width": 320,
            "height": 240,
            "source_width": source_width,
            "source_height": source_height,
            "source_kind": "browser",
            "source": "video_eval",
            "session_id": None,
            "ts": ts.isoformat(),
        }
        records.append(
            {
                "stream": "frames_raw",
                "event_type": "RawFrame",
                "ts": ts.isoformat(),
                "recorded_at": start.timestamp() + float(frame["t_s"]),
                "payload": payload,
            }
        )
    write_jsonl(output, records)
    return len(records)


def _relative_events(bus_rows: list[dict[str, Any]], start: datetime) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in bus_rows:
        payload = row.get("payload") or {}
        ts = datetime.fromisoformat(str(row["ts"]))
        events.append(
            {
                "event_type": row.get("event_type"),
                "stream": row.get("stream"),
                "t_s": (ts - start).total_seconds(),
                **payload,
            }
        )
    return sorted(events, key=lambda row: row["t_s"])


def compare_e2e(
    bus_rows: list[dict[str, Any]],
    reference: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    start: datetime,
    absent_limit_seconds: float = 600.0,
) -> dict[str, Any]:
    events = _relative_events(bus_rows, start)
    persons = [row for row in events if row["event_type"] == "PersonState"]
    sessions = [row for row in events if row["event_type"] == "SessionState"]
    offline = [
        (row["state"], row["zone"])
        for row in predictions
        if row.get("published") and row.get("state") is not None
    ]
    live = [(row.get("state"), row.get("zone")) for row in persons]
    perception_match = live == offline

    timeline = reference.get("timeline") or []
    up = [row for row in timeline if row["state"] in {"sitting_up", "standing", "walking"}]
    first_up = float(up[0]["from_s"]) if up else None
    observing_times = [row["t_s"] for row in sessions if row.get("phase") == "OBSERVING"]
    engaged_times = [row["t_s"] for row in sessions if row.get("phase") == "ENGAGED"]
    observing_ok = (
        None
        if first_up is None
        else any(first_up <= value <= first_up + 5 for value in observing_times)
    )
    engaged_ok = (
        None
        if first_up is None
        else any(first_up + 20 <= value <= first_up + 25 for value in engaged_times)
    )

    restroom_entries = [
        float(row["from_s"]) for row in timeline if row.get("zone") in {"door", "bathroom_path"}
    ]
    restroom_goals = [
        row["t_s"]
        for row in events
        if (row["event_type"] == "SessionState" and row.get("goal") == "restroom")
        or (row["event_type"] == "GoalChanged" and row.get("to_goal") == "restroom")
    ]
    restroom_ok = None
    if restroom_entries:
        restroom_ok = all(
            any(expected <= actual <= expected + 5 for actual in restroom_goals)
            for expected in restroom_entries
        )

    floor_expected = any(row["state"] == "on_floor" for row in timeline)
    long_absent_expected = any(
        row["state"] == "absent"
        and float(row["to_s"]) - float(row["from_s"]) >= absent_limit_seconds
        for row in timeline
    )
    risk_expected = floor_expected or long_absent_expected
    escalated = any(row.get("phase") == "ESCALATED" for row in sessions)
    no_unsafe_escalation = risk_expected or not escalated
    checks = {
        "offline_live_perception_match": perception_match,
        "observing_within_5s": observing_ok,
        "engaged_after_20s": engaged_ok,
        "restroom_goal_within_5s": restroom_ok,
        "no_escalation_without_reference_risk": no_unsafe_escalation,
    }
    measured = [value for value in checks.values() if value is not None]
    return {
        "checks": checks,
        "passed": bool(measured) and all(measured),
        "live_person_transitions": len(live),
        "offline_person_transitions": len(offline),
        "session_transitions": [
            {"t_s": row["t_s"], "phase": row.get("phase"), "goal": row.get("goal")}
            for row in sessions
        ],
    }


def render_e2e_report(clip_id: str, result: dict[str, Any], *, night_window: str) -> str:
    lines = [
        f"# End-to-end video replay: {clip_id}",
        "",
        f"Night window: `{night_window}` (equal endpoints make it always active).",
        "",
        f"Overall: {'PASS' if result['passed'] else 'FAIL'}",
        "",
        "## Checks",
        "",
    ]
    for name, value in result["checks"].items():
        verdict = "not measurable" if value is None else ("PASS" if value else "FAIL")
        lines.append(f"- `{name}`: {verdict}")
    lines.extend(
        [
            "",
            f"Live person transitions: {result['live_person_transitions']}",
            f"Offline person transitions: {result['offline_person_transitions']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"required command is not installed: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed ({exc.returncode}): {' '.join(command)}") from exc


def _filter_trace(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("stream") in RETAINED_STREAMS]
    write_jsonl(path, rows)
    return rows


def replay_clip(
    clip_id: str,
    *,
    root: Path | None = None,
    tag: str | None = None,
    variant: str = "squash",
    force: bool = False,
    speed: float = 1.0,
    settle_seconds: float = 5.0,
    repo_root: Path | None = None,
) -> dict[str, object]:
    if speed <= 0:
        raise ValueError("end-to-end replay speed must be positive")
    started = time.monotonic()
    paths = EvalPaths.for_clip(clip_id, root)
    try:
        clip_card = yaml.safe_load((paths.clip / "clip.yaml").read_text(encoding="utf-8")) or {}
        source_width = int(clip_card["video"]["width"])
        source_height = int(clip_card["video"]["height"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        source_width, source_height = 320, 240
    reference_path = paths.labels / "reference.yaml"
    if not reference_path.exists():
        raise RuntimeError("confirmed reference.yaml is required for replay")
    reference = yaml.safe_load(reference_path.read_text(encoding="utf-8")) or {}
    if not reference.get("confirmed_by") or not reference.get("confirmed_at"):
        raise RuntimeError("reference.yaml is not human-confirmed")
    validate_timeline(reference.get("timeline"))
    predictions = sorted(paths.predictions.glob("*.jsonl"))
    if tag:
        predictions = [paths.predictions / f"{tag}.jsonl"]
    if len(predictions) != 1 or not predictions[0].exists():
        raise RuntimeError("select exactly one existing prediction with --tag")
    prediction_meta_path = paths.predictions / f"{predictions[0].stem}.meta.json"
    try:
        prediction_meta = json.loads(prediction_meta_path.read_text(encoding="utf-8"))
        prediction_parameters = prediction_meta["parameters"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"valid prediction metadata is required: {prediction_meta_path}"
        ) from exc
    if prediction_parameters.get("variant") != variant:
        raise RuntimeError("replay variant must match the selected offline prediction")
    if prediction_parameters.get("gated") is not True:
        raise RuntimeError("end-to-end replay comparison requires a gated offline prediction")

    run_date = datetime.now(UTC).date().isoformat()
    run_dir = paths.e2e / run_date
    report_path, bus_path = run_dir / "report.md", run_dir / "bus.jsonl"
    meta_path, input_path = run_dir / "replay.meta.json", run_dir / "frames_raw.jsonl"
    parameters = {
        "clip_id": clip_id,
        "prediction_tag": predictions[0].stem,
        "prediction_sha256": _file_digest(predictions[0]),
        "reference_sha256": _file_digest(reference_path),
        "speed": speed,
        "variant": variant,
        "vision_enabled": False,
        "night_window": "00:00–00:00",
    }
    if (
        not force
        and report_path.exists()
        and bus_path.exists()
        and matching_meta(meta_path, parameters)
    ):
        return {"status": "skipped", "run": run_date}
    run_dir.mkdir(parents=True, exist_ok=True)
    bus_path.unlink(missing_ok=True)
    replay_start: datetime | None = None
    frame_count = 0

    checkout = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    override_path = run_dir / "compose.override.yaml"
    override = {
        "services": {
            "perceive": {
                "environment": {
                    "PERCEIVE_VISION_ENABLED": "false",
                    "ZONES_PATH": "/eval/zones.yaml",
                },
                "volumes": [f"{paths.clip}:/eval:ro"],
            }
        }
    }
    override_path.write_text(yaml.safe_dump(override, sort_keys=False), encoding="utf-8")
    compose = [
        "docker",
        "compose",
        "-f",
        str(checkout / "docker-compose.yml"),
        "-f",
        str(override_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "AGENT_NIGHT_START": "00:00",
            "AGENT_NIGHT_END": "00:00",
            "CAPTURE_SOURCE": "browser",
            "DASHBOARD_PASSWORD": secrets.token_urlsafe(24),
            "AGENT_OBSERVE_SECONDS": "20",
            "AGENT_COOLDOWN_SECONDS": "300",
            "AGENT_IN_BED_STABLE_SECONDS": "120",
            "AGENT_FLOOR_LIMIT_SECONDS": "0",
            "AGENT_ABSENT_LIMIT_SECONDS": "600",
            "AGENT_RESTROOM_TIMEOUT_SECONDS": "900",
            "AGENT_ZONE_CONFIRM_READINGS": "3",
            "AGENT_SAY_MIN_GAP_SECONDS": "8",
            "PERCEIVE_VISION_ENABLED": "false",
            "PERCEIVE_POSE_BACKEND": str(prediction_parameters["backend"]),
        }
    )
    optional_environment = {
        "PERCEIVE_CONFIRM_FRAMES": prediction_parameters.get("confirm_frames"),
        "PERCEIVE_MIN_CONFIDENCE": prediction_parameters.get("min_confidence"),
        "PERCEIVE_WALK_THRESHOLD": prediction_parameters.get("walk_threshold"),
    }
    environment.update(
        {key: str(value) for key, value in optional_environment.items() if value is not None}
    )
    recorder: subprocess.Popen[str] | None = None
    try:
        _run(compose + ["build"], cwd=checkout, env=environment)
        _run(compose + ["up", "-d"], cwd=checkout, env=environment)
        recorder = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "nc_shared.replay",
                "record",
                "redis://localhost:6379",
                str(bus_path),
            ],
            cwd=checkout,
            env=environment,
            text=True,
        )
        time.sleep(2)
        replay_start = datetime.now(UTC)
        frame_count = build_raw_replay(
            read_jsonl(paths.frames),
            paths.root,
            input_path,
            variant=variant,
            start=replay_start,
            source_width=source_width,
            source_height=source_height,
        )
        _run(
            [
                sys.executable,
                "-m",
                "nc_shared.replay",
                "play",
                "redis://localhost:6379",
                str(input_path),
                "--speed",
                str(speed),
            ],
            cwd=checkout,
            env=environment,
        )
        time.sleep(settle_seconds)
    finally:
        operation_failed = sys.exc_info()[0] is not None
        if recorder is not None and recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
            try:
                recorder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                recorder.terminate()
                recorder.wait(timeout=5)
        if bus_path.exists():
            _filter_trace(bus_path)
        cleanup_errors: list[str] = []
        try:
            _run(
                compose + ["exec", "-T", "bus", "redis-cli", "DEL", "frames_raw", "frames"],
                cwd=checkout,
                env=environment,
            )
        except RuntimeError as exc:
            cleanup_errors.append(str(exc))
        try:
            _run(compose + ["down"], cwd=checkout, env=environment)
        except RuntimeError as exc:
            cleanup_errors.append(str(exc))
        if cleanup_errors and not operation_failed:
            raise RuntimeError("; ".join(cleanup_errors))

    rows = _filter_trace(bus_path)
    assert replay_start is not None
    comparison = compare_e2e(
        rows,
        reference,
        read_jsonl(predictions[0]),
        start=replay_start,
        absent_limit_seconds=float(environment["AGENT_ABSENT_LIMIT_SECONDS"]),
    )
    report_path.write_text(
        render_e2e_report(clip_id, comparison, night_window="00:00–00:00"), encoding="utf-8"
    )
    write_meta(
        meta_path,
        command="replay",
        parameters=parameters,
        started_at=started,
        versions=("docker", "pillow", "pyyaml"),
    )
    update_index(paths.root, clip_id, f"replay_{run_date}", "complete")
    return {
        "status": "complete",
        "run": run_date,
        "frames": frame_count,
        "events": len(rows),
        "passed": comparison["passed"],
    }


def _file_digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
