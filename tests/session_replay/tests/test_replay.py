"""Small deterministic replay fixtures and the checked-in regression corpus."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from nc_shared.events import Activity, PersonState, Utterance

from session_replay.core import check_expectations, extract, read_jsonl, run_scenario

SCENARIOS = Path(__file__).parents[1] / "scenarios"


def _line(event, stream, *, observed=False):
    return {
        "stream": stream,
        "event_type": type(event).__name__,
        "ts": event.ts.isoformat(),
        "recorded_at": event.ts.timestamp(),
        "payload": event.model_dump(mode="json"),
        **({"observed": True} if observed else {}),
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_extract_filters_and_marks_observed(tmp_path):
    at = datetime(2026, 9, 22, 22, 0, tzinfo=UTC)
    person = PersonState(source="perceive", ts=at, state="standing", zone="bed", confidence=1)
    heard = Utterance(
        source="listen", ts=at + timedelta(seconds=1), text="Hello", confidence=1, duration_s=1
    )
    source = tmp_path / "raw.jsonl"
    target = tmp_path / "scenario.jsonl"
    rows = [
        _line(person, "person"),
        _line(heard, "speech_in"),
        {
            "stream": "frames",
            "event_type": "Frame",
            "ts": at.isoformat(),
            "payload": {"jpeg": "private"},
        },
        {
            "stream": "session",
            "event_type": "SessionState",
            "ts": at.isoformat(),
            "payload": {"source": "agent", "phase": "OBSERVING"},
        },
    ]
    _write(source, rows)
    assert extract(source, target, since=at + timedelta(seconds=1)) == 1
    assert read_jsonl(target)[0]["payload"]["text"] == "Hello"
    assert extract(source, target) == 3
    assert read_jsonl(target)[1]["observed"] is True
    assert all(row["event_type"] != "Frame" for row in read_jsonl(target))


def test_extract_keeps_only_text_free_llm_timing(tmp_path):
    at = datetime(2026, 9, 22, 22, tzinfo=UTC)
    source, target = tmp_path / "raw.jsonl", tmp_path / "scene.jsonl"
    activity = Activity(
        source="agent",
        service="agent",
        kind="compose",
        phase="end",
        duration_ms=3100,
        detail="private prompt text",
        ts=at,
    )
    _write(source, [_line(activity, "activity")])
    assert extract(source, target) == 1
    row = read_jsonl(target)[0]
    assert row["observed"] is True
    assert row["payload"] == {"kind": "compose", "phase": "end", "duration_ms": 3100.0}


def test_recorded_latency_blocks_input_and_uses_kind_durations(tmp_path):
    at = datetime(2026, 9, 22, 22, tzinfo=UTC)
    person = PersonState(source="perceive", ts=at, state="sitting_up", zone="bed", confidence=1)
    first = Utterance(
        source="listen",
        ts=at + timedelta(seconds=53),
        text="What time is it?",
        confidence=1,
        duration_s=1,
    )
    second = Utterance(
        source="listen",
        ts=at + timedelta(seconds=55),
        text="Hello again",
        confidence=1,
        duration_s=1,
    )
    rows = [_line(person, "person"), _line(first, "speech_in"), _line(second, "speech_in")]
    for kind, duration in (
        ("interpret", 1000),
        ("compose", 3000),
        ("interpret", 1000),
        ("compose", 3000),
    ):
        activity = Activity(
            source="agent",
            service="agent",
            kind=kind,
            phase="end",
            duration_ms=duration,
            ts=at + timedelta(seconds=60),
        )
        rows.append(_line(activity, "activity", observed=True))
    path = tmp_path / "scene.jsonl"
    _write(path, rows)
    spec = {"interpretations": {"What time is it?": {"intent": "confused_time", "distress": 0}}}
    with pytest.warns(UserWarning, match="missing recorded plan duration"):
        timeline = run_scenario(
            path, expect=spec, llm_mode="recorded", llm_latency="recorded", tail_s=60
        )
    compose = next(
        row
        for row in timeline
        if row["type"] == "Activity" and row["kind"] == "compose" and row["phase"] == "end"
    )
    late = next(row for row in timeline if row.get("heard") == "Hello again")
    say = next(
        row
        for row in timeline
        if row["type"] == "Say" and row.get("strategy") == "orient_time_place"
    )
    assert compose["duration_ms"] == 3000
    assert compose["t"] == late["t"] == 57
    assert say["t"] >= 57


def test_recorded_latency_warns_and_falls_back(tmp_path):
    at = datetime(2026, 9, 22, 22, tzinfo=UTC)
    person = PersonState(source="perceive", ts=at, state="sitting_up", zone="bed", confidence=1)
    heard = Utterance(
        source="listen", ts=at + timedelta(seconds=25), text="Hello", confidence=1, duration_s=1
    )
    path = tmp_path / "missing.jsonl"
    _write(path, [_line(person, "person"), _line(heard, "speech_in")])
    with pytest.warns(UserWarning, match="missing.jsonl: no recorded LLM durations"):
        timeline = run_scenario(
            path, llm_mode="recorded", llm_latency="recorded", llm_latency_fallback=2.5, tail_s=5
        )
    end = next(
        row
        for row in timeline
        if row["type"] == "Activity" and row["kind"] == "interpret" and row["phase"] == "end"
    )
    assert end["duration_ms"] == 2500


@pytest.mark.parametrize("mode", ["none", "recorded"])
def test_run_simulated_inputs_and_expectations(tmp_path, mode):
    at = datetime(2026, 9, 22, 22, 0, tzinfo=UTC)
    person = PersonState(source="perceive", ts=at, state="sitting_up", zone="bed", confidence=1)
    heard = Utterance(
        source="listen",
        ts=at + timedelta(seconds=1.25),
        text="Restroom please",
        confidence=1,
        duration_s=1,
    )
    path = tmp_path / "scenario.jsonl"
    _write(path, [_line(person, "person"), _line(heard, "speech_in")])
    spec = {"interpretations": {"Restroom please": {"intent": "need_restroom", "distress": 0}}}
    timeline = run_scenario(path, expect=spec, llm_mode=mode, tail_s=12)
    assert any(row.get("heard") == "Restroom please" and row["t"] == 1.25 for row in timeline)
    repeated = run_scenario(path, expect=spec, llm_mode=mode, tail_s=12)

    def stable(rows):
        return [{key: value for key, value in row.items() if key != "duration_ms"} for row in rows]

    assert stable(timeline) == stable(repeated)
    expected = {
        "expect": [
            {
                "after": {"heard": "Restroom please"},
                "within_s": 5,
                "events": [{"type": "GoalChanged", "to_goal": "restroom"}],
            }
        ]
    }
    passed, _ = check_expectations(timeline, expected)
    assert passed is (mode == "recorded")


def test_ordered_window_and_never():
    rows = [
        {"t": 0, "type": "IN", "heard": "go"},
        {"t": 1, "type": "Say"},
        {"t": 2, "type": "GoalChanged"},
        {"t": 5, "type": "Notify"},
    ]
    spec = {
        "expect": [
            {
                "after": {"heard": "go"},
                "within_s": 3,
                "events": [{"type": "GoalChanged"}, {"type": "Say"}],
            }
        ],
        "never": [{"type": "Notify"}],
    }
    passed, report = check_expectations(rows, spec)
    assert not passed
    assert any("FAIL expectation" in line for line in report)
    assert any("FAIL never" in line for line in report)


@pytest.mark.parametrize(
    "scenario",
    sorted(path for path in SCENARIOS.glob("*.jsonl") if path.with_suffix(".expect.yaml").exists()),
    ids=lambda path: path.stem,
)
def test_regression_scenario(scenario):
    from session_replay.core import load_expect

    spec = load_expect(scenario.with_suffix(".expect.yaml"))
    assert spec["llm"] in {"recorded", "none"}
    timeline = run_scenario(scenario, expect=spec, llm_mode=spec["llm"])
    passed, report = check_expectations(timeline, spec)
    assert passed, "\n".join(report)


def test_windowed_absent_rejects_only_events_in_anchored_window():
    rows = [
        {"t": 0, "type": "IN", "heard": "I'll go to bed"},
        {"t": 2, "type": "Say", "strategy": "acknowledge_return"},
        {"t": 12, "type": "Say", "strategy": "guided_return"},
    ]
    spec = {
        "expect": [
            {
                "after": {"heard": "I'll go to bed"},
                "within_s": 10,
                "events": [{"type": "Say", "strategy": "acknowledge_return"}],
                "absent": [{"type": "Say", "strategy": "guided_return"}],
            }
        ]
    }
    assert check_expectations(rows, spec)[0]
    spec["expect"][0]["within_s"] = 15
    assert not check_expectations(rows, spec)[0]


def test_cli_invariants_capture_inputs_outputs_and_decisions(tmp_path, capsys):
    from session_replay.__main__ import main

    scenario = SCENARIOS / "desk-restroom-2026-09-22.jsonl"
    assert (
        main(
            [
                "run",
                str(scenario),
                "--llm",
                "recorded",
                "--invariants",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "Invariants run:" in capsys.readouterr().out
    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert (run / "bugs.jsonl").exists()
    assert (run / "bugs.md").exists()
    scene = run / scenario.stem
    assert (scene / "report.json").exists()
    events = [json.loads(line) for line in (scene / "trace.jsonl").read_text().splitlines()]
    assert any(event["kind"] == "input" for event in events)
    assert any(event["kind"] == "output" for event in events)


def test_cli_latency_in_invariant_report(tmp_path, capsys):
    from session_replay.__main__ import main

    scenario = SCENARIOS / "desk-restroom-2026-09-22.jsonl"
    assert (
        main(
            [
                "run",
                str(scenario),
                "--llm",
                "recorded",
                "--llm-latency",
                "fixed:2.5",
                "--invariants",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "LLM latency: fixed:2.5" in capsys.readouterr().out
    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    report = json.loads((run / scenario.stem / "report.json").read_text())
    assert report["llm_latency"] == "fixed:2.5"


def test_decision_activity_survives_session_timeline():
    from zoneinfo import ZoneInfo

    from nc_shared.events import Activity
    from scene_lab.trace import from_session_replay

    from session_replay.core import TimelineBus

    at = datetime(2026, 9, 22, tzinfo=UTC)
    bus = TimelineBus(ZoneInfo("America/New_York"))
    bus.start = bus.now = at
    bus.publish(
        Activity(
            source="agent",
            service="agent",
            kind="decision",
            phase="end",
            detail='{"decision":"vetoed","rule":"test"}',
        )
    )
    assert from_session_replay(bus.timeline, "test").events[0].kind == "decision"


def test_cli_multi_scenario_invariants(tmp_path, capsys):
    from session_replay.__main__ import main

    paths = sorted(SCENARIOS.glob("*.jsonl"))[:2]
    assert (
        main(
            [
                "run",
                *(str(path) for path in paths),
                "--invariants",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert all((run / path.stem / "trace.jsonl").exists() for path in paths)
