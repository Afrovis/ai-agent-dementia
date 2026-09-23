"""Small deterministic replay fixtures and the checked-in regression corpus."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from nc_shared.events import PersonState, Utterance

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
    assert timeline == run_scenario(path, expect=spec, llm_mode=mode, tail_s=12)
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
