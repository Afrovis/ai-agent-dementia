"""A promoted failure stays replayable and fails until the agent is fixed."""

import json
from datetime import UTC, datetime, timedelta

import pytest
import yaml
from decision_bench.schema import load_scenario
from session_replay.core import check_expectations, load_expect, read_jsonl, run_scenario

from scene_lab.promote import promote, template


def _row(origin, t, stream, event_type, payload):
    ts = (origin + timedelta(seconds=t)).isoformat()
    return {
        "stream": stream,
        "event_type": event_type,
        "ts": ts,
        "payload": {
            "ts": ts,
            "source": {"person": "scene_lab", "speech_in": "listen"}.get(stream, "agent"),
            **payload,
        },
    }


def _run(tmp_path):
    root = tmp_path / "run"
    scene = root / "question"
    scene.mkdir(parents=True)
    origin = datetime(2026, 9, 22, 2, 0, tzinfo=UTC)
    rows = [
        _row(
            origin,
            0,
            "person",
            "PersonState",
            {"state": "sitting_up", "zone": "bed", "confidence": 0.9},
        ),
        _row(
            origin,
            25,
            "speech_in",
            "Utterance",
            {"text": "What time is it?", "confidence": 0.9, "duration_s": 1.2},
        ),
        _row(
            origin,
            26,
            "activity",
            "Activity",
            {"kind": "interpret", "phase": "end", "duration_ms": 300},
        ),
        _row(
            origin,
            28,
            "session",
            "SessionState",
            {"phase": "ENGAGED", "goal": "return_to_bed", "strategy_index": 0},
        ),
        _row(
            origin,
            30,
            "activity",
            "Activity",
            {"kind": "decision", "phase": "end", "detail": "pending_say_dropped"},
        ),
    ]
    (scene / "export.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (scene / "scene.yaml").write_text(
        yaml.safe_dump(
            {"id": "question", "category": "conversation", "start": "02:00", "duration_s": 60}
        )
    )
    (root / "bugs.jsonl").write_text(
        json.dumps(
            {
                "scene": "question",
                "t": 30,
                "check": "TT-1",
                "summary": "no reply",
                "evidence": ["Utterance@25"],
            }
        )
        + "\n"
    )
    return root


def test_promote_both_replays_recorded_question_reply(tmp_path):
    root = _run(tmp_path)
    files = promote(
        root, scene="question", at=30, to="both", out_dir=tmp_path / "out", claude=False
    )
    rows = read_jsonl(files["session_replay"])
    assert [r["event_type"] for r in rows] == [
        "PersonState",
        "Utterance",
        "Activity",
        "SessionState",
        "InterpretationMap",
    ]
    assert rows[2]["payload"] == {"kind": "interpret", "phase": "end", "duration_ms": 300}
    spec = load_expect(files["expect"])
    assert spec["status"] == "draft"
    assert spec["llm"] == "recorded"
    assert rows[-1]["payload"]["interpretations"] == spec["interpretations"]
    assert (files["expect"].parent / "promoted-question-30.scene.yaml").exists()
    timeline = run_scenario(
        files["session_replay"], expect=spec, llm_mode="recorded", llm_latency="recorded"
    )
    passed, _ = check_expectations(timeline, spec)
    assert passed
    bench = load_scenario(files["decision_bench"])
    assert not bench.labelled
    assert [round(e.t) for e in bench.timeline] == [0, 25]
    for path in files.values():
        assert not any(
            key in path.read_text().lower() for key in ('"frames"', '"audio"', '"pcm16"', '"jpeg"')
        )


def test_promote_without_nearby_bug_reports_and_keeps_inputs(tmp_path, capsys):
    root = _run(tmp_path)
    files = promote(root, scene="question", at=25, out_dir=tmp_path / "out")
    assert files["session_replay"].exists()
    assert "expect" not in files
    assert "No bug entry" in capsys.readouterr().out


def test_promote_rejects_media_before_writing(tmp_path):
    root = _run(tmp_path)
    export = root / "question" / "export.jsonl"
    rows = read_jsonl(export)
    rows[0]["payload"]["jpeg"] = "secret"
    export.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="media key"):
        promote(root, scene="question", at=30, out_dir=tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.jsonl"))


@pytest.mark.parametrize(
    "check,state,event_key",
    [
        ("TT-1", "standing", "events"),
        ("TT-5", "standing", "min_spacing_s"),
        ("TT-6", "absent", "absent"),
        ("SM-1", "in_bed", "absent"),
        ("SM-3", "standing", "events"),
        ("SM-5", "standing", "absent"),
    ],
)
def test_templates(check, state, event_key):
    rows = [
        {"event_type": "PersonState", "t": 0, "payload": {"state": state, "zone": "bed"}},
        {"event_type": "Utterance", "t": 1, "payload": {"text": "Hello?"}},
    ]
    item = template(check, rows, 5, {"evidence": ["Say@5"]})
    assert item[event_key]
    assert item["after"] in ({"heard": "Hello?"}, {"person": {"state": state, "zone": "bed"}})
