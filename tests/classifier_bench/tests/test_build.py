from __future__ import annotations

from pathlib import Path

import yaml
from decision_bench.schema import load_scenario

from classifier_bench.build import build_questions, serialize_state


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_build_agree_disputed_and_absent(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios"
    annotations = tmp_path / "annotations"
    scenarios.mkdir()
    annotations.mkdir()
    _write_yaml(
        scenarios / "sample.yaml",
        {
            "id": "sample",
            "category": "restroom",
            "summary": "Gets up and asks for the bathroom.",
            "start": "02:10",
            "timeline": [
                {"t": 0, "person": {"state": "sitting_up", "zone": "bed"}},
                {"t": 10, "utterance": {"text": "Bathroom?", "confidence": 0.8}},
                {"t": 30, "person": {"state": "walking", "zone": "bathroom_path"}},
            ],
            "checkpoints": [{"id": "ask", "window": [0, 20], "question": "What is appropriate?"}],
        },
    )
    _write_yaml(
        annotations / "sample.yaml",
        {
            "scenario": "sample",
            "checkpoints": [
                {
                    "id": "ask",
                    "acceptable": [{"strategy": "path_light"}, {"goal": "restroom"}],
                    "must_not": [{"say": "memory_question"}],
                }
            ],
            "second_opinion": {
                "checkpoints": [
                    {
                        "id": "ask",
                        "acceptable": [{"strategy": "path_light"}],
                        "must_not": [
                            {"say": "memory_question"},
                            {"notify": "critical"},
                        ],
                    }
                ]
            },
        },
    )

    questions = build_questions(annotations, scenarios)

    by_qid = {item["qid"]: item for item in questions}
    assert len(questions) == 4
    assert by_qid["sample::ask::strategy:path_light"]["truth"] == "acceptable"
    assert by_qid["sample::ask::say:memory_question"]["truth"] == "must_not"
    assert by_qid["sample::ask::goal:restroom"]["set"] == "disputed"
    assert by_qid["sample::ask::goal:restroom"]["run2"] == "absent"
    assert by_qid["sample::ask::notify:critical"]["run1"] == "absent"


def test_state_serializer_shape_and_window_truncation(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    _write_yaml(
        path,
        {
            "id": "sample",
            "category": "false_alarm",
            "summary": "A compact summary.",
            "start": "01:05",
            "timeline": [
                {"t": 0, "person": {"state": "in_bed", "zone": "bed"}},
                {"t": 5, "utterance": {"text": "Hello", "confidence": 0.7}},
                {"t": 10, "person": {"state": "standing", "zone": "door"}},
                {"t": 11, "utterance": {"text": "future"}},
            ],
            "checkpoints": [{"id": "now", "window": [5, 10], "question": "What now?"}],
        },
    )
    scenario = load_scenario(path)

    state = serialize_state(scenario, scenario.checkpoints[0])

    assert "summary: A compact summary." in state
    assert "start: 01:05" in state
    assert "window: 5-10 seconds" in state
    assert 't=5 said: "Hello" conf=0.7' in state
    assert "t=10 person=standing/door conf=0.9" in state
    assert "future" not in state
    assert state.endswith("question: What now?")
