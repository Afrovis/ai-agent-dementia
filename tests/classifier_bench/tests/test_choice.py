from __future__ import annotations

from pathlib import Path

import pytest
from decision_bench.schema import (
    GOALS,
    NOTIFY_LEVELS,
    SCENARIOS_DIR,
    STRATEGIES,
    load_scenarios,
)

from classifier_bench.ask import ask_questions, read_jsonl
from classifier_bench.build import serialize_state
from classifier_bench.choice import (
    BASELINE_SEED,
    CHOICE_PROMPT_PATH,
    DIMENSIONS,
    build_choice_questions,
    choice_schema,
    choice_user_prompt,
    outcome,
    score_choices,
    validate_choice,
)


def _question(qid: str = "s::c::goal") -> dict[str, object]:
    return {
        "qid": qid,
        "scenario": "s",
        "checkpoint": "c",
        "dimension": "goal",
        "category": "restroom",
        "noisy": False,
        "options": list(DIMENSIONS["goal"]),
        "acceptable": ["bed"],
        "must_not": ["comfort"],
        "silence": False,
        "state": "summary: test",
        "question": "What now?",
    }


def test_options_match_schema() -> None:
    assert set(DIMENSIONS["strategy"]) - {"none"} == STRATEGIES
    assert set(DIMENSIONS["goal"]) - {"none"} == GOALS
    assert set(DIMENSIONS["notify"]) - {"none"} == NOTIFY_LEVELS - {"any"}


def test_build_real_fixtures() -> None:
    questions = build_choice_questions(SCENARIOS_DIR)
    assert len(questions) == 47 * 3
    assert len({item["qid"] for item in questions}) == len(questions)
    first = questions[0]
    scenario = next(item for item in load_scenarios(SCENARIOS_DIR) if item.id == first["scenario"])
    checkpoint = next(item for item in scenario.checkpoints if item.id == first["checkpoint"])
    assert first["state"] == serialize_state(scenario, checkpoint)


@pytest.mark.parametrize(
    ("choice", "acceptable", "must_not", "silence", "error", "expected"),
    [
        ("bed", ["bed"], [], False, False, "pass"),
        ("bed", ["any"], [], False, False, "pass"),
        ("bed", ["bed"], ["bed"], False, False, "critical"),
        ("bed", [], ["any"], False, False, "critical"),
        ("bed", [], [], False, False, "unlabelled"),
        ("none", ["bed"], ["any"], True, False, "miss"),
        ("none", [], ["any"], False, False, "pass"),
        ("none", [], [], True, False, "pass"),
        ("none", [], ["bed"], False, False, "unlabelled"),
        ("none", [], [], False, True, "error"),
    ],
)
def test_outcome(choice, acceptable, must_not, silence, error, expected) -> None:
    record = dict(
        _question(), choice=choice, acceptable=acceptable, must_not=must_not, silence=silence
    )
    if error:
        record["error"] = "failed"
    assert outcome(record) == expected


def test_ask_retry_validation_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    temperatures = []

    def fake(_system, _prompt, _schema, _model, temperature):
        temperatures.append(temperature)
        return {"choice": "invalid" if temperature == 0 else "bed", "confidence": 0.7}

    assert pytest.raises(
        ValueError, validate_choice, {"choice": "invalid", "confidence": 0.7}, _question()
    )
    kwargs = {
        "guidelines_text": "rules",
        "runner": fake,
        "system_prompt_path": CHOICE_PROMPT_PATH,
        "prompt_builder": choice_user_prompt,
        "schema_for": choice_schema,
        "validate": validate_choice,
    }
    written = ask_questions([_question()], path, **kwargs)
    assert written[0]["choice"] == "bed"
    assert written[0]["attempts"] == 2
    assert temperatures == [0.0, 0.3]
    assert ask_questions([_question()], path, **kwargs) == []
    assert len(read_jsonl(path)) == 1


def test_baselines_and_random_seed() -> None:
    questions = [_question("a"), _question("b")]
    questions[1]["acceptable"] = []
    questions[1]["must_not"] = ["any"]
    report = score_choices([], questions)
    none = report["baselines"]["always_none"]["overall"]
    assert (none["pass"], none["miss"], none["critical"]) == (1, 1, 0)
    assert report["uniform_random_seed"] == BASELINE_SEED
    assert (
        report["baselines"]["uniform_random"]
        == score_choices([], questions)["baselines"]["uniform_random"]
    )


def test_prompt_contains_guidelines_and_options() -> None:
    prompt = choice_user_prompt(_question(), "one\ntwo\n")
    assert "one\ntwo" in prompt
    assert "Pick one `goal`. Options: bed, comfort, restroom, none" in prompt
