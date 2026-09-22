from __future__ import annotations

import json
from pathlib import Path

from classifier_bench.ask import ask_questions, read_jsonl, user_prompt


def _question(qid: str = "s::c::goal:bed") -> dict[str, object]:
    return {
        "qid": qid,
        "scenario": "s",
        "checkpoint": "c",
        "action": {"goal": "bed"},
        "action_kind": "goal",
        "set": "agree",
        "truth": "acceptable",
        "run1": "acceptable",
        "run2": "acceptable",
        "state": "summary: test\nquestion: What now?",
        "question": "What now?",
    }


def test_prompt_contains_guidelines_verbatim_and_one_action() -> None:
    prompt = user_prompt(_question(), "line one\nline two\n")
    assert "line one\nline two" in prompt
    assert prompt.count('{"goal": "bed"}') == 1


def test_retry_failure_continue_and_resume(tmp_path: Path) -> None:
    output = tmp_path / "answers.jsonl"
    calls: list[tuple[str, float]] = []

    def fake(_system, prompt, _schema, _model, temperature):
        calls.append((prompt, temperature))
        if "goal" in prompt and temperature == 0:
            return {"label": "bad", "confidence": 1}
        if "goal" in prompt:
            return {"label": "acceptable", "confidence": 0.8}
        raise RuntimeError("offline")

    second = _question("s::c::say:any")
    second["action"] = {"say": "any"}
    second["action_kind"] = "say"
    written = ask_questions([_question(), second], output, guidelines_text="rules", runner=fake)

    assert written[0]["label"] == "acceptable"
    assert written[0]["attempts"] == 2
    assert written[1]["attempts"] == 2
    assert "RuntimeError: offline" in written[1]["error"]
    assert [temperature for _, temperature in calls] == [0.0, 0.3, 0.0, 0.3]
    assert ask_questions([_question(), second], output, guidelines_text="rules", runner=fake) == []
    assert len(read_jsonl(output)) == 2
    assert all(isinstance(json.loads(line), dict) for line in output.read_text().splitlines())


def test_interrupted_run_flushes_answers_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "answers.jsonl"
    questions = [_question(f"s::c::goal:{index}") for index in range(4)]
    interrupted_calls = 0

    def interrupt_part_way(_system, _prompt, _schema, _model, _temperature):
        nonlocal interrupted_calls
        interrupted_calls += 1
        if interrupted_calls == 3:
            assert [item["qid"] for item in read_jsonl(output)] == [
                questions[0]["qid"],
                questions[1]["qid"],
            ]
            raise KeyboardInterrupt
        return {"label": "acceptable", "confidence": 0.8}

    try:
        ask_questions(
            questions,
            output,
            guidelines_text="rules",
            runner=interrupt_part_way,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected the fake backend to interrupt the run")

    assert [item["qid"] for item in read_jsonl(output)] == [
        questions[0]["qid"],
        questions[1]["qid"],
    ]

    resumed_calls = 0

    def finish_run(_system, _prompt, _schema, _model, _temperature):
        nonlocal resumed_calls
        resumed_calls += 1
        return {"label": "irrelevant", "confidence": 0.7}

    written = ask_questions(questions, output, guidelines_text="rules", runner=finish_run)

    assert resumed_calls == 2
    assert [item["qid"] for item in written] == [questions[2]["qid"], questions[3]["qid"]]
    assert [item["qid"] for item in read_jsonl(output)] == [
        question["qid"] for question in questions
    ]


def test_progress_is_reported_every_25_answers(tmp_path: Path, capsys) -> None:
    questions = [_question(f"s::c::goal:{index}") for index in range(51)]
    calls = 0

    def fake(_system, _prompt, _schema, _model, _temperature):
        nonlocal calls
        calls += 1
        label = "irrelevant" if calls == 13 else "acceptable"
        return {"label": label, "confidence": 0.8}

    ask_questions(
        questions,
        tmp_path / "answers.jsonl",
        guidelines_text="rules",
        runner=fake,
    )

    assert capsys.readouterr().err.splitlines() == [
        "25/51 answered (1 abstentions, 0 failures)",
        "50/51 answered (1 abstentions, 0 failures)",
    ]
