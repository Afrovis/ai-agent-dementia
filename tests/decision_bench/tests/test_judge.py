from datetime import datetime

import pytest

from decision_bench.annotate import AnnotatorError, _subscription_env
from decision_bench.judge import (
    Sentence,
    first_occurrences,
    judge_input,
    judge_sentences,
    output_schema,
    sentences_from_trace,
)
from decision_bench.runner import Trace, TraceEntry

PROFILE = {"name": "Jean", "caregiver_name": "Tom", "restroom_location": "Left"}


def _trace() -> Trace:
    trace = Trace(scenario_id="s", start=datetime(2026, 1, 1, 2, 0), entries=[])
    trace.entries += [
        TraceEntry(0, "PersonState", {"input": True, "state": "standing", "zone": "door"}),
        TraceEntry(5, "Utterance", {"input": True, "text": "Where is Tom?"}),
        TraceEntry(6, "Say", {"text": "Tom is  here."}),
        TraceEntry(7, "Notify", {"level": "attention"}),
        TraceEntry(8, "Say", {"text": "Someone is coming to help."}),
    ]
    return trace


def _verdict(index: int, claim: bool = False) -> dict[str, object]:
    return {
        "index": index,
        "correction_of_reality": False,
        "infantilising": False,
        "unsupported_claim": claim,
        "evidence": "e",
    }


def test_sentences_carry_what_the_agent_knew():
    first, second = sentences_from_trace(_trace(), PROFILE)
    assert first.text == "Tom is here."
    assert first.context["person_last_said"] == "Where is Tom?"
    assert first.context["camera"]["zone"] == "door"
    assert first.context["notified"] is False
    assert second.context["notified"] is True
    assert "night" in str(first.context["time_words"])


def test_judge_input_and_schema():
    sentences = sentences_from_trace(_trace(), PROFILE)
    text = judge_input(sentences)
    assert "restroom_location: Left" in text and "Tom is here." in text
    schema = output_schema(2)
    item = schema["properties"]["verdicts"]["items"]
    assert set(item["required"]) == {
        "index",
        "correction_of_reality",
        "infantilising",
        "unsupported_claim",
        "evidence",
    }


def test_judge_retries_once_then_fails():
    sentences = [Sentence("a", PROFILE), Sentence("b", PROFILE)]
    calls = []

    def runner(path, prompt, schema, model, effort):
        calls.append(prompt)
        verdicts = [_verdict(0)] if len(calls) == 1 else [_verdict(0), _verdict(1, True)]
        return {"structured_output": {"verdicts": verdicts}, "model": "claude-opus-5"}

    results, model = judge_sentences(sentences, runner=runner)
    assert model == "claude-opus-5"
    assert [item["unsupported_claim"] for item in results] == [False, True]
    assert "missing indices: [1]" in calls[1]

    def bad(*args):
        return {"structured_output": {"verdicts": []}}

    with pytest.raises(AnnotatorError):
        judge_sentences(sentences, runner=bad)


def test_first_occurrences_keeps_trace_order():
    sentences = [Sentence("a", {}), Sentence("b", {}), Sentence("a", {})]
    assert [s.text for s in first_occurrences(sentences, {"a", "b"})] == ["a", "b"]


def test_claude_runs_never_see_api_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("HOME", "/home/x")
    env = _subscription_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert env["HOME"] == "/home/x"
