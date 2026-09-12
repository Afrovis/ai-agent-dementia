from agent.llm import Composition, FakeLLM, Intent, Interpretation

from dialogue_bench.scenarios import DialogueScenario
from dialogue_bench.scoring import run_model


def _scenario(identifier: str, expected: Intent) -> DialogueScenario:
    return DialogueScenario(id=identifier, utterance="hello", expected_intent=expected)


def test_run_model_scores_intents_and_deterministic_tone_rules():
    scenarios = [
        _scenario("safe", Intent.FINE),
        _scenario("forbidden", Intent.PAIN),
        _scenario("question", Intent.UNCLEAR),
        _scenario("missing", Intent.NEED_RESTROOM),
    ]
    client = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.FINE, distress=0),
            Interpretation(intent=Intent.UNCLEAR, distress=1),
            Interpretation(intent=Intent.UNCLEAR, distress=0),
            None,
        ],
        compositions=[
            Composition(text="I hear you, and we can rest together now."),
            Composition(text="No, you need to stay here."),
            Composition(text="Do you remember where your bed is?"),
            None,
        ],
    )

    ticks = iter(float(value) for value in range(16))
    result = run_model("fake", client, scenarios, clock=lambda: next(ticks))

    assert result.intent_correct == 2
    assert result.intent_accuracy == 0.5
    assert result.safe_compositions == 1
    assert result.composition_pass_rate == 0.25
    assert result.confusion["pain"]["unclear"] == 1
    assert result.confusion["need_restroom"]["missing"] == 1
    assert result.intent_accuracy_by_class["fine"] == 1.0
    assert result.intent_accuracy_by_class["looking_for_person"] is None
    assert result.mean_interpret_latency_seconds == 1.0
    assert result.max_interpret_latency_seconds == 1.0
    assert result.mean_compose_latency_seconds == 1.0
    assert result.max_compose_latency_seconds == 1.0
    assert result.failures_by_reason["missing or structurally invalid response"] == 1
    assert any("forbidden phrase" in reason for reason in result.failures_by_reason)
    assert any("question" in reason for reason in result.failures_by_reason)


def test_every_scenario_drives_both_calls_with_context():
    scenario = DialogueScenario(
        id="context",
        utterance="Where is Tom?",
        expected_intent=Intent.LOOKING_FOR_PERSON,
        turns=("I woke up.",),
        profile={"caregiver_name": "Tom"},
        scene_note="Standing beside the bed.",
    )
    client = FakeLLM(
        interpretations=[Interpretation(intent=Intent.LOOKING_FOR_PERSON, distress=1)],
        compositions=[Composition(text="Tom is nearby, and you are safe here with me.")],
    )

    run_model("fake", client, [scenario])

    assert [name for name, _ in client.calls] == ["interpret", "compose"]
    assert client.calls[0][1]["last_turns"] == ["I woke up."]
    assert client.calls[1][1]["scene_note"] == "Standing beside the bed."
    assert client.calls[1][1]["profile"] == {"caregiver_name": "Tom"}
