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


def test_compose_gets_the_interpreted_goal_and_copies_are_counted():
    template = "It's alright{name_vocative}, let's rest now."
    scenarios = [
        DialogueScenario(
            id=identifier,
            utterance="hello",
            expected_intent=intent,
            profile={"preferred_address": "Jean"},
            caregiver_phrase_template=template,
        )
        for identifier, intent in (("toilet", Intent.NEED_RESTROOM), ("time", Intent.FINE))
    ]
    client = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.NEED_RESTROOM, distress=0),
            Interpretation(intent=Intent.FINE, distress=0),
        ],
        compositions=[
            Composition(text="Jean, the restroom is just through the door."),
            Composition(text="It's alright Jean, let's rest now!"),
        ],
    )

    result = run_model("fake", client, scenarios)

    composes = [payload for name, payload in client.calls if name == "compose"]
    assert [payload["goal"] for payload in composes] == ["restroom", "return_to_bed"]
    assert composes[0]["caregiver_phrase_template"] == "It's alright, Jean, let's rest now."
    assert [item.composition_copies_template for item in result.scenarios] == [False, True]
    assert result.template_copies == 1
    assert result.distinct_compositions == 2


def test_run_model_scores_prompt_rule_assertions():
    scenarios = [
        DialogueScenario(
            id="clean",
            utterance="hello",
            expected_intent=Intent.FINE,
            profile={"preferred_address": "Jean"},
            must=("addresses_by_name",),
            must_not=("conjunction_but", "avoid_terms"),
            avoid_terms=("hospital",),
        ),
        DialogueScenario(
            id="violates_but_and_missing_name",
            utterance="hello",
            expected_intent=Intent.FINE,
            profile={"preferred_address": "Jean"},
            must=("addresses_by_name",),
            must_not=("conjunction_but", "avoid_terms"),
            avoid_terms=("hospital",),
        ),
        DialogueScenario(
            id="no_checks_configured",
            utterance="hello",
            expected_intent=Intent.FINE,
        ),
        DialogueScenario(
            id="missing_composition",
            utterance="hello",
            expected_intent=Intent.FINE,
            must=("addresses_by_name",),
        ),
    ]
    client = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.FINE, distress=0),
            Interpretation(intent=Intent.FINE, distress=0),
            Interpretation(intent=Intent.FINE, distress=0),
            Interpretation(intent=Intent.FINE, distress=0),
        ],
        compositions=[
            Composition(text="Jean, everything is settled now."),
            Composition(text="Everything is settled now, but rest well."),
            Composition(text="Everything is settled now."),
            None,
        ],
    )

    result = run_model("fake", client, scenarios)

    clean, violating, no_checks, missing = result.scenarios
    assert clean.assertions_passed
    assert clean.assertion_failures == ()
    assert not violating.assertions_passed
    assert {outcome.name for outcome in violating.assertion_failures} == {
        "addresses_by_name",
        "conjunction_but",
    }
    assert no_checks.check_outcomes == ()
    assert missing.check_outcomes == ()

    assert result.scenarios_with_assertions == 2
    assert result.assertion_pass_rate == 0.5
    assert result.violations_by_check == {"addresses_by_name": 1, "conjunction_but": 1}
