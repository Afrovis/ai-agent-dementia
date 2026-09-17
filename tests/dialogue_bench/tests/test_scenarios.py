from pathlib import Path

import pytest
from agent.llm import Intent

from dialogue_bench.scenarios import EXPECTED_SCENARIO_COUNT, load_scenarios


def test_shipped_suite_has_exactly_50_unique_scenarios_and_every_intent():
    scenarios = load_scenarios()
    assert len(scenarios) == EXPECTED_SCENARIO_COUNT
    assert len({scenario.id for scenario in scenarios}) == EXPECTED_SCENARIO_COUNT
    assert {scenario.expected_intent for scenario in scenarios} == set(Intent)
    assert all(scenario.profile["caregiver_name"] == "Tom" for scenario in scenarios)
    assert all("things_to_avoid" in scenario.profile for scenario in scenarios)


def test_shipped_suite_applies_document_defaults_and_narrows_confused_time():
    scenarios = load_scenarios()
    by_id = {scenario.id: scenario for scenario in scenarios}

    default_scenario = by_id["restroom-01"]
    assert default_scenario.must == ("addresses_by_name",)
    assert default_scenario.must_not == (
        "conjunction_but",
        "avoid_terms",
        "states_clock_time",
        "invents_proper_noun",
    )
    assert default_scenario.avoid_terms == ("hospital",)

    confused_time_scenarios = [s for s in scenarios if s.expected_intent.value == "confused_time"]
    assert len(confused_time_scenarios) == 7
    for scenario in confused_time_scenarios:
        assert scenario.must_not == ("conjunction_but", "avoid_terms", "invents_proper_noun")
        assert scenario.must == ("addresses_by_name",)


@pytest.mark.parametrize(
    "document, message",
    [
        ("not_scenarios: []\n", "scenarios list"),
        (
            "scenarios:\n  - id: duplicate\n    utterance: hi\n    expected_intent: fine\n"
            "  - id: duplicate\n    utterance: hello\n    expected_intent: fine\n",
            "unique",
        ),
        (
            "scenarios:\n  - id: bad\n    utterance: hi\n    expected_intent: imaginary\n",
            "invalid expected_intent",
        ),
        ("default_profile: bad\nscenarios: []\n", "default_profile must be a mapping"),
        (
            "default_must: [not_a_real_check]\nscenarios: []\n",
            "document defaults has unknown check",
        ),
        (
            "scenarios:\n  - id: bad\n    utterance: hi\n    expected_intent: fine\n"
            "    must: [not_a_real_check]\n",
            "scenario 0 has unknown check",
        ),
        (
            "scenarios:\n  - id: bad\n    utterance: hi\n    expected_intent: fine\n"
            "    must_not: [not_a_real_check]\n",
            "scenario 0 has unknown check",
        ),
    ],
)
def test_loader_rejects_bad_documents(tmp_path: Path, document: str, message: str):
    path = tmp_path / "bad.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_scenarios(path, expected_count=None)


def test_loader_rejects_accidentally_truncated_suite(tmp_path: Path):
    path = tmp_path / "short.yaml"
    path.write_text(
        "scenarios:\n  - id: only\n    utterance: hello\n    expected_intent: fine\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected 50 scenarios, found 1"):
        load_scenarios(path)


def test_per_scenario_check_lists_replace_rather_than_merge_document_defaults(tmp_path: Path):
    path = tmp_path / "defaults.yaml"
    path.write_text(
        "default_must: [addresses_by_name]\n"
        "default_must_not: [conjunction_but, avoid_terms]\n"
        "default_avoid_terms: [hospital]\n"
        "scenarios:\n"
        "  - id: uses_defaults\n"
        "    utterance: hello\n"
        "    expected_intent: fine\n"
        "  - id: narrows_must_not\n"
        "    utterance: hello\n"
        "    expected_intent: fine\n"
        "    must_not: [conjunction_but]\n"
        "    must: []\n"
        "    avoid_terms: []\n",
        encoding="utf-8",
    )
    scenarios = load_scenarios(path, expected_count=None)
    by_id = {scenario.id: scenario for scenario in scenarios}

    assert by_id["uses_defaults"].must == ("addresses_by_name",)
    assert by_id["uses_defaults"].must_not == ("conjunction_but", "avoid_terms")
    assert by_id["uses_defaults"].avoid_terms == ("hospital",)

    assert by_id["narrows_must_not"].must == ()
    assert by_id["narrows_must_not"].must_not == ("conjunction_but",)
    assert by_id["narrows_must_not"].avoid_terms == ()
