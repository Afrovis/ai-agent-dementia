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
