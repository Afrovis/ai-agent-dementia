"""Loading and validation for the 50 synthetic dialogue scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from agent.llm import Intent

DEFAULT_SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "scenarios.yaml"
EXPECTED_SCENARIO_COUNT = 50


@dataclass(frozen=True)
class DialogueScenario:
    id: str
    utterance: str
    expected_intent: Intent
    turns: tuple[str, ...] = ()
    profile: dict[str, object] = field(default_factory=dict)
    time_words: str = "3 o'clock at night"
    scene_note: str | None = None
    caregiver_phrase_template: str = (
        "It's alright{name_vocative}, let's rest now and talk more in the morning."
    )


def _as_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _parse_scenario(
    raw: object, index: int, default_profile: dict[str, object]
) -> DialogueScenario:
    if not isinstance(raw, dict):
        raise ValueError(f"scenario {index} must be a mapping")
    try:
        expected_intent = Intent(raw.get("expected_intent"))
    except ValueError as exc:
        raise ValueError(f"scenario {index} has an invalid expected_intent") from exc

    turns = raw.get("turns", [])
    if not isinstance(turns, list) or any(not isinstance(turn, str) for turn in turns):
        raise ValueError(f"scenario {index} turns must be a list of strings")
    profile_overrides = raw.get("profile", {})
    if not isinstance(profile_overrides, dict):
        raise ValueError(f"scenario {index} profile must be a mapping")
    profile = {**default_profile, **profile_overrides}
    scene_note = raw.get("scene_note")
    if scene_note is not None and not isinstance(scene_note, str):
        raise ValueError(f"scenario {index} scene_note must be a string or null")

    return DialogueScenario(
        id=_as_text(raw.get("id"), f"scenario {index} id"),
        utterance=_as_text(raw.get("utterance"), f"scenario {index} utterance"),
        expected_intent=expected_intent,
        turns=tuple(turns[-3:]),
        profile=profile,
        time_words=_as_text(
            raw.get("time_words", "3 o'clock at night"), f"scenario {index} time_words"
        ),
        scene_note=scene_note,
        caregiver_phrase_template=_as_text(
            raw.get(
                "caregiver_phrase_template",
                "It's alright{name_vocative}, let's rest now and talk more in the morning.",
            ),
            f"scenario {index} caregiver_phrase_template",
        ),
    )


def load_scenarios(
    path: str | Path = DEFAULT_SCENARIOS_PATH,
    *,
    expected_count: int | None = EXPECTED_SCENARIO_COUNT,
) -> list[DialogueScenario]:
    """Load a scenario file, rejecting malformed or accidentally truncated suites."""
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("scenarios"), list):
        raise ValueError("scenario document must contain a scenarios list")
    default_profile = document.get("default_profile", {})
    if not isinstance(default_profile, dict):
        raise ValueError("default_profile must be a mapping")
    scenarios = [
        _parse_scenario(raw, index, default_profile)
        for index, raw in enumerate(document["scenarios"])
    ]
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario ids must be unique")
    if expected_count is not None and len(scenarios) != expected_count:
        raise ValueError(f"expected {expected_count} scenarios, found {len(scenarios)}")
    return scenarios
