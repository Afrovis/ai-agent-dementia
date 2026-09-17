"""Run dialogue scenarios and score intent accuracy and safe composition."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from agent.llm import Intent, LLMClient
from agent.rules import validate_say
from agent.session import DEFAULT_GOAL, INTENT_GOALS

from dialogue_bench.scenarios import DialogueScenario


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    expected_intent: Intent
    actual_intent: Intent | None
    intent_correct: bool
    composition_text: str | None
    composition_safe: bool
    composition_failure: str | None
    composition_copies_template: bool
    interpret_latency_seconds: float
    compose_latency_seconds: float


@dataclass(frozen=True)
class ModelResult:
    model: str
    scenarios: tuple[ScenarioResult, ...]

    @property
    def scenario_count(self) -> int:
        return len(self.scenarios)

    @property
    def intent_correct(self) -> int:
        return sum(item.intent_correct for item in self.scenarios)

    @property
    def intent_accuracy(self) -> float | None:
        return self.intent_correct / self.scenario_count if self.scenario_count else None

    @property
    def safe_compositions(self) -> int:
        return sum(item.composition_safe for item in self.scenarios)

    @property
    def composition_pass_rate(self) -> float | None:
        return self.safe_compositions / self.scenario_count if self.scenario_count else None

    @property
    def template_copies(self) -> int:
        """Replies that just repeat the caregiver phrase, ignoring the utterance."""
        return sum(item.composition_copies_template for item in self.scenarios)

    @property
    def distinct_compositions(self) -> int:
        return len(
            {_words(item.composition_text) for item in self.scenarios if item.composition_text}
        )

    @property
    def mean_interpret_latency_seconds(self) -> float | None:
        if not self.scenarios:
            return None
        return sum(item.interpret_latency_seconds for item in self.scenarios) / len(self.scenarios)

    @property
    def max_interpret_latency_seconds(self) -> float | None:
        return max(
            (item.interpret_latency_seconds for item in self.scenarios),
            default=None,
        )

    @property
    def mean_compose_latency_seconds(self) -> float | None:
        if not self.scenarios:
            return None
        return sum(item.compose_latency_seconds for item in self.scenarios) / len(self.scenarios)

    @property
    def max_compose_latency_seconds(self) -> float | None:
        return max((item.compose_latency_seconds for item in self.scenarios), default=None)

    @property
    def failures_by_reason(self) -> dict[str, int]:
        return dict(
            Counter(
                item.composition_failure
                for item in self.scenarios
                if item.composition_failure is not None
            )
        )

    @property
    def confusion(self) -> dict[str, dict[str, int]]:
        labels = [intent.value for intent in Intent] + ["missing"]
        matrix = {expected: {actual: 0 for actual in labels} for expected in labels[:-1]}
        for item in self.scenarios:
            actual = item.actual_intent.value if item.actual_intent is not None else "missing"
            matrix[item.expected_intent.value][actual] += 1
        return matrix

    @property
    def intent_accuracy_by_class(self) -> dict[str, float | None]:
        scores: dict[str, float | None] = {}
        for intent in Intent:
            matching = [item for item in self.scenarios if item.expected_intent == intent]
            scores[intent.value] = (
                sum(item.intent_correct for item in matching) / len(matching) if matching else None
            )
        return scores


def run_model(
    model_name: str,
    client: LLMClient,
    scenarios: list[DialogueScenario],
    *,
    clock: Callable[[], float] = perf_counter,
) -> ModelResult:
    """Run both model tasks for every scenario; a rejected response is a measured failure."""
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        interpret_started = clock()
        interpretation = client.interpret(scenario.utterance, scenario.turns, scenario.profile)
        interpret_latency = clock() - interpret_started
        # The goal the agent would hold after this interpretation.
        goal = DEFAULT_GOAL
        if interpretation is not None:
            goal = INTENT_GOALS.get(interpretation.intent.value, DEFAULT_GOAL)
        template = render_phrase(scenario.caregiver_phrase_template, scenario.profile)
        compose_started = clock()
        composition = client.compose(
            "validate_and_redirect",
            template,
            scenario.profile,
            scenario.time_words,
            scenario.scene_note,
            scenario.utterance,
            goal,
        )
        compose_latency = clock() - compose_started
        actual_intent = interpretation.intent if interpretation is not None else None
        if composition is None:
            composition_text = None
            composition_safe = False
            composition_failure = "missing or structurally invalid response"
        else:
            composition_text = composition.text
            verdict = validate_say(
                composition_text, seconds_since_last_say=None, min_gap_seconds=8.0
            )
            composition_safe = verdict.accepted
            composition_failure = verdict.reason
        results.append(
            ScenarioResult(
                scenario_id=scenario.id,
                expected_intent=scenario.expected_intent,
                actual_intent=actual_intent,
                intent_correct=actual_intent == scenario.expected_intent,
                composition_text=composition_text,
                composition_safe=composition_safe,
                composition_failure=composition_failure,
                composition_copies_template=composition_text is not None
                and _words(composition_text) == _words(template),
                interpret_latency_seconds=interpret_latency,
                compose_latency_seconds=compose_latency,
            )
        )
    return ModelResult(model=model_name, scenarios=tuple(results))


def render_phrase(template: str, profile: dict[str, object]) -> str:
    """Fill the name placeholders the way ``agent.strategies.render_template`` does."""
    name = str(profile.get("preferred_address") or profile.get("name") or "")
    return template.replace("{name_vocative}", f", {name}" if name else "").replace("{name}", name)


def _words(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z]+", (text or "").lower()))
