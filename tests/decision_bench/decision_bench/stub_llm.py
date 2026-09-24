"""Deterministic LLM client for tests and smoke runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.llm import Composition, Intent, Interpretation, Plan


class StubLLM:
    """Small keyword-based implementation of the agent's ``LLMClient`` protocol."""

    def __init__(
        self,
        *,
        interpretation: Interpretation | None = None,
        compose_text: str = "You're safe here, let's rest now.",
        plan_result: Plan | None = None,
    ) -> None:
        self.interpretation = interpretation
        self.compose_text = compose_text
        self.plan_result = plan_result

    def interpret(
        self,
        utterance: str,
        turns: Sequence[str],
        profile: Mapping[str, object],
        person_state: str | None = None,
    ) -> Interpretation | None:
        del turns, profile, person_state
        if self.interpretation is not None:
            return self.interpretation
        text = utterance.lower()
        if (
            "back to bed" in text
            or "go to bed" in text
            or "return to bed" in text
            or any(
                phrase in text
                for phrase in (
                    "back from the restroom",
                    "back from the bathroom",
                    "back from the toilet",
                    "finished in the restroom",
                    "finished in the bathroom",
                    "finished in the toilet",
                    "finished with the restroom",
                    "finished with the bathroom",
                )
            )
        ):
            return Interpretation(intent=Intent.WANTS_BED, distress=0)
        if any(word in text for word in ("toilet", "bathroom", "restroom", "loo", "wee")):
            return Interpretation(intent=Intent.NEED_RESTROOM, distress=0)
        if any(word in text for word in ("hurt", "pain", "ache")):
            return Interpretation(intent=Intent.PAIN, distress=2)
        if any(word in text for word in ("kids", "children", "work", "home", "go")):
            return Interpretation(intent=Intent.WANTS_TO_LEAVE, distress=0)
        if "where is" in text or "looking for" in text:
            return Interpretation(intent=Intent.LOOKING_FOR_PERSON, distress=0)
        if "what time" in text or "is it morning" in text:
            return Interpretation(intent=Intent.CONFUSED_TIME, distress=0)
        if any(word in text for word in ("help", "please", "crying")):
            return Interpretation(intent=Intent.UNCLEAR, distress=2)
        if "fine" in text or "okay" in text or "that's better" in text:
            return Interpretation(intent=Intent.FINE, distress=0)
        return Interpretation(intent=Intent.UNCLEAR, distress=0)

    def compose(
        self,
        strategy_name: str,
        caregiver_phrase_template: str,
        profile: Mapping[str, object],
        time_words: str,
        scene_note: str | None,
        utterance: str | None = None,
        goal: str | None = None,
    ) -> Composition | None:
        del (
            strategy_name,
            caregiver_phrase_template,
            profile,
            time_words,
            scene_note,
            utterance,
            goal,
        )
        return Composition(text=self.compose_text)

    def plan(self, session_state: Mapping[str, Any], profile: Mapping[str, object]) -> Plan | None:
        del session_state, profile
        return self.plan_result
