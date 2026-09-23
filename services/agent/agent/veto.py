"""The veto layer: deterministic prohibitions checked before anything is published.

`docs/CHOICE_VETO_HANDOFF.md` Task 2. The decision layer (the state machine
today, a model picking one option per dimension later) proposes; this module
may only say no. It never substitutes an action -- a veto that picks a
replacement is a second, untested decision layer -- so a denial means the
proposed `Show`/`Say`/`Notify`/`LightCommand` is simply not published, and
if everything is denied the agent does nothing, which is the safe outcome
at night.

It is not a general safety system. It encodes the specific prohibitions that
`gemma4:e4b-mlx` got wrong in the 2026-09-22 classification probe
(`docs/CLASSIFIER_BENCH.md`): 16 forbidden actions it called acceptable,
every one at confidence 0.8-0.9. Each rule cites the clause in
`tests/decision_bench/guidelines.md` it comes from, and each probe case is a
regression test in `tests/test_veto.py`. What is enforced here, and what is
still left to judgment, is listed in `docs/VETO.md`.

Pure and synchronous: no I/O, no clock, no model call. `VetoContext` is a
snapshot the caller builds from the session; `check` is a handful of string
comparisons and short regex searches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GUIDED_RETURN = "guided_return"
ORIENT_TIME_PLACE = "orient_time_place"


@dataclass(frozen=True)
class VetoContext:
    """What the veto may look at. Built by `agent.main.veto_context`."""

    phase: str
    goal: str
    person_state: str | None
    recent_utterances: tuple[str, ...]
    # In bed and silent since lying down: the person has settled on their own.
    settled: bool
    in_night_window: bool
    things_to_avoid: tuple[str, ...] = ()
    restroom_need_resolved: bool = False


@dataclass(frozen=True)
class Proposal:
    """One proposed outgoing action.

    `kind` is `strategy` (its `Show`, and its `Say` if it has one), `say`
    (the final text, after rendering and composition), `notify` or `light`.
    `terminal` marks the escalation strategy's sentence, which is never
    silenced: it tells a person who may be on the floor that help is coming.
    """

    kind: str
    value: str
    text: str | None = None
    terminal: bool = False


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    rule: str | None = None
    clause: str | None = None
    reason: str | None = None


ALLOW = Verdict(allowed=True)

# A stated or signalled toilet need, including an accident that still needs
# dealing with. Matched against the session's recent utterances.
_TOILET_RE = re.compile(
    r"\b(?:toilet|loo|lavatory|bathroom|restroom|pee|wee|pish|piss|"
    r"wet (?:myself|the bed|my(?:self)?)|need to go|have to go)\b",
    re.IGNORECASE,
)
# Another stated need the person is waiting on: cold, pain, feeling unwell,
# a call for help. Not exhaustive; see docs/VETO.md.
_NEED_RE = re.compile(
    r"\b(?:help|cold|freezing|can't get warm|hurts?|hurting|pain|ache|aching|"
    r"feel (?:strange|sick|ill|unwell|funny)|can't breathe|thirsty|hungry|"
    r"don't know what to do)\b",
    re.IGNORECASE,
)
# The same phrasing `decision_bench` scores as `memory_question`.
_MEMORY_RE = re.compile(
    r"\b(?:do you remember|don't you remember|can you remember|remember when|"
    r"don't you know|do you know what day|do you recall)\b",
    re.IGNORECASE,
)
# "Do not mention the hospital" -> "hospital". A thing to avoid phrased as
# anything else ("Avoid loud or urgent language") is style, not a term, and is
# left to the wording checks and the model.
_AVOID_TERM_RE = re.compile(
    r"^\s*(?:do not|don't|never|avoid)\s+"
    r"(?:mention(?:ing)?|say(?:ing)?|talk(?:ing)? about|us(?:e|ing) the words?|bring(?:ing)? up)"
    r"\s+(?:the\s+|a\s+|an\s+|their\s+|his\s+|her\s+)?(?P<term>[^.;]+?)\s*\.?\s*$",
    re.IGNORECASE,
)


def avoid_terms(things_to_avoid: tuple[str, ...]) -> tuple[str, ...]:
    """The literal terms a profile's `things_to_avoid` forbids mentioning."""
    terms = []
    for item in things_to_avoid:
        match = _AVOID_TERM_RE.match(item)
        if match:
            terms.append(match.group("term").strip().lower())
    return tuple(terms)


def _deny(rule: str, clause: str, reason: str) -> Verdict:
    return Verdict(allowed=False, rule=rule, clause=clause, reason=reason)


def _check_strategy(strategy: str, context: VetoContext) -> Verdict:
    if strategy == GUIDED_RETURN:
        if context.goal == "restroom" or (
            not context.restroom_need_resolved
            and any(_TOILET_RE.search(text) for text in context.recent_utterances)
        ):
            return _deny(
                "no_redirect_from_toilet_need",
                "TOIL-01",
                "guided_return while a stated toilet need is unmet",
            )
        if any(_NEED_RE.search(text) for text in context.recent_utterances):
            return _deny(
                "no_redirect_from_stated_need",
                "NICE-01",
                "guided_return while a stated need is unmet",
            )
    if strategy == ORIENT_TIME_PLACE:
        if context.settled or context.person_state == "in_bed" or context.phase == "COOLDOWN":
            return _deny(
                "no_orienting_a_settling_person",
                "NICE-05",
                "orient_time_place while the person is settling or settled",
            )
        if not context.in_night_window:
            # Its sentence says it is night-time; outside the night window
            # that contradicts someone who is simply up for the day.
            return _deny(
                "no_night_orientation_by_day",
                "AA-03",
                "orient_time_place outside the night window",
            )
    return ALLOW


def _check_say(proposal: Proposal, context: VetoContext) -> Verdict:
    if context.settled and not proposal.terminal:
        return _deny(
            "silence_when_settled",
            "NICE-05",
            "speech to a person who has settled back in bed",
        )
    text = proposal.text or ""
    if _MEMORY_RE.search(text):
        return _deny("no_memory_question", "VAL-01", "speech asks the person to recall")
    for term in avoid_terms(context.things_to_avoid):
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            return _deny("no_avoided_term", "NICE-04", "speech uses a term the profile avoids")
    return ALLOW


def check(proposal: Proposal, context: VetoContext) -> Verdict:
    """Allow `proposal`, or deny it with the rule id and the clause it cites.

    Reasons never include the proposed text, which may paraphrase private
    speech and must not be copied into logs.
    """
    if proposal.kind == "strategy":
        return _check_strategy(proposal.value, context)
    if proposal.kind == "say":
        return _check_say(proposal, context)
    # `notify` and `light` carry no prohibitions yet. Escalation deadlines and
    # levels stay with the state machine (docs/VETO.md); they pass through here
    # so a future rule has one place to live.
    return ALLOW
