"""The deterministic rule layer for `agent`'s session state machine (issue #12).

This is the layer HANDOFF.md rule 1 is about: "The LLM never owns safety.
State transitions, escalation timers, and hard limits live in deterministic
code in the `agent` service. The LLM interprets, composes, and proposes.
Every LLM proposal passes through `rules.validate()` before it has any
effect." Nothing in this module calls an LLM. Issue #15's structured
interpretation, composition, and planning results all reach deterministic
session, strategy, or speech checks before taking effect. `Session` also
routes every transition it computes itself through `validate()`.

`ALLOWED_TRANSITIONS` is the transition table HANDOFF.md section 6
describes -- `IDLE -> OBSERVING -> ENGAGED -> COOLDOWN -> IDLE`, with
`ENGAGED -> ESCALATED -> COOLDOWN` and `OBSERVING -> IDLE` ("if back
`in_bed`, return to `IDLE`") -- plus every edge rule 5 needs. Rule 5
("`on_floor` or `absent` from the room beyond the configured limit skips
every strategy and goes straight to `escalate_phone`") has no "only if a
session is already live" qualifier and outranks section 6's phase summary,
so it can fire from *any* phase, including ones with no nudging session
running yet:

- `OBSERVING -> ESCALATED` and `ENGAGED -> ESCALATED`: rule 5 firing while
  a nudging session is already under way.
- `IDLE -> ESCALATED`: rule 5 firing with no session running at all, e.g.
  a fall straight from `in_bed` to `on_floor` with no intervening
  `sitting_up`/`standing` reading, or `on_floor`/`absent` outside the
  night window (rule 5 is not gated by the night window; see
  `agent.session.Session._rule5_transition`).
- `COOLDOWN -> ESCALATED`: rule 5 firing during cooldown, when no new
  *nudging* session may start but a fall still must not be ignored.

Beyond that, nothing else. In particular:

- `ESCALATED -> COOLDOWN` is allowed; `ESCALATED -> ENGAGED` is not. Once a
  session has escalated, the rule layer never lets it quietly go back to
  ordinary strategy-running.
- `COOLDOWN` only exits to `IDLE` or, via rule 5, to `ESCALATED`. No new
  *nudging* session may start during cooldown, so
  `COOLDOWN -> OBSERVING`/`ENGAGED` are absent from the table on purpose.

`validate_goal()` is the goal-change equivalent of `validate()`, added for
issue #13: it checks a proposed goal change against `agent.goals.
ALLOWED_GOAL_CHANGES` the same way `validate()` checks a proposed phase
change against `ALLOWED_TRANSITIONS`, and for the same reason -- issue
#15's `plan` proposes goal changes, and HANDOFF.md rule 1 requires
every one of those proposals to pass through this layer before it has any
effect. `agent.session.Session` already routes its own deterministic goal
switches through it too, exactly as it does for phase changes.

`validate_strategy()` is issue #14's real replacement for the seam
`validate()` used to carry as a caller-supplied `strategy_disabled` bool
with no one actually computing it. `agent.strategies.StrategyEngine` is
now that caller: every candidate it considers while selecting or advancing
a strategy is run through `validate_strategy()`, which rejects it if the
engine reports it disabled (the caregiver turned it off in config) or
still on cooldown (its `cooldown_seconds` has not elapsed since it was
last used) -- an ordinary, expected outcome the engine reacts to by
trying the next candidate in order, not an exception. `validate()` keeps
its own `strategy`/`strategy_disabled` parameters and delegates to this
function internally, so a caller pairing a phase transition with a
strategy choice (as `agent.session.Session._apply` does for the strategy
selected on entering `ENGAGED` or forced on entering `ESCALATED`) still
gets one combined accept/reject decision from a single call, unchanged
from issue #12's shape.

`validate_say()` is HANDOFF.md rule 3 ("Spoken output is one sentence,
then silence for at least 8 seconds ... Never the words 'no', 'you
can't', 'you're wrong'. Validate, then redirect.") made into deterministic
code every outgoing `Say` passes through, in `agent.main`, regardless of
whether the text came from a caregiver's template or `compose`'s
LLM output (issue #15) -- the LLM never owns safety, so this check sits
after composition, not instead of it. See its own docstring for exactly
what the memory-testing-question check does and does not catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from agent.goals import ALLOWED_GOAL_CHANGES
from agent.profile import PersonProfile

# HANDOFF.md rule 3, verbatim: never these words/phrases in a `Say`, checked
# case-insensitively as whole words/phrases so "known" does not trip on
# "no". Matched against the raw text `agent.main` is about to publish,
# whatever composed it.
_FORBIDDEN_PHRASES: tuple[str, ...] = ("no", "you can't", "you're wrong")

# A single interior `.`/`!`/`?` (with anything non-whitespace on both
# sides) means more than one sentence. A trailing one is fine and does not
# split the text into a second, empty sentence.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]+")

# Conservative, form-only interrogative check -- see `validate_say`'s
# docstring for exactly what this does and does not catch.
_QUESTION_STARTERS = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "do you",
    "did you",
    "does",
    "can you",
    "could you",
    "would you",
    "will you",
    "remember",
    "recall",
    "is it",
    "isn't it",
    "was it",
    "wasn't it",
)

# Composition is allowed to repeat a caregiver-authored calming phrase such
# as "Tom is nearby", but the model must not strengthen that phrase into a
# promise of presence, arrival, checking, or help. This intentionally small,
# explicit list is a deterministic proxy, not semantic fact checking: it
# will miss paraphrases, which is why a rejected candidate falls back to the
# caregiver's fixed template and why prompt instructions remain important.
_CAREGIVER_CLAIM_PHRASES = (
    r"is\s+here",
    r"is\s+right\s+here",
    r"is\s+coming",
    r"will\s+come",
    r"comes",
    r"is\s+on\s+(?:his|her|their)\s+way",
    r"will\s+check",
    r"can\s+check",
    r"checks\s+on",
    r"helps\s+you",
    r"will\s+help",
    r"is\s+with\s+you",
    r"stays\s+with\s+you",
)

# Keep this explicit clock-reading pattern aligned with dialogue_bench's
# states_clock_time check. Spoken day-parts are allowed; exact times are not.
_CLOCK_TIME_RE = re.compile(
    r"\bo'?clock\b|\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b|\bmidnight\b|\bnoon\b",
    re.IGNORECASE,
)


class Phase(StrEnum):
    """The five session phases from HANDOFF.md section 6."""

    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    ENGAGED = "ENGAGED"
    COOLDOWN = "COOLDOWN"
    ESCALATED = "ESCALATED"


ALLOWED_TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    # `ESCALATED` here is rule 5 firing with no nudging session running.
    Phase.IDLE: frozenset({Phase.OBSERVING, Phase.ESCALATED}),
    Phase.OBSERVING: frozenset({Phase.IDLE, Phase.ENGAGED, Phase.ESCALATED}),
    Phase.ENGAGED: frozenset({Phase.COOLDOWN, Phase.ESCALATED}),
    Phase.ESCALATED: frozenset({Phase.COOLDOWN}),
    # `ESCALATED` here is rule 5 firing during cooldown.
    Phase.COOLDOWN: frozenset({Phase.IDLE, Phase.ESCALATED}),
}


@dataclass(frozen=True)
class RuleResult:
    """The outcome of `validate()`: distinguishes an ordinary rejection
    (`accepted=False`, `reason` explaining why) from acceptance, without
    raising -- an LLM proposing an illegal transition, or a disabled
    strategy, is an everyday event this layer exists to handle quietly,
    not an exceptional one."""

    accepted: bool
    reason: str | None = None


def validate(
    current: Phase,
    proposed: Phase,
    *,
    strategy: str | None = None,
    strategy_disabled: bool = False,
) -> RuleResult:
    """Decide whether `current -> proposed` may happen at all.

    Rejects any transition not present in `ALLOWED_TRANSITIONS`, regardless
    of who is proposing it -- `agent.session.Session`'s own logic today, an
    LLM's `plan` output from issue #15 tomorrow.

    `strategy`/`strategy_disabled` let a caller pair this phase check with
    a strategy choice in one call -- `agent.session.Session._apply` does
    this for the strategy selected on entering `ENGAGED` or forced on
    entering `ESCALATED`. `strategy_disabled` is real state now (issue
    #14): `agent.strategies.StrategyEngine` computes whether `strategy` is
    disabled in config or still on cooldown before calling this, exactly
    like an illegal phase transition. The check itself is delegated to
    `validate_strategy`, which the engine also calls directly when
    choosing between candidates with no phase change involved (e.g.
    advancing to the next strategy within `ENGAGED`).
    """
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if proposed not in allowed:
        return RuleResult(
            accepted=False,
            reason=f"{current.value} -> {proposed.value} is not an allowed transition",
        )

    if strategy_disabled:
        return validate_strategy(strategy, disabled=True)

    return RuleResult(accepted=True, reason=None)


def validate_strategy(strategy: str | None, *, disabled: bool) -> RuleResult:
    """Decide whether `strategy` may be selected right now.

    `disabled` is computed by the caller (`agent.strategies.StrategyEngine`):
    true if the caregiver turned the strategy off in config, or if it is
    still on its configured cooldown. This function does not track either
    of those itself -- it is the single point every strategy choice is
    routed through before it takes effect, per HANDOFF.md rule 1, not the
    place cooldown state lives.
    """
    if disabled:
        return RuleResult(
            accepted=False,
            reason=f"strategy {strategy!r} is disabled or on cooldown",
        )
    return RuleResult(accepted=True, reason=None)


def validate_goal(current: str, proposed: str) -> RuleResult:
    """Decide whether `current -> proposed` may happen at all, for goals.

    Same shape and the same never-raise-for-an-ordinary-rejection
    behaviour as `validate()`: rejects any goal change not present in
    `agent.goals.ALLOWED_GOAL_CHANGES`, regardless of who is proposing it
    -- `agent.session.Session`'s own deterministic switches today, an
    LLM's `plan` output from issue #15 tomorrow. An unknown goal on either
    side (not in `agent.goals.GOALS`) is rejected the same way, since it
    simply cannot appear as a key or member of `ALLOWED_GOAL_CHANGES`.
    """
    allowed = ALLOWED_GOAL_CHANGES.get(current, frozenset())
    if proposed not in allowed:
        return RuleResult(
            accepted=False,
            reason=f"{current} -> {proposed} is not an allowed goal change",
        )
    return RuleResult(accepted=True, reason=None)


def validate_composition(text: str, profile: PersonProfile) -> RuleResult:
    """Reject two known ways model-composed speech invents or weakens facts.

    This gate is deliberately narrower than `validate_say`: it applies only
    to the text returned by `LLMClient.compose`, before `agent.main` decides
    whether to use that text or the caregiver's fixed fallback template.
    It rejects the whole word "but", whose contrast can cancel an attempted
    acknowledgement, and a caregiver name followed within four intervening
    words by one of `_CAREGIVER_CLAIM_PHRASES`.

    The caregiver check is an explicit proxy for common model failures, not
    a general natural-language entailment system. In particular, it permits
    profile wording such as "Tom is nearby" or "Tom is near" while catching
    stronger claims such as "Tom is here", "Tom can check on you", and
    "Tom helps you settle". It will miss unlisted paraphrases; keeping the
    list inspectable and falling back on rejection is preferable to an
    opaque heuristic deciding what a person hears at night.

    Reasons never include `text`: callers log the reason, and model-composed
    speech may contain private utterance-derived material that must not be
    copied into logs.
    """
    if _CLOCK_TIME_RE.search(text):
        return RuleResult(accepted=False, reason="composition states an exact clock time")
    if re.search(r"\bbut\b", text, flags=re.IGNORECASE):
        return RuleResult(accepted=False, reason="composition contains the word 'but'")

    caregiver_name = profile.caregiver_name.strip()
    if caregiver_name:
        name_pattern = rf"(?<!\w){re.escape(caregiver_name)}(?!\w)"
        few_words = r"(?:[^\w]+[\w'-]+){0,4}[^\w]+"
        claims = "(?:" + "|".join(_CAREGIVER_CLAIM_PHRASES) + ")"
        if re.search(name_pattern + few_words + claims, text, flags=re.IGNORECASE):
            return RuleResult(
                accepted=False,
                reason="composition makes an unsupported caregiver presence or arrival claim",
            )

    return RuleResult(accepted=True, reason=None)


def validate_say(
    text: str,
    *,
    seconds_since_last_say: float | None,
    min_gap_seconds: float,
) -> RuleResult:
    """HANDOFF.md rule 3, as a deterministic gate every outgoing `Say` must
    pass before `agent.main` publishes it -- whether the text is a
    caregiver's fixed template (every strategy this issue implements) or
    an LLM's `compose` output (issue #15). Rejecting is the everyday,
    expected outcome for a bad candidate, exactly like `validate()`/
    `validate_strategy()`: silence is always safe, a wrong sentence at 3am
    is not, so `agent.main` falls back to silence rather than publishing
    on rejection, and never raises here.

    Checks, in order:

    1. **Non-empty.** An empty or whitespace-only string is not a sentence.
    2. **Exactly one sentence.** Splits on `.`/`!`/`?` and rejects if more
       than one non-empty piece remains, or if none does.
    3. **No forbidden phrasing.** Case-insensitive, whole-word/phrase match
       against "no", "you can't", "you're wrong" (HANDOFF.md rule 3,
       verbatim).
    4. **No question that tests memory** -- *the honest scope of this
       check*: it rejects text that is a question in *form* only, either
       ending in `?` or opening with a small, fixed list of interrogative
       starters ("who", "what", "remember", "do you", ...). This is a
       conservative syntactic filter, not a semantic one: it will reject
       "Do you want tea?" (harmless) exactly as it rejects "Do you
       remember your address?" (a memory test), because this module has no
       way to tell those apart, and correctly rejecting the second matters
       more than wrongly rejecting the first. It will just as certainly
       *miss* a memory-testing question phrased as a statement ("Tell me
       your address.") or one lacking a `?` mark and any listed starter.
       Nothing in this codebase claims to detect memory-testing intent in
       general -- only interrogative form -- and every strategy template
       this issue ships was written not to need this catch at all.
    5. **Minimum silence gap.** `seconds_since_last_say` is `None` before
       any `Say` has been published this session, which always passes this
       check; otherwise it must be at least `min_gap_seconds` (default 8,
       HANDOFF.md rule 3), the minimum silence HANDOFF.md requires after
       one sentence before the next.
    """
    stripped = text.strip()
    if not stripped:
        return RuleResult(accepted=False, reason="say text is empty")

    sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(stripped) if s.strip()]
    if len(sentences) == 0:
        return RuleResult(accepted=False, reason=f"no sentence found in {text!r}")
    if len(sentences) > 1:
        return RuleResult(accepted=False, reason=f"more than one sentence in {text!r}")

    lowered = stripped.lower()
    for phrase in _FORBIDDEN_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return RuleResult(accepted=False, reason=f"forbidden phrase {phrase!r} in {text!r}")

    if stripped.endswith("?") or lowered.startswith(_QUESTION_STARTERS):
        return RuleResult(
            accepted=False,
            reason=f"looks like a question, not allowed by rule 3: {text!r}",
        )

    if seconds_since_last_say is not None and seconds_since_last_say < min_gap_seconds:
        return RuleResult(
            accepted=False,
            reason=(
                f"only {seconds_since_last_say:.1f}s since the last Say, "
                f"minimum is {min_gap_seconds}s"
            ),
        )

    return RuleResult(accepted=True, reason=None)
