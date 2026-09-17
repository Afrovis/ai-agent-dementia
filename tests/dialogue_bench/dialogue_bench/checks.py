"""Deterministic assertions for the compose-prompt rules the bench never measured.

`services/agent/agent/llm.py`'s `_COMPOSE_TASK` instructs the model to avoid "but",
respect `profile.things_to_avoid`, only state the time when the utterance is about it,
invent nothing, and address the person by `profile.preferred_address`. None of that is
scored by `agent.rules.validate_say`, which only checks non-empty, one sentence,
forbidden phrases, question form, and the silence gap. These five checks are pure string
work over the composed text -- no LLM judging -- so each one is honest, in its own
docstring, about exactly what it does and does not catch.

This module must not import `dialogue_bench.scenarios`: scenarios.py imports this module
to validate check names, and the dependency is one-way.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

# Common capitalised non-names: I, I'm, I'll, I've, It's, Let's, Okay, OK, Yes, We'll,
# We're, You're, There's, That's, Here's, Your, You. Compared against the leading piece
# after splitting the token on apostrophes (so "It's" -> "it", "We're" -> "we").
_ALLOWED_CAPITALISED_WORDS = frozenset(
    {"i", "it", "let", "okay", "ok", "yes", "we", "you", "there", "that", "here", "your"}
)

_BUT_RE = re.compile(r"\bbut\b", re.IGNORECASE)

_CLOCK_TIME_RE = re.compile(
    r"\bo'?clock\b"
    r"|\b\d{1,2}[:.]\d{2}\b"
    r"|\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b"
    r"|\bmidnight\b"
    r"|\bnoon\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z']+")


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # not applicable to this scenario


@dataclass(frozen=True)
class CheckContext:
    """The primitives a check needs, gathered from a scenario without importing it."""

    profile: Mapping[str, object]
    utterance: str | None
    time_words: str
    scene_note: str | None
    caregiver_phrase_template: str
    avoid_terms: tuple[str, ...]


@dataclass(frozen=True)
class CheckOutcome:
    name: str
    status: CheckStatus
    detail: str | None  # human-readable reason when FAIL


CheckFn = Callable[[str, CheckContext], CheckOutcome]


def _outcome(name: str, status: CheckStatus, detail: str | None = None) -> CheckOutcome:
    return CheckOutcome(name=name, status=status, detail=detail)


def conjunction_but(text: str, ctx: CheckContext) -> CheckOutcome:
    """FAIL if the composed text uses the word "but", which `_COMPOSE_TASK` forbids
    because it cancels the acknowledgement that came before it ("I hear you, but...").

    Matches "but" as a whole word only, case-insensitively, so "button", "butter" and
    "rebut" do not trip it. Never SKIPs: this rule applies to every scenario.
    """
    match = _BUT_RE.search(text)
    if match:
        return _outcome(
            "conjunction_but", CheckStatus.FAIL, f"found {match.group(0)!r} in {text!r}"
        )
    return _outcome("conjunction_but", CheckStatus.PASS)


def avoid_terms(text: str, ctx: CheckContext) -> CheckOutcome:
    """FAIL if the text mentions a term from `ctx.avoid_terms` (drawn from
    `profile.things_to_avoid`, e.g. "hospital"). SKIP when the scenario configures no
    avoid terms at all -- there is nothing to check.

    Each term is matched as a whole word/phrase, case-insensitively; a multi-word term
    is escaped and wrapped in `\\b...\\b` the same way.
    """
    if not ctx.avoid_terms:
        return _outcome("avoid_terms", CheckStatus.SKIP)
    for term in ctx.avoid_terms:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            return _outcome("avoid_terms", CheckStatus.FAIL, f"mentioned avoided term {term!r}")
    return _outcome("avoid_terms", CheckStatus.PASS)


def states_clock_time(text: str, ctx: CheckContext) -> CheckOutcome:
    """FAIL only on an explicit clock reading: "o'clock"/"oclock", `H:MM`/`H.MM`,
    "<hour> am/pm", or the whole words "midnight"/"noon".

    Deliberately narrow scope: this does NOT match vague day-parts like "morning",
    "night" or "evening" -- the shipped caregiver template says "talk more in the
    morning" and `_COMPOSE_TASK` allows that; it only forbids stating the time when the
    latest utterance is not about it. This check cannot tell whether a clock reading was
    warranted by the utterance, so pair it with a narrowed `must_not` list on scenarios
    (e.g. `confused_time`) where stating the time is legitimate. It never SKIPs.
    """
    match = _CLOCK_TIME_RE.search(text)
    if match:
        return _outcome(
            "states_clock_time",
            CheckStatus.FAIL,
            f"found clock time {match.group(0)!r} in {text!r}",
        )
    return _outcome("states_clock_time", CheckStatus.PASS)


def _split_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _vocabulary_from(value: object, into: set[str]) -> None:
    if isinstance(value, str):
        for word in _split_words(value):
            for piece in word.split("'"):
                if piece:
                    into.add(piece.lower())
    elif isinstance(value, list):
        for item in value:
            _vocabulary_from(item, into)
    elif isinstance(value, Mapping):
        for item in value.values():
            _vocabulary_from(item, into)


def invents_proper_noun(text: str, ctx: CheckContext) -> CheckOutcome:
    """A deliberately approximate proxy for `_COMPOSE_TASK`'s "only state facts found in
    the input: never invent people, places, times or plans." FAIL if the text contains a
    capitalised token that does not appear, case-insensitively, anywhere in the input the
    model was given (every string in `ctx.profile`, recursing into lists, plus
    `utterance`, `scene_note`, `time_words` and `caregiver_phrase_template`).

    This is a proxy, not a fact-checker: it MISSES an invented lowercase fact or an
    invented plan entirely, and it can FALSE-POSITIVE on an unusual capitalisation the
    model happens to use. The first token of the text is skipped because grammar
    capitalises it regardless of content (the output is one sentence), and a small
    allowlist of common capitalised non-names ("I", "I'm", "Okay", "You're", ...) is
    skipped too. Possessives and contractions are compared by splitting on apostrophes.
    Never SKIPs.
    """
    vocabulary: set[str] = set()
    _vocabulary_from(ctx.profile, vocabulary)
    for extra in (ctx.utterance, ctx.scene_note, ctx.time_words, ctx.caregiver_phrase_template):
        if extra is not None:
            _vocabulary_from(extra, vocabulary)

    words = _split_words(text)
    capitalised_positions = [
        index
        for index, word in enumerate(words)
        if word and word[0].isupper() and any(char.isalpha() for char in word)
    ]
    # Skip the first word of the text: grammar capitalises it regardless of content.
    for index in capitalised_positions:
        if index == 0:
            continue
        word = words[index]
        pieces = [piece.lower() for piece in word.split("'") if piece]
        if not pieces:
            continue
        if pieces[0] in _ALLOWED_CAPITALISED_WORDS:
            continue
        if any(piece in vocabulary for piece in pieces):
            continue
        return _outcome(
            "invents_proper_noun",
            CheckStatus.FAIL,
            f"unrecognised capitalised word {word!r} in {text!r}",
        )
    return _outcome("invents_proper_noun", CheckStatus.PASS)


def addresses_by_name(text: str, ctx: CheckContext) -> CheckOutcome:
    """PASS if the text addresses the person by `profile.preferred_address` (falling
    back to `profile.name`), matched as a whole word, case-insensitively. SKIP when the
    profile sets neither, since there is then no name to require.
    """
    name = ctx.profile.get("preferred_address") or ctx.profile.get("name")
    if not name:
        return _outcome("addresses_by_name", CheckStatus.SKIP)
    name = str(name)
    if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
        return _outcome("addresses_by_name", CheckStatus.PASS)
    return _outcome("addresses_by_name", CheckStatus.FAIL, f"{name!r} not found in {text!r}")


CHECKS: Mapping[str, CheckFn] = {
    "conjunction_but": conjunction_but,
    "avoid_terms": avoid_terms,
    "states_clock_time": states_clock_time,
    "invents_proper_noun": invents_proper_noun,
    "addresses_by_name": addresses_by_name,
}

CHECK_NAMES: frozenset[str] = frozenset(CHECKS)


def run_checks(text: str, names: Sequence[str], ctx: CheckContext) -> tuple[CheckOutcome, ...]:
    return tuple(CHECKS[name](text, ctx) for name in names)
