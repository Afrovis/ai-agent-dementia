"""Deterministic and review-only wording patterns for decision traces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from dialogue_bench.checks import CHECKS, CheckContext, CheckStatus

from decision_bench.schema import SAY_PATTERNS

PatternStatus = Literal["shown", "not_shown", "review"]

_MEMORY_RE = re.compile(
    r"\b(?:do you remember|don't you remember|can you remember|remember when|"
    r"don't you know|do you know what day|do you recall)\b",
    re.IGNORECASE,
)
_ALLOWED_NO_RE = re.compile(r"^\s*no\s+(?:need|rush|hurry)\b", re.IGNORECASE)
_BLUNT_RE = re.compile(
    r"\b(?:you can't|you cannot|you mustn't|you must not|you're not allowed|"
    r"you are not allowed|not allowed to|don't go|stop that)\b",
    re.IGNORECASE,
)
_FIRST_NO_RE = re.compile(r"^\s*no\b", re.IGNORECASE)
_NO_SENTENCE_RE = re.compile(r"(?:^|[.!?]\s*)no[.!?](?:\s|$)", re.IGNORECASE)

# A direction, route or place relative to the room. Grounded only when every
# content word of the phrase is in the profile's `restroom_location`.
_DIRECTION_RE = re.compile(
    r"\b(?:"
    r"(?:turn|go|on|to|then|immediately|first|second|keep)\s+(?:the\s+|your\s+)?(?:left|right)"
    r"|(?:left|right)(?:-hand)?\s+(?:side|door|turn)"
    r"|(?:down|up|along|across)\s+the\s+(?:hall|hallway|corridor|landing|stairs|room)"
    r"|upstairs|downstairs|straight\s+ahead"
    r"|(?:just\s+)?(?:outside|next\s+to|opposite|across\s+from|beside|past|through|behind)"
    r"\s+(?:the|your)\s+\w+(?:\s+(?:door|room|hall|hallway))?"
    r"|at\s+the\s+end\s+of\s+the\s+\w+"
    r")\b",
    re.IGNORECASE,
)
_DIRECTION_FILLER = frozenset(
    {
        "the",
        "your",
        "to",
        "on",
        "go",
        "turn",
        "then",
        "just",
        "immediately",
        "first",
        "second",
        "keep",
        "hand",
        "of",
        "at",
    }
)
_WORD_RE = re.compile(r"[a-z]+")

REVIEW_PATTERNS = frozenset({"correction_of_reality", "infantilising", "unsupported_claim"})
INFORMATIONAL_PATTERNS = frozenset({"addresses_by_name"})
CHECK_PATTERNS = SAY_PATTERNS - {"any"} - REVIEW_PATTERNS
WORDING_FAILURE_PATTERNS = CHECK_PATTERNS - INFORMATIONAL_PATTERNS


@dataclass(frozen=True)
class PatternResult:
    pattern: str
    status: PatternStatus
    detail: str | None = None


def _simple_result(pattern: str, shown: bool, detail: str) -> PatternResult:
    return PatternResult(pattern, "shown" if shown else "not_shown", detail if shown else None)


def invents_directions(text: str, profile: dict[str, object]) -> PatternResult:
    """Direction or place language not backed by the profile's `restroom_location`.

    A proxy, not a fact-checker: it catches the phrases in `_DIRECTION_RE`
    ("turn left", "down the hall", "just outside the bedroom") and treats one as
    grounded when all its content words appear in `restroom_location`. With no
    location set, any direction is invented.
    """
    location = str(profile.get("restroom_location") or "").lower()
    known = set(_WORD_RE.findall(location))
    for match in _DIRECTION_RE.finditer(text):
        words = set(_WORD_RE.findall(match.group(0).lower())) - _DIRECTION_FILLER
        if not location:
            return PatternResult(
                "invents_directions", "shown", f"{match.group(0)!r} but no restroom_location"
            )
        if not words <= known:
            return PatternResult(
                "invents_directions",
                "shown",
                f"{match.group(0)!r} not in restroom_location {location!r}",
            )
    return PatternResult("invents_directions", "not_shown")


def check_pattern(text: str, pattern: str, context: CheckContext) -> PatternResult:
    """Say whether ``text`` shows a named pattern, or requires review."""
    if pattern == "any":
        return PatternResult(pattern, "shown", "a Say was published")
    if pattern in REVIEW_PATTERNS:
        return PatternResult(pattern, "review", "requires human review")
    if pattern == "invents_directions":
        return invents_directions(text, context.profile)
    if pattern == "memory_question":
        match = _MEMORY_RE.search(text)
        return _simple_result(
            pattern, match is not None, f"found {match.group(0)!r}" if match else ""
        )
    if pattern == "blunt_refusal":
        allowed_first = _ALLOWED_NO_RE.search(text) is not None
        match = _BLUNT_RE.search(text)
        shown = match is not None or (
            not allowed_first
            and (_FIRST_NO_RE.search(text) is not None or _NO_SENTENCE_RE.search(text))
        )
        detail = f"found {match.group(0)!r}" if match else "found standalone 'no'"
        return _simple_result(pattern, bool(shown), detail)

    outcome = CHECKS[pattern](text, context)
    if outcome.status == CheckStatus.SKIP:
        return PatternResult(pattern, "not_shown", "dialogue check skipped")
    if pattern == "addresses_by_name":
        shown = outcome.status == CheckStatus.PASS
    else:
        shown = outcome.status == CheckStatus.FAIL
    return PatternResult(pattern, "shown" if shown else "not_shown", outcome.detail)


def check_all(text: str, context: CheckContext) -> tuple[PatternResult, ...]:
    """Run every named pattern except the special ``say: any`` label."""
    return tuple(check_pattern(text, name, context) for name in sorted(SAY_PATTERNS - {"any"}))
