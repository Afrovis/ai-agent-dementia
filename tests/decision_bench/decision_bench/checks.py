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

REVIEW_PATTERNS = frozenset({"correction_of_reality", "infantilising"})
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


def check_pattern(text: str, pattern: str, context: CheckContext) -> PatternResult:
    """Say whether ``text`` shows a named pattern, or requires review."""
    if pattern == "any":
        return PatternResult(pattern, "shown", "a Say was published")
    if pattern in REVIEW_PATTERNS:
        return PatternResult(pattern, "review", "requires human review")
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
