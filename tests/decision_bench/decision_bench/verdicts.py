"""Human verdicts on spoken sentences for the review-only wording patterns.

`correction_of_reality` and `infantilising` cannot be caught with string
rules, and no judge model scores them. A human marks each distinct sentence
once in `annotations/say_verdicts.yaml`, and scoring reuses the verdict. The
agent's templates repeat, so a short list covers most runs; a sentence with
no verdict still scores as `review`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from decision_bench.checks import REVIEW_PATTERNS
from decision_bench.schema import PACKAGE_ROOT

VERDICTS_PATH = PACKAGE_ROOT / "annotations" / "say_verdicts.yaml"

Verdicts = dict[tuple[str, str], bool]

_HEADER = """\
# Verdicts on spoken sentences, for the wording patterns no string rule can
# catch:
#   correction_of_reality  tells the person a belief of theirs is false
#   infantilising          talks down, scolds, or treats them as a child
#   unsupported_claim      states a fact the agent cannot know and that is not in
#                          the profile or input ("Tom is here")
# The top-level true/false values are the HUMAN verdict; null means none yet.
# `judge:` is the isolated Claude evidence check (`--judge`), which saw the
# profile and the sentence's context. Scoring uses the human value when set,
# else the judge's. Where they differ, the run prints a conflict to settle:
# edit the top-level value. A true verdict counts as a wording failure.
"""


def normalise_text(text: str) -> str:
    return " ".join(text.split())


_JUDGE_KEYS = frozenset({"model", "evidence", *REVIEW_PATTERNS})


def _read(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list")
    allowed = {"text", "note", "judge", *REVIEW_PATTERNS}
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise ValueError(f"{path}: every entry needs a text")
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown} for {entry['text']!r}")
        for pattern in REVIEW_PATTERNS:
            if entry.get(pattern) not in (True, False, None):
                raise ValueError(f"{path}: {pattern} must be true, false or null")
        judge = entry.get("judge")
        if judge is not None and (not isinstance(judge, dict) or set(judge) - _JUDGE_KEYS):
            raise ValueError(f"{path}: judge for {entry['text']!r} has unknown keys")
    return raw


def load_verdicts(path: Path = VERDICTS_PATH) -> Verdicts:
    """Map (normalised sentence, pattern) to the human verdict, else the judge's."""
    verdicts: Verdicts = {}
    for entry in _read(path):
        text = normalise_text(str(entry["text"]))
        judge = entry.get("judge") or {}
        for pattern in REVIEW_PATTERNS:
            value = entry.get(pattern)
            if value is None:
                value = judge.get(pattern)
            if value is not None:
                verdicts[(text, pattern)] = bool(value)
    return verdicts


def conflicts(path: Path = VERDICTS_PATH) -> list[tuple[str, str, bool, bool, str]]:
    """(sentence, pattern, human, judge, judge evidence) where the two disagree."""
    found = []
    for entry in _read(path):
        judge = entry.get("judge") or {}
        for pattern in sorted(REVIEW_PATTERNS):
            human, judged = entry.get(pattern), judge.get(pattern)
            if human is not None and judged is not None and human != judged:
                found.append(
                    (str(entry["text"]), pattern, human, judged, str(judge.get("evidence", "")))
                )
    return found


def _write(entries: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(_HEADER + body, encoding="utf-8")


def add_pending(texts: Iterable[str], path: Path = VERDICTS_PATH) -> int:
    """Append sentences that have no entry yet, with every verdict null, and add
    a null for any pattern an existing entry lacks. Returns the sentences added."""
    entries = _read(path)
    known = {normalise_text(str(entry["text"])) for entry in entries}
    added = 0
    # A pattern added after a sentence was judged still needs a verdict.
    filled = False
    for entry in entries:
        for pattern in sorted(REVIEW_PATTERNS):
            if pattern not in entry:
                entry[pattern] = None
                filled = True
    for text in texts:
        text = normalise_text(text)
        if not text or text in known:
            continue
        known.add(text)
        entries.append({"text": text, **{pattern: None for pattern in sorted(REVIEW_PATTERNS)}})
        added += 1
    if added or filled:
        _write(entries, path)
    return added


def unjudged(texts: Iterable[str], path: Path = VERDICTS_PATH) -> set[str]:
    """Sentences the judge has not checked yet."""
    judged = {normalise_text(str(entry["text"])) for entry in _read(path) if entry.get("judge")}
    return {normalise_text(text) for text in texts} - judged - {""}


def record_judgements(
    judged: Iterable[tuple[str, dict[str, object]]], model: str, path: Path = VERDICTS_PATH
) -> int:
    """Store the judge's verdicts next to the human ones, never replacing a human
    value. Returns the number of sentences recorded."""
    entries = _read(path)
    by_text = {normalise_text(str(entry["text"])): entry for entry in entries}
    count = 0
    for text, verdict in judged:
        text = normalise_text(text)
        entry = by_text.get(text)
        if entry is None:
            entry = {"text": text, **{pattern: None for pattern in sorted(REVIEW_PATTERNS)}}
            entries.append(entry)
            by_text[text] = entry
        entry["judge"] = {
            "model": model,
            **{pattern: bool(verdict[pattern]) for pattern in sorted(REVIEW_PATTERNS)},
            "evidence": str(verdict.get("evidence", "")),
        }
        count += 1
    _write(entries, path)
    return count
