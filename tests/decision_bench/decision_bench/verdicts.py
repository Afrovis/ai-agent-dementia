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
# Human verdicts on spoken sentences, for the wording patterns that only a
# human can judge. For each sentence, set every pattern to true (the sentence
# shows it) or false (it does not). Leave null to keep it as `review`.
#   correction_of_reality  tells the person a belief of theirs is false
#   infantilising          talks down, scolds, or treats them as a child
#   unsupported_claim      states a fact the agent cannot know and that is not in
#                          the profile or input ("Tom is here")
# A true verdict also counts as a wording failure on every run.
# `python -m decision_bench ... --collect-verdicts` appends new sentences.
"""


def normalise_text(text: str) -> str:
    return " ".join(text.split())


def _read(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list")
    allowed = {"text", "note", *REVIEW_PATTERNS}
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str):
            raise ValueError(f"{path}: every entry needs a text")
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown} for {entry['text']!r}")
        for pattern in REVIEW_PATTERNS:
            if entry.get(pattern) not in (True, False, None):
                raise ValueError(f"{path}: {pattern} must be true, false or null")
    return raw


def load_verdicts(path: Path = VERDICTS_PATH) -> Verdicts:
    """Map (normalised sentence, pattern) to the human's verdict."""
    verdicts: Verdicts = {}
    for entry in _read(path):
        text = normalise_text(str(entry["text"]))
        for pattern in REVIEW_PATTERNS:
            if entry.get(pattern) is not None:
                verdicts[(text, pattern)] = bool(entry[pattern])
    return verdicts


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
        path.parent.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=100)
        path.write_text(_HEADER + body, encoding="utf-8")
    return added
