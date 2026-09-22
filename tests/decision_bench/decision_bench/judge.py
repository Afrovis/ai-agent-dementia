"""Evidence check on spoken sentences by an isolated Claude judge.

The judge decides the review-only wording patterns (`unsupported_claim`,
`correction_of_reality`, `infantilising`) for each distinct sentence, from the
profile and that sentence's context only, never the agent's code. It runs
through the Claude Code CLI on the claude.ai subscription, like the
annotator. Its verdicts are stored under `judge:` in
`annotations/say_verdicts.yaml`, next to the human's; a human verdict wins and
a disagreement is reported rather than overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import yaml
from agent.strategies import time_as_words

from decision_bench.annotate import AnnotatorError, Runner, run_claude
from decision_bench.checks import REVIEW_PATTERNS
from decision_bench.runner import Trace
from decision_bench.verdicts import normalise_text

JUDGE_PROMPT_PATH = Path(__file__).with_name("judge_prompt.md")
BATCH_SIZE = 30


@dataclass(frozen=True)
class Sentence:
    """One spoken sentence and what the agent knew when it said it."""

    text: str
    profile: dict[str, object]
    context: dict[str, object] = field(default_factory=dict)


def sentences_from_trace(trace: Trace, profile: dict[str, object]) -> list[Sentence]:
    sentences = []
    utterance = None
    reading = None
    notified = False
    for entry in trace.entries:
        if entry.kind == "Utterance" and entry.data.get("input"):
            utterance = entry.data.get("text")
        elif entry.kind == "PersonState" and entry.data.get("input"):
            reading = entry.data
        elif entry.kind == "Notify":
            notified = True
        elif entry.kind == "Say":
            context: dict[str, object] = {
                "time_words": time_as_words(trace.start + timedelta(seconds=entry.t)),
                "person_last_said": utterance,
                "camera": (
                    {key: reading.get(key) for key in ("state", "zone", "scene_note")}
                    if reading
                    else None
                ),
                "notified": notified,
            }
            text = normalise_text(str(entry.data.get("text", "")))
            sentences.append(Sentence(text, profile, context))
    return sentences


def judge_input(sentences: list[Sentence]) -> str:
    profiles: list[dict[str, object]] = []
    items = []
    for index, sentence in enumerate(sentences):
        if sentence.profile not in profiles:
            profiles.append(sentence.profile)
        items.append(
            {
                "index": index,
                "sentence": sentence.text,
                "profile": f"P{profiles.index(sentence.profile) + 1}",
                "context": sentence.context,
            }
        )
    parts = ["## Profiles", ""]
    for number, profile in enumerate(profiles, start=1):
        parts += [f"### P{number}", "", yaml.safe_dump(profile, sort_keys=False).strip(), ""]
    parts += ["## Sentences", "", yaml.safe_dump(items, sort_keys=False, width=100).strip(), ""]
    parts.append(f"Give a verdict for every index from 0 to {len(sentences) - 1}.")
    return "\n".join(parts)


def output_schema(count: int) -> dict[str, object]:
    verdict = {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "minimum": 0, "maximum": count - 1},
            **{pattern: {"type": "boolean"} for pattern in sorted(REVIEW_PATTERNS)},
            "evidence": {"type": "string"},
        },
        "required": ["index", *sorted(REVIEW_PATTERNS), "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": verdict}},
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def _problems(data: dict[str, object], count: int) -> list[str]:
    verdicts = data.get("verdicts")
    if not isinstance(verdicts, list):
        return ["verdicts must be a list"]
    indices = [item.get("index") for item in verdicts if isinstance(item, dict)]
    missing = sorted(set(range(count)) - set(indices))
    duplicates = sorted({i for i in indices if indices.count(i) > 1})
    problems = []
    if missing:
        problems.append(f"missing indices: {missing}")
    if duplicates:
        problems.append(f"duplicate indices: {duplicates}")
    return problems


def judge_sentences(
    sentences: list[Sentence],
    *,
    runner: Runner = run_claude,
    model: str = "opus",
    effort: str = "medium",
) -> tuple[list[dict[str, object]], str]:
    """Judge `sentences` in batches. Returns one verdict per sentence, in order,
    and the model id that answered."""
    results: list[dict[str, object]] = []
    used_model = model
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start : start + BATCH_SIZE]
        prompt = judge_input(batch)
        schema = output_schema(len(batch))
        problems: list[str] = []
        for _attempt in range(2):
            text = prompt
            if problems:
                text += "\n\n## Your previous answer was rejected\n\n" + "\n".join(
                    f"- {problem}" for problem in problems
                )
            result = runner(JUDGE_PROMPT_PATH, text, schema, model, effort)
            data = result["structured_output"]
            assert isinstance(data, dict)
            problems = _problems(data, len(batch))
            if not problems:
                break
        if problems:
            raise AnnotatorError("judge output rejected: " + "; ".join(problems))
        used_model = str(result.get("model", model))
        by_index = {item["index"]: item for item in data["verdicts"]}
        results.extend(by_index[index] for index in range(len(batch)))
    return results, used_model


def first_occurrences(sentences: list[Sentence], wanted: set[str]) -> list[Sentence]:
    """The first occurrence of each sentence text in `wanted`, in trace order."""
    seen: set[str] = set()
    picked = []
    for sentence in sentences:
        if sentence.text in wanted and sentence.text not in seen:
            seen.add(sentence.text)
            picked.append(sentence)
    return picked
