"""Derive noisy variants of reviewed scenarios.

A noisy variant keeps its parent's checkpoints and labels unchanged
(PLAN.md, "Noisy variants") and perturbs only the inputs, the way a real
night degrades them: perception flickering to a neighbouring state for a
moment, low confidence throughout, speech-to-text typos, or a dropped
utterance. The perturbations are deterministic so a variant is reproducible
and reviewable. Use a kind only where the parent's labels still hold; in
particular, never drop the only utterance that states the person's need.

    python -m decision_bench noise restroom-01 typo
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from decision_bench.schema import SCENARIOS_DIR, load_scenario

KINDS = ("flicker", "low_confidence", "typo", "dropped")

# The state perception is most likely to confuse each state with, briefly.
_NEIGHBOUR = {
    "in_bed": "sitting_up",
    "sitting_up": "in_bed",
    "standing": "walking",
    "walking": "standing",
    "on_floor": "sitting_up",
    "absent": "walking",
}
FLICKER_SECONDS = 2.0
LOW_CONFIDENCE = 0.55
TYPO_CONFIDENCE = 0.6


def _typo(text: str) -> str:
    """Lower-case, no punctuation, and the two middle letters of the longest
    word swapped, as a small speech-to-text slip."""
    words = re.sub(r"[^\w\s']", "", text).lower().split()
    if not words:
        return text
    index = max(range(len(words)), key=lambda i: len(words[i]))
    word = words[index]
    if len(word) >= 4:
        middle = len(word) // 2
        words[index] = word[: middle - 1] + word[middle] + word[middle - 1] + word[middle + 1 :]
    return " ".join(words)


def perturb(timeline: list[dict[str, object]], kind: str) -> list[dict[str, object]]:
    """Return a perturbed copy of a timeline given as plain dicts."""
    if kind not in KINDS:
        raise ValueError(f"unknown noise kind {kind!r}; expected one of {', '.join(KINDS)}")
    events = [dict(event) for event in timeline]
    if kind == "low_confidence":
        for event in events:
            if "person" in event:
                event["person"] = dict(event["person"], confidence=LOW_CONFIDENCE)
        return events
    if kind == "typo":
        for event in events:
            if "utterance" in event:
                utterance = dict(event["utterance"])
                utterance["text"] = _typo(str(utterance["text"]))
                utterance["confidence"] = TYPO_CONFIDENCE
                event["utterance"] = utterance
        return events
    if kind == "dropped":
        # Drop the last utterance: an earlier one has already stated the
        # need, so the parent's labels still hold.
        spoken = [i for i, event in enumerate(events) if "utterance" in event]
        if len(spoken) < 2:
            raise ValueError("'dropped' needs at least two utterances so the need is still stated")
        del events[spoken[-1]]
        return events
    # flicker: after each person reading (except the first), a brief reading
    # of the neighbouring state, then the original state again.
    result: list[dict[str, object]] = []
    for index, event in enumerate(events):
        result.append(event)
        person = event.get("person")
        if not isinstance(person, dict) or index == 0:
            continue
        t = float(event["t"])
        next_t = float(events[index + 1]["t"]) if index + 1 < len(events) else None
        if next_t is not None and next_t - t <= 2 * FLICKER_SECONDS:
            continue
        flicker = dict(person, state=_NEIGHBOUR[str(person["state"])], confidence=0.6)
        result.append({"t": t + FLICKER_SECONDS, "person": flicker})
        result.append({"t": t + 2 * FLICKER_SECONDS, "person": dict(person)})
    return result


def variant_text(parent_path: Path, kind: str) -> tuple[str, str]:
    """(variant id, YAML text) for `kind` applied to the parent fixture. The
    parent's `checkpoints:` block is copied byte for byte."""
    parent = load_scenario(parent_path)
    if parent.noise_of is not None:
        raise ValueError(f"{parent.id} is itself a noisy variant")
    if not parent.labelled:
        raise ValueError(f"{parent.id} is not labelled; noisy variants inherit reviewed labels")
    raw = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    original = parent_path.read_text(encoding="utf-8")
    checkpoints = original[original.index("\ncheckpoints:") + 1 :]
    variant_id = f"{parent.id}-{kind.replace('_', '-')}"
    timeline = perturb(raw["timeline"], kind)
    lines = [
        f"id: {variant_id}",
        f"category: {parent.category}",
        "summary: >-",
        f"  Noisy variant ({kind.replace('_', ' ')}) of {parent.id}: {parent.summary}",
        f'start: "{parent.start.strftime("%H:%M")}"',
    ]
    if parent.profile:
        lines.append("profile: " + yaml.safe_dump(parent.profile, default_flow_style=True).strip())
    if parent.voice_clip:
        lines.append("voice_clip: true")
    lines += [f"noise_of: {parent.id}", "timeline:"]
    for event in timeline:
        event = {"t": _number(event["t"]), **{k: v for k, v in event.items() if k != "t"}}
        dumped = yaml.safe_dump(event, default_flow_style=True, sort_keys=False, width=200)
        lines.append("  - " + dumped.strip())
    return variant_id, "\n".join(lines) + "\n" + checkpoints


def _number(value: object) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a noisy variant of a labelled scenario.")
    parser.add_argument("parent")
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    variant_id, text = variant_text(args.scenarios / f"{args.parent}.yaml", args.kind)
    path = args.scenarios / f"{variant_id}.yaml"
    if path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8")
    load_scenario(path)
    print(path)
    return 0
