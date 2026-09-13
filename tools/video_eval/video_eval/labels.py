"""Shared, strictly validated output contract for reference labellers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

POSTURES = frozenset({"in_bed", "sitting_up", "upright", "on_floor", "absent"})
LOCATIONS = frozenset({"bed", "door", "bathroom_path", "other"})

LABEL_PROMPT = """You label bedroom monitoring frames for a fall and wandering safety system.
Answer with JSON only:
{"person_visible": true|false,
 "posture": "in_bed"|"sitting_up"|"upright"|"on_floor"|"absent",
 "location": "bed"|"door"|"bathroom_path"|"other",
 "confidence": 0.0-1.0,
 "note": "at most 15 words"}
in_bed = lying on the bed, even under a blanket. sitting_up = torso upright
while on the bed or its edge. upright = standing or walking anywhere.
on_floor = lying, kneeling or crawling on the floor. absent = no person.
Do not describe identity, clothing or the room."""


@dataclass(frozen=True)
class Label:
    person_visible: bool
    posture: str
    location: str
    confidence: float
    note: str


def validate_label(raw: Any) -> Label:
    """Validate a model response without silently coercing invalid fields."""
    if not isinstance(raw, dict):
        raise ValueError("label must be a JSON object")
    if type(raw.get("person_visible")) is not bool:
        raise ValueError("person_visible must be a boolean")
    posture = raw.get("posture")
    if posture not in POSTURES:
        raise ValueError(f"invalid posture: {posture!r}")
    location = raw.get("location")
    if location not in LOCATIONS:
        raise ValueError(f"invalid location: {location!r}")
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    note = raw.get("note")
    if not isinstance(note, str) or len(note.split()) > 15:
        raise ValueError("note must be a string of at most 15 words")
    return Label(raw["person_visible"], posture, location, float(confidence), note)


def label_record(label: Label, *, frame_index: int, t_s: float, labeller: str) -> dict[str, Any]:
    return {
        "confidence": label.confidence,
        "frame_index": frame_index,
        "labeller": labeller,
        "location": label.location,
        "note": label.note,
        "person_visible": label.person_visible,
        "posture": label.posture,
        "t_s": t_s,
    }


def failed_label_record(
    *, frame_index: int, t_s: float, labeller: str, reason: str
) -> dict[str, Any]:
    """Represent an exhausted local retry without inventing a model label."""
    return {
        "confidence": 0.0,
        "frame_index": frame_index,
        "labeller": labeller,
        "location": None,
        "note": reason[:120],
        "person_visible": None,
        "posture": None,
        "t_s": t_s,
    }
