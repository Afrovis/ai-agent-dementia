"""Caregiver-authored person profile loading (issue #16).

The profile is local configuration, used for strategy-template interpolation
and supplied as structured data to every text-LLM prompt. Missing or invalid
configuration falls back to a generic profile so damaged YAML cannot stop the
night-time agent.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

SERVICE_NAME = "agent"
DEFAULT_PERSON_DIR = Path("config")
DEFAULT_PERSON_FILENAME = "person.yaml"
DEFAULT_PERSON_EXAMPLE_FILENAME = "person.example.yaml"

logger = logging.getLogger(SERVICE_NAME)


@dataclass(frozen=True)
class PersonProfile:
    """The fields PLAN.md section 5.5 requires in every model prompt."""

    name: str = "there"
    preferred_address: str = ""
    caregiver_name: str = "your caregiver"
    caregiver_relationship: str = ""
    night_themes: tuple[str, ...] = ()
    calming_things: tuple[str, ...] = ()
    things_to_avoid: tuple[str, ...] = ()
    physical_notes: tuple[str, ...] = ()
    restroom_location: str = ""

    def prompt_data(self) -> dict[str, object]:
        """Return a JSON-safe copy; prompts never receive the live object."""
        data = asdict(self)
        for field in ("night_themes", "calming_things", "things_to_avoid", "physical_notes"):
            data[field] = list(data[field])
        return data


DEFAULT_PROFILE = PersonProfile()


def _clean_text(value: object, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value.strip()


def _clean_text_list(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def _parse_profile(raw: object) -> PersonProfile:
    if not isinstance(raw, Mapping):
        raise TypeError("profile document must be a mapping")
    # Also accept a `person:` wrapper so a future dashboard can write a
    # self-describing document without breaking the original flat format.
    values: Mapping[str, Any] = raw.get("person", raw)
    if not isinstance(values, Mapping):
        raise TypeError("person must be a mapping")

    caregiver = values.get("caregiver", {})
    if isinstance(caregiver, str):
        caregiver_name, caregiver_relationship = caregiver, ""
    elif isinstance(caregiver, Mapping):
        caregiver_name = caregiver.get("name", "")
        caregiver_relationship = caregiver.get("relationship", "")
    else:
        raise TypeError("caregiver must be a string or mapping")

    name = _clean_text(values.get("name"), field="name") or DEFAULT_PROFILE.name
    preferred_address = _clean_text(values.get("preferred_address"), field="preferred_address")
    caregiver_name = (
        _clean_text(caregiver_name, field="caregiver.name") or DEFAULT_PROFILE.caregiver_name
    )
    return PersonProfile(
        name=name,
        preferred_address=preferred_address,
        caregiver_name=caregiver_name,
        caregiver_relationship=_clean_text(caregiver_relationship, field="caregiver.relationship"),
        night_themes=_clean_text_list(values.get("night_themes"), field="night_themes"),
        calming_things=_clean_text_list(values.get("calming_things"), field="calming_things"),
        things_to_avoid=_clean_text_list(values.get("things_to_avoid"), field="things_to_avoid"),
        physical_notes=_clean_text_list(values.get("physical_notes"), field="physical_notes"),
        restroom_location=_clean_text(values.get("restroom_location"), field="restroom_location"),
    )


def _log_fallback(reason: str, path: Path) -> None:
    logger.warning(
        json.dumps(
            {
                "service": SERVICE_NAME,
                "message": "person profile unavailable, using safe defaults",
                "reason": reason,
                "path": str(path),
            }
        )
    )


def load_profile(
    path: str | Path | None = None, *, env: Mapping[str, str] | None = None
) -> PersonProfile:
    """Load `person.yaml`, with env/example/code fallbacks."""
    env = os.environ if env is None else env
    primary = (
        Path(path)
        if path is not None
        else Path(env.get("PERSON_PATH") or DEFAULT_PERSON_DIR / DEFAULT_PERSON_FILENAME)
    )
    candidate = primary if primary.exists() else primary.parent / DEFAULT_PERSON_EXAMPLE_FILENAME
    if not candidate.exists():
        _log_fallback("no person.yaml or person.example.yaml found", candidate)
        return DEFAULT_PROFILE
    try:
        with candidate.open(encoding="utf-8") as handle:
            return _parse_profile(yaml.safe_load(handle) or {})
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        # Parser/type error strings can echo the offending caregiver-authored
        # line. Log only the category; profile contents are private.
        _log_fallback(f"failed to load profile ({type(exc).__name__})", candidate)
        return DEFAULT_PROFILE
