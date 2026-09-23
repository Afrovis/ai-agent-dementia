"""Validated, atomic caregiver settings and media persistence.

The dashboard is the only service allowed to write ``config/``.  This module
keeps that boundary small and makes every browser supplied value pass through
validation before replacing a file the night-time services will read.
"""

from __future__ import annotations

import io
import math
import os
import re
import tempfile
import wave
from collections.abc import Mapping
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

PROFILE_LIST_FIELDS = ("night_themes", "calming_things", "things_to_avoid", "physical_notes")
PROFILE_TEXT_FIELDS = ("name", "preferred_address", "restroom_location")
CAREGIVER_FIELDS = ("name", "relationship")

FACES = frozenset({"asleep", "awake", "speaking", "listening"})
KNOWN_STRATEGY_IDS = frozenset(
    {
        "ambient_orient",
        "soft_greeting",
        "orient_time_place",
        "validate_and_redirect",
        "guided_return",
        "familiar_voice",
        "path_light",
        "acknowledge_return",
        "reassure_waiting",
        "acknowledge_pain",
        "comfort_pain",
        "acknowledge_progress",
        "escalate_phone",
    }
)
FORBIDDEN_PHRASES = ("no", "you can't", "you're wrong")
QUESTION_STARTERS = (
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
MEDIA_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_VOICE_BYTES = 25 * 1024 * 1024


def _atomic_yaml(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _load_yaml(primary: Path, example_name: str) -> dict:
    candidate = primary if primary.exists() else primary.parent / example_name
    if not candidate.exists():
        return {}
    try:
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_profile_document(path: str | Path) -> dict:
    raw = _load_yaml(Path(path), "person.example.yaml")
    person = raw.get("person", raw)
    if not isinstance(person, Mapping):
        return {}
    caregiver = person.get("caregiver", {})
    if isinstance(caregiver, str):
        caregiver = {"name": caregiver, "relationship": ""}
    elif not isinstance(caregiver, Mapping):
        caregiver = {}
    result = {field: str(person.get(field) or "").strip() for field in PROFILE_TEXT_FIELDS}
    result["enable_cloud_fallback"] = person.get("enable_cloud_fallback") is True
    result["caregiver"] = {
        field: str(caregiver.get(field) or "").strip() for field in CAREGIVER_FIELDS
    }
    for field in PROFILE_LIST_FIELDS:
        value = person.get(field, [])
        result[field] = (
            [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, list)
            else []
        )
    return result


def save_profile_document(path: str | Path, values: Mapping[str, object]) -> Path:
    def text(field: str) -> str:
        value = str(values.get(field, "")).strip()
        if len(value) > 500:
            raise ValueError(f"{field.replace('_', ' ')} is too long")
        return value

    name = text("name")
    if not name:
        raise ValueError("name is required")
    payload: dict[str, object] = {
        "name": name,
        "preferred_address": text("preferred_address"),
        "caregiver": {
            "name": text("caregiver_name"),
            "relationship": text("caregiver_relationship"),
        },
    }
    for field in PROFILE_LIST_FIELDS:
        lines = [line.strip() for line in str(values.get(field, "")).splitlines() if line.strip()]
        if len(lines) > 50 or any(len(line) > 500 for line in lines):
            raise ValueError(f"{field.replace('_', ' ')} has too much text")
        payload[field] = lines
    payload["restroom_location"] = text("restroom_location")
    payload["enable_cloud_fallback"] = "enable_cloud_fallback" in values
    return _atomic_yaml(Path(path), payload)


def load_strategy_document(path: str | Path) -> list[dict]:
    raw = _load_yaml(Path(path), "strategies.example.yaml")
    entries = raw.get("strategies", [])
    if not isinstance(entries, list):
        return []
    return [
        dict(entry)
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("id") in KNOWN_STRATEGY_IDS
    ]


def _valid_template(value: str) -> None:
    try:
        value.format_map(_MissingFields())
    except (ValueError, IndexError, KeyError) as exc:
        raise ValueError(f"invalid template: {exc}") from exc


class _MissingFields(dict):
    def __missing__(self, key: str) -> str:
        return "value"


def _validate_say(value: str) -> None:
    rendered = value.format_map(_MissingFields()).strip()
    pieces = [part.strip() for part in re.split(r"[.!?]+", rendered) if part.strip()]
    if len(pieces) != 1:
        raise ValueError("spoken text must be exactly one sentence")
    lowered = rendered.casefold()
    for phrase in FORBIDDEN_PHRASES:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered):
            raise ValueError(f'spoken text cannot contain "{phrase}"')
    if rendered.endswith("?") or any(
        lowered == starter or lowered.startswith(f"{starter} ") for starter in QUESTION_STARTERS
    ):
        raise ValueError("spoken text cannot be a question")


def save_strategy_document(
    path: str | Path, current: list[dict], submitted: Mapping[str, object]
) -> Path:
    updated: list[dict] = []
    orders: set[int] = set()
    for existing in current:
        strategy_id = str(existing["id"])
        prefix = f"{strategy_id}__"
        try:
            order = int(str(submitted.get(prefix + "order", "")))
            dwell = float(str(submitted.get(prefix + "dwell_seconds", "")))
            cooldown = float(str(submitted.get(prefix + "cooldown_seconds", "")))
            intrusiveness = int(str(submitted.get(prefix + "intrusiveness", "")))
            brightness = float(str(submitted.get(prefix + "brightness", "")))
        except ValueError as exc:
            raise ValueError(f"{strategy_id} has an invalid number") from exc
        if order < 1 or order > 100 or order in orders:
            raise ValueError("strategy order values must be unique numbers from 1 to 100")
        orders.add(order)
        dwell_is_allowed_infinity = strategy_id == "escalate_phone" and dwell == math.inf
        if (not math.isfinite(dwell) and not dwell_is_allowed_infinity) or dwell < 8:
            raise ValueError(f"{strategy_id} has an invalid dwell")
        if not math.isfinite(cooldown) or cooldown < 0:
            raise ValueError(f"{strategy_id} cooldown must be a non-negative finite number")
        if not math.isfinite(brightness) or not 0 <= brightness <= 1:
            raise ValueError(f"{strategy_id} brightness is out of range")
        if not 1 <= intrusiveness <= 5:
            raise ValueError(f"{strategy_id} intrusiveness or brightness is out of range")
        face = str(submitted.get(prefix + "face", ""))
        if face not in FACES:
            raise ValueError(f"{strategy_id} has an invalid face")
        entry: dict[str, object] = {
            "id": strategy_id,
            "order": order,
            "enabled": prefix + "enabled" in submitted,
            "dwell_seconds": dwell,
            "cooldown_seconds": cooldown,
            "intrusiveness": intrusiveness,
            "face": face,
            "brightness": brightness,
        }
        for field in ("headline", "body"):
            value = str(submitted.get(prefix + field, "")).strip()
            if len(value) > 500:
                raise ValueError(f"{strategy_id} {field} is too long")
            _valid_template(value)
            entry[field] = value
        say = str(submitted.get(prefix + "say", "")).strip()
        if say:
            if len(say) > 500:
                raise ValueError(f"{strategy_id} spoken text is too long")
            _valid_template(say)
            _validate_say(say)
            entry["say"] = say
        else:
            entry["say"] = None
        photo_id = str(submitted.get(prefix + "photo_id", "")).strip()
        if photo_id and not MEDIA_ID_RE.fullmatch(photo_id):
            raise ValueError(f"{strategy_id} photo id is invalid")
        entry["photo_id"] = photo_id or None
        if strategy_id == "familiar_voice":
            clip_id = str(submitted.get(prefix + "clip_id", "")).strip()
            if clip_id and not MEDIA_ID_RE.fullmatch(clip_id):
                raise ValueError(f"{strategy_id} clip id is invalid")
            entry["clip_id"] = clip_id or None
        updated.append(entry)
    return _atomic_yaml(
        Path(path), {"strategies": sorted(updated, key=lambda item: int(item["order"]))}
    )


def _media_id(filename: str) -> str:
    stem = Path(filename).stem.casefold()
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-_")[:48]
    return cleaned or "upload"


def _available_path(directory: Path, media_id: str, extension: str) -> tuple[str, Path]:
    candidate_id = media_id
    counter = 2
    while any(directory.glob(f"{candidate_id}.*")):
        candidate_id = f"{media_id[:56]}-{counter}"
        counter += 1
    return candidate_id, directory / f"{candidate_id}{extension}"


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_photo(directory: str | Path, filename: str, content: bytes) -> tuple[str, Path]:
    if not content or len(content) > MAX_PHOTO_BYTES:
        raise ValueError("photo must be between 1 byte and 10 MB")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
            image_format = image.format
            width, height = image.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("file is not a valid supported photo") from exc
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format or "")
    if extension is None:
        raise ValueError("photo must be JPEG, PNG, or WebP")
    if width > 10_000 or height > 10_000 or width * height > 25_000_000:
        raise ValueError("photo dimensions are too large")
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    media_id, target = _available_path(target_dir, _media_id(filename), extension)
    _atomic_bytes(target, content)
    return media_id, target


def save_voice_clip(directory: str | Path, filename: str, content: bytes) -> tuple[str, Path]:
    if not content or len(content) > MAX_VOICE_BYTES:
        raise ValueError("voice clip must be between 1 byte and 25 MB")
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise ValueError("voice clip must contain uncompressed PCM audio")
            if audio.getnchannels() not in (1, 2) or audio.getsampwidth() not in (1, 2, 3, 4):
                raise ValueError("voice clip has an unsupported PCM layout")
            if audio.getframerate() < 8_000 or audio.getframerate() > 96_000:
                raise ValueError("voice clip has an unsupported sample rate")
            duration = audio.getnframes() / audio.getframerate()
            if duration <= 0 or duration > 180:
                raise ValueError("voice clip must be no longer than 3 minutes")
            expected_bytes = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
            if len(audio.readframes(audio.getnframes())) != expected_bytes:
                raise ValueError("voice clip audio data is incomplete")
    except (wave.Error, EOFError) as exc:
        raise ValueError("voice clip must be a valid uncompressed WAV file") from exc
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    media_id, target = _available_path(target_dir, _media_id(filename), ".wav")
    _atomic_bytes(target, content)
    return media_id, target


def list_media(directory: str | Path, extensions: frozenset[str]) -> list[dict[str, object]]:
    root = Path(directory)
    if not root.exists():
        return []
    return [
        {"id": item.stem, "filename": item.name, "bytes": item.stat().st_size}
        for item in sorted(root.iterdir())
        if item.is_file() and item.suffix.casefold() in extensions and not item.name.startswith(".")
    ]
