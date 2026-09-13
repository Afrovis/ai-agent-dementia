"""Local Piper speech synthesis and startup phrase warming (issue #19).

The event bus carries only the text in :class:`nc_shared.events.Say`.  This
module turns that text into a WAV inside the embodiment container, caches it
under an opaque digest, and lets the browser fetch it over the same HTTPS
origin as the face page.  Generated audio never crosses Redis.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import wave
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

logger = logging.getLogger("embodiment")

DEFAULT_VOICE_MODEL = Path("/app/models/piper/en_US-lessac-medium.onnx")
DEFAULT_CACHE_DIR = Path("/tmp/night-companion-tts")
DEFAULT_SPEED = 0.85
DEFAULT_STRATEGY_TEMPLATES = (
    "Hello {name}, it's {time_words}.",
    "You are home in your bedroom, and it is {time_words}.",
    "It's alright{name_vocative}, let's rest now and talk more in the morning.",
    "Let's go back to bed now{name_vocative}.",
    "Someone is coming to help.",
)

_AUDIO_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class PiperSpeech:
    """Synthesize speech with one loaded Piper voice and cache WAV results."""

    def __init__(
        self,
        voice,
        synthesis_config,
        cache_dir: str | Path,
        *,
        cache_namespace: str,
    ) -> None:
        self._voice = voice
        self._synthesis_config = synthesis_config
        self.cache_dir = Path(cache_dir)
        self._cache_namespace = cache_namespace
        self._lock = threading.Lock()

    @classmethod
    def from_model(
        cls,
        model_path: str | Path = DEFAULT_VOICE_MODEL,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        *,
        speed: float = DEFAULT_SPEED,
    ) -> PiperSpeech:
        """Load a Piper voice configured for ``speed`` times normal speech.

        Piper expresses pace as ``length_scale`` (larger means slower), so
        the requested 0.85x night-time voice is represented as ``1 / 0.85``.
        Imports stay here so unit tests can inject a tiny fake voice without
        loading ONNX or model weights.
        """
        if not 0.25 <= speed <= 2.0:
            raise ValueError("PIPER_SPEED must be between 0.25 and 2.0")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Piper voice model not found: {path}")

        from piper import PiperVoice, SynthesisConfig

        voice = PiperVoice.load(path)
        config = SynthesisConfig(length_scale=1.0 / speed)
        stat = path.stat()
        namespace = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}:{speed}"
        return cls(voice, config, cache_dir, cache_namespace=namespace)

    def synthesize(self, text: str) -> str:
        """Return the opaque audio id for ``text``, synthesizing once if needed."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("cannot synthesize empty speech")
        audio_id = hashlib.sha256(f"{self._cache_namespace}\0{clean_text}".encode()).hexdigest()
        destination = self.cache_dir / f"{audio_id}.wav"
        if destination.is_file():
            return audio_id

        # The broadcast loop is normally serial, but the lock also makes
        # startup warming and a coincident Say event safe against each other.
        with self._lock:
            if destination.is_file():
                return audio_id
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_dir / f".{audio_id}.wav.tmp"
            try:
                with wave.open(str(temporary), "wb") as wav_file:
                    self._voice.synthesize_wav(
                        clean_text,
                        wav_file,
                        syn_config=self._synthesis_config,
                    )
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        return audio_id

    def pre_render(self, phrases: Iterable[str]) -> int:
        """Ensure every distinct non-empty phrase has a cached WAV."""
        unique = tuple(dict.fromkeys(phrase.strip() for phrase in phrases if phrase.strip()))
        for phrase in unique:
            self.synthesize(phrase)
        return len(unique)

    def resolve(self, audio_id: str) -> Path | None:
        """Resolve one digest to its cached WAV without allowing path traversal."""
        if not _AUDIO_ID_RE.fullmatch(audio_id):
            return None
        path = self.cache_dir / f"{audio_id}.wav"
        return path if path.is_file() else None


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


def _read_yaml(path: Path, example_name: str) -> Mapping[str, object]:
    candidate = path if path.is_file() else path.parent / example_name
    if not candidate.is_file():
        return {}
    try:
        raw = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return raw if isinstance(raw, Mapping) else {}


def _render_template(template: str, values: Mapping[str, str]) -> str:
    try:
        rendered = template.format_map(_SafeFormatDict(values))
    except ValueError:
        return ""
    return " ".join(rendered.split())


def load_prerender_phrases(
    strategies_path: str | Path,
    person_path: str | Path,
) -> tuple[str, ...]:
    """Expand configured fixed Say templates into phrases to warm at startup.

    The production greeting includes the person's name and one of twelve
    hour phrases.  Expanding each possible hour makes its first use a cache
    hit while leaving LLM-composed responses to be synthesized on demand.
    Invalid/missing caregiver files degrade to the safe built-in templates;
    their contents are never logged.
    """
    profile_doc = _read_yaml(Path(person_path), "person.example.yaml")
    profile = profile_doc.get("person", profile_doc)
    if not isinstance(profile, Mapping):
        profile = {}
    caregiver = profile.get("caregiver", {})
    caregiver_name = (
        caregiver.get("name", "your caregiver") if isinstance(caregiver, Mapping) else caregiver
    )
    name = profile.get("name", "there")
    if not isinstance(name, str) or not name.strip():
        name = "there"
    if not isinstance(caregiver_name, str) or not caregiver_name.strip():
        caregiver_name = "your caregiver"

    strategy_doc = _read_yaml(Path(strategies_path), "strategies.example.yaml")
    configured = strategy_doc.get("strategies", [])
    templates: list[str] = []
    if isinstance(configured, list):
        for strategy in configured:
            if not isinstance(strategy, Mapping) or strategy.get("enabled", True) is False:
                continue
            say = strategy.get("say")
            if isinstance(say, str) and say.strip():
                templates.append(say)
    if not templates:
        templates.extend(DEFAULT_STRATEGY_TEMPLATES)

    base_values = {
        "name": name.strip(),
        "name_vocative": "" if name.strip() == "there" else f", {name.strip()}",
        "caregiver_name": caregiver_name.strip(),
    }
    phrases: list[str] = []
    for template in templates:
        hours = range(1, 13) if "{time_words}" in template else (None,)
        for hour in hours:
            values = dict(base_values)
            values["time_words"] = "" if hour is None else f"{hour} o'clock at night"
            rendered = _render_template(template, values)
            if rendered:
                phrases.append(rendered)
    return tuple(dict.fromkeys(phrases))
