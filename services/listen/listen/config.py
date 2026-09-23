"""Environment-backed configuration for the listen service."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ListenConfig:
    """Tuning for VAD, utterance boundaries, and local transcription."""

    model: str = "small.en"
    model_cache: str = "/app/data/models/faster-whisper"
    device: str = "cpu"
    compute_type: str = "int8"
    vad_aggressiveness: int = 2
    end_silence_ms: int = 600
    min_speech_ms: int = 180
    max_utterance_seconds: float = 15.0
    barge_in_ms: int = 600
    barge_in_while_speaking_ms: int = 1500

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ListenConfig:
        """Load configuration from ``env``, using safe local defaults."""
        env = os.environ if env is None else env
        config = cls(
            model=env.get("LISTEN_MODEL", "small.en"),
            model_cache=env.get("LISTEN_MODEL_CACHE", "/app/data/models/faster-whisper"),
            device=env.get("LISTEN_DEVICE", "cpu"),
            compute_type=env.get("LISTEN_COMPUTE_TYPE", "int8"),
            vad_aggressiveness=int(env.get("LISTEN_VAD_AGGRESSIVENESS", "2")),
            end_silence_ms=int(env.get("LISTEN_END_SILENCE_MS", "600")),
            min_speech_ms=int(env.get("LISTEN_MIN_SPEECH_MS", "180")),
            max_utterance_seconds=float(env.get("LISTEN_MAX_UTTERANCE_SECONDS", "15")),
            barge_in_ms=int(env.get("LISTEN_BARGE_IN_MS", "600")),
            barge_in_while_speaking_ms=int(env.get("LISTEN_BARGE_IN_WHILE_SPEAKING_MS", "1500")),
        )
        if not 0 <= config.vad_aggressiveness <= 3:
            raise ValueError("LISTEN_VAD_AGGRESSIVENESS must be between 0 and 3")
        if config.end_silence_ms < 30 or config.min_speech_ms < 30:
            raise ValueError("listen silence and speech durations must be at least 30 ms")
        if config.max_utterance_seconds <= 0:
            raise ValueError("LISTEN_MAX_UTTERANCE_SECONDS must be positive")
        if config.barge_in_ms < 30 or config.barge_in_while_speaking_ms < 30:
            raise ValueError("listen barge-in durations must be at least 30 ms")
        return config
