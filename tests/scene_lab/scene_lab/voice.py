"""Synthetic person speech and continuous microphone room tone (memory only)."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
# script.js: 4096 native-rate frames per callback. At 48 kHz this yields
# floor(4096 / 3) = 1365 PCM16 samples, about 85 ms per /media message.
CHUNK_SAMPLES = 1365
DEFAULT_PIPER_DIR = Path("/Users/mathiasserver/Documents/data-ai-agent-dementia/models/piper-sim")
ALLOWED_VOICES = {"en_US-amy-medium", "en_US-ryan-medium"}


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or len(samples) == 0:
        return samples.copy()
    size = max(1, round(len(samples) * target_rate / source_rate))
    return np.interp(
        np.arange(size) * source_rate / target_rate,
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)


class Synthesizer(Protocol):
    def __call__(self, text: str, *, length_scale: float) -> tuple[np.ndarray, int]: ...


class Voice:
    def __init__(
        self,
        voice_name: str,
        sample_rate: int = SAMPLE_RATE,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        if voice_name not in ALLOWED_VOICES:
            raise ValueError("simulated person must use amy or ryan, never the agent voice")
        self.voice_name = voice_name
        self.sample_rate = sample_rate
        self._synthesizer = synthesizer
        self._voice = None

    def _render(self, text: str, length_scale: float) -> tuple[np.ndarray, int]:
        if self._synthesizer is not None:
            return self._synthesizer(text, length_scale=length_scale)
        from piper import PiperVoice, SynthesisConfig

        if self._voice is None:
            directory = Path(os.environ.get("SCENE_LAB_PIPER_DIR", DEFAULT_PIPER_DIR))
            model = directory / f"{self.voice_name}.onnx"
            if not model.is_file():
                raise FileNotFoundError(model)
            self._voice = PiperVoice.load(model)
        chunks = list(
            self._voice.synthesize(text, syn_config=SynthesisConfig(length_scale=length_scale))
        )
        data = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        return np.frombuffer(data, dtype="<i2").astype(np.float32), self._voice.config.sample_rate

    def synthesize(self, text: str, style: str = "normal") -> np.ndarray:
        if style not in {"normal", "mumble", "trailing"}:
            raise ValueError("unknown speech style")
        if not text.strip():
            return np.empty(0, dtype=np.int16)
        # trailing: Piper length_scale 1.12 slows speech by roughly 12%; the
        # final 40% fades to 2% amplitude. mumble: -12 dB and a one-pole
        # ~900 Hz low-pass to soften consonants.
        raw, rate = self._render(text, 1.12 if style == "trailing" else 1.0)
        data = _resample(np.asarray(raw, dtype=np.float32), rate, self.sample_rate)
        if style == "mumble":
            alpha = 1 - np.exp(-2 * np.pi * 900 / self.sample_rate)
            filtered = np.empty_like(data)
            previous = 0.0
            for index, sample in enumerate(data):
                previous += alpha * (float(sample) - previous)
                filtered[index] = previous
            data = filtered * (10 ** (-12 / 20))
        elif style == "trailing":
            start = int(len(data) * 0.6)
            data[start:] *= np.linspace(1, 0.02, len(data) - start, dtype=np.float32)
        return np.clip(np.rint(data), -32768, 32767).astype(np.int16)


class RoomNoise:
    def __init__(self, level_dbfs: float = -55, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.scale = 32768 * 10 ** (level_dbfs / 20)

    def next_chunk(self, samples: int = CHUNK_SAMPLES) -> np.ndarray:
        return np.clip(
            np.rint(self.rng.standard_normal(samples) * self.scale), -32768, 32767
        ).astype(np.int16)


class AudioStream:
    def __init__(self, noise: RoomNoise | None = None, chunk_samples: int = CHUNK_SAMPLES) -> None:
        self.noise = noise or RoomNoise()
        self.chunk_samples = chunk_samples
        self._queue: deque[np.ndarray] = deque()
        self._offset = 0

    @property
    def speaking(self) -> bool:
        return bool(self._queue)

    def queue(self, samples: np.ndarray) -> None:
        data = np.asarray(samples, dtype=np.int16)
        if data.ndim != 1:
            raise ValueError("speech must be mono")
        if len(data):
            self._queue.append(data.copy())

    def cancel(self) -> None:
        self._queue.clear()
        self._offset = 0

    def finish_current_only(self) -> None:
        """Discard queued speech while allowing the line on the microphone to finish."""
        if self._queue:
            current = self._queue[0]
            self._queue.clear()
            self._queue.append(current)

    def next_chunk(self) -> np.ndarray:
        mixed = self.noise.next_chunk(self.chunk_samples).astype(np.int32)
        position = 0
        while position < self.chunk_samples and self._queue:
            head = self._queue[0]
            count = min(self.chunk_samples - position, len(head) - self._offset)
            mixed[position : position + count] += head[self._offset : self._offset + count]
            position += count
            self._offset += count
            if self._offset == len(head):
                self._queue.popleft()
                self._offset = 0
        return np.clip(mixed, -32768, 32767).astype(np.int16)
