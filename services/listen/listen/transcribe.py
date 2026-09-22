"""Local faster-whisper transcription boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

# Whisper's own silence rule (openai/whisper `transcribe`): a segment the model
# thinks is probably not speech, decoded with low confidence, is dropped.
# On noise, Whisper otherwise invents stock phrases such as "Thank you."
NO_SPEECH_PROB_THRESHOLD = 0.6
LOW_LOGPROB_THRESHOLD = -1.0


@dataclass(frozen=True)
class Transcript:
    """Text and a normalised confidence for one in-memory utterance."""

    text: str
    confidence: float


class Transcriber(Protocol):
    """Injectable transcription boundary used by the Redis consume loop."""

    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript: ...


class FasterWhisperTranscriber:
    """Lazily load faster-whisper and transcribe 16 kHz PCM on CPU by default."""

    def __init__(
        self,
        model: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | None = None,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        return self._model

    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript:
        if sample_rate != 16_000:
            raise ValueError("faster-whisper input must be 16 kHz")

        import numpy as np

        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        segments_iter, _info = self._load_model().transcribe(
            audio,
            language="en",
            beam_size=1,
            # WebRTC VAD upstream is coarse and passes fan hum, clicks and
            # typing. Silero (bundled with faster-whisper) strips non-speech
            # before decoding, so noise-only audio costs almost nothing.
            vad_filter=True,
            condition_on_previous_text=False,
        )
        # Inference happens while consuming the generator.
        segments = [segment for segment in segments_iter if not _is_non_speech(segment)]
        text = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        ).strip()
        if not segments:
            return Transcript(text="", confidence=0.0)
        weights = [max(0.001, segment.end - segment.start) for segment in segments]
        mean_log_probability = sum(
            segment.avg_logprob * weight for segment, weight in zip(segments, weights, strict=True)
        ) / sum(weights)
        confidence = min(1.0, max(0.0, math.exp(mean_log_probability)))
        return Transcript(text=text, confidence=confidence)


def _is_non_speech(segment) -> bool:
    """Whisper's silence rule for one decoded segment."""
    no_speech_prob = getattr(segment, "no_speech_prob", 0.0)
    return no_speech_prob > NO_SPEECH_PROB_THRESHOLD and segment.avg_logprob < LOW_LOGPROB_THRESHOLD
