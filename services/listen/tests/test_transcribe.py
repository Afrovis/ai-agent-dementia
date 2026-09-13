"""Unit tests for the faster-whisper adapter without loading model weights."""

from dataclasses import dataclass

import pytest

from listen.transcribe import FasterWhisperTranscriber


@dataclass
class FakeSegment:
    text: str
    start: float
    end: float
    avg_logprob: float


class FakeModel:
    def __init__(self) -> None:
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        return iter(
            [
                FakeSegment(" I need", 0.0, 0.5, -0.1),
                FakeSegment(" water. ", 0.5, 1.0, -0.3),
            ]
        ), object()


def test_pcm_is_transcribed_as_english_without_second_vad_pass():
    transcriber = FasterWhisperTranscriber()
    model = FakeModel()
    transcriber._model = model

    result = transcriber.transcribe(b"\x00\x00\xff\x7f", 16_000)

    assert result.text == "I need water."
    assert result.confidence == pytest.approx(0.8187, rel=1e-3)
    assert model.audio.dtype.name == "float32"
    assert model.kwargs == {
        "language": "en",
        "beam_size": 1,
        "vad_filter": False,
        "condition_on_previous_text": False,
    }


def test_non_16khz_audio_is_rejected():
    with pytest.raises(ValueError, match="16 kHz"):
        FasterWhisperTranscriber().transcribe(b"\x00\x00", 8_000)
