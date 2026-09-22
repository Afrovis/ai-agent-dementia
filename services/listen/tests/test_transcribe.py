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
    no_speech_prob: float = 0.0


class FakeModel:
    def __init__(self, segments=None) -> None:
        self.kwargs = None
        self.segments = segments

    def transcribe(self, audio, **kwargs):
        self.audio = audio
        self.kwargs = kwargs
        segments = self.segments or [
            FakeSegment(" I need", 0.0, 0.5, -0.1),
            FakeSegment(" water. ", 0.5, 1.0, -0.3),
        ]
        return iter(segments), object()


def test_pcm_is_transcribed_as_english_with_silero_vad():
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
        "vad_filter": True,
        "condition_on_previous_text": False,
    }


def test_non_16khz_audio_is_rejected():
    with pytest.raises(ValueError, match="16 kHz"):
        FasterWhisperTranscriber().transcribe(b"\x00\x00", 8_000)


def test_segments_whisper_marks_as_non_speech_are_dropped():
    transcriber = FasterWhisperTranscriber()
    transcriber._model = FakeModel(
        [FakeSegment(" Thank you.", 0.0, 2.0, avg_logprob=-1.2, no_speech_prob=0.8)]
    )

    result = transcriber.transcribe(b"\x00\x00", 16_000)

    assert result.text == ""
    assert result.confidence == 0.0


def test_confident_speech_survives_a_high_no_speech_probability():
    transcriber = FasterWhisperTranscriber()
    transcriber._model = FakeModel(
        [FakeSegment(" I need the bathroom.", 0.0, 2.0, avg_logprob=-0.2, no_speech_prob=0.7)]
    )

    assert transcriber.transcribe(b"\x00\x00", 16_000).text == "I need the bathroom."
