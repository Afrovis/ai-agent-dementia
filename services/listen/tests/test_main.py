"""Hardware-free integration tests for the listen consume loop."""

from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta

from nc_shared.bus import FakeBus
from nc_shared.events import AudioChunk, Health, SessionState, Utterance

from listen.main import ListenState, maybe_emit_health, run_once
from listen.transcribe import Transcript
from listen.vad import FRAME_MS, SAMPLE_RATE, SpeechSegmenter

FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def pcm_frame(value: int) -> bytes:
    return struct.pack(f"<{FRAME_SAMPLES}h", *([value] * FRAME_SAMPLES))


class NonZeroDetector:
    def __call__(self, pcm16: bytes, sample_rate: int) -> bool:
        assert sample_rate == SAMPLE_RATE
        return any(pcm16)


class FakeTranscriber:
    def __init__(self, text: str = "I need the toilet", confidence: float = 0.91) -> None:
        self.result = Transcript(text=text, confidence=confidence)
        self.calls: list[tuple[bytes, int]] = []

    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript:
        self.calls.append((pcm16, sample_rate))
        return self.result


class FailingTranscriber:
    def transcribe(self, pcm16: bytes, sample_rate: int) -> Transcript:
        raise RuntimeError("model unavailable")


def make_state(**segmenter_kwargs) -> ListenState:
    return ListenState(SpeechSegmenter(NonZeroDetector(), pre_roll_ms=0, **segmenter_kwargs))


def publish_session(bus: FakeBus, phase: str, session_id: str | None = "session-1") -> None:
    bus.publish(
        SessionState(
            source="agent",
            session_id=session_id,
            phase=phase,
            goal="return_to_bed",
            strategy_index=0,
        )
    )


def publish_audio(bus: FakeBus, pcm16: bytes) -> None:
    bus.publish(AudioChunk(source="embodiment", pcm16=pcm16, sample_rate=SAMPLE_RATE), maxlen=50)


def read_utterances(bus: FakeBus) -> list[Utterance]:
    bus.ensure_group("speech_in", "test")
    return [event for _, event in bus.read("speech_in", "test", "test-1")]


def test_segmenter_handles_arbitrary_chunks_and_ends_after_silence():
    segmenter = SpeechSegmenter(
        NonZeroDetector(), end_silence_ms=60, min_speech_ms=60, pre_roll_ms=0
    )
    audio = pcm_frame(1000) * 2 + pcm_frame(0) * 2

    first = segmenter.accept(audio[:317])
    second = segmenter.accept(audio[317:])

    assert first == []
    assert second == [audio]


def test_segmenter_rejects_a_click_shorter_than_minimum_speech():
    segmenter = SpeechSegmenter(
        NonZeroDetector(), end_silence_ms=60, min_speech_ms=60, pre_roll_ms=0
    )

    assert segmenter.accept(pcm_frame(1000) + pcm_frame(0) * 2) == []


def test_idle_audio_is_discarded_without_transcription():
    bus = FakeBus()
    state = make_state(end_silence_ms=60, min_speech_ms=60)
    transcriber = FakeTranscriber()
    publish_audio(bus, pcm_frame(1000) * 2 + pcm_frame(0) * 2)

    assert run_once(bus, state, transcriber) == 0
    assert transcriber.calls == []
    assert bus.pending("audio_in", "listen") == []


def test_observing_audio_publishes_session_attributed_utterance():
    bus = FakeBus()
    state = make_state(end_silence_ms=60, min_speech_ms=60)
    transcriber = FakeTranscriber()
    audio = pcm_frame(1000) * 2 + pcm_frame(0) * 2
    publish_session(bus, "OBSERVING")
    publish_audio(bus, audio)

    assert run_once(bus, state, transcriber) == 1

    utterances = read_utterances(bus)
    assert len(utterances) == 1
    assert utterances[0].text == "I need the toilet"
    assert utterances[0].confidence == 0.91
    assert utterances[0].duration_s == 0.12
    assert utterances[0].session_id == "session-1"
    assert transcriber.calls == [(audio, SAMPLE_RATE)]


def test_idle_transition_discards_partial_speech_before_next_session():
    bus = FakeBus()
    state = make_state(end_silence_ms=60, min_speech_ms=60)
    transcriber = FakeTranscriber()
    publish_session(bus, "ENGAGED", "old-session")
    publish_audio(bus, pcm_frame(1000))
    run_once(bus, state, transcriber)

    publish_session(bus, "IDLE", None)
    run_once(bus, state, transcriber)
    publish_session(bus, "OBSERVING", "new-session")
    publish_audio(bus, pcm_frame(1000) + pcm_frame(0) * 2)

    assert run_once(bus, state, transcriber) == 0
    assert read_utterances(bus) == []


def test_empty_transcript_is_not_published():
    bus = FakeBus()
    state = make_state(end_silence_ms=60, min_speech_ms=60)
    transcriber = FakeTranscriber(text="   ", confidence=0.0)
    publish_session(bus, "ENGAGED")
    publish_audio(bus, pcm_frame(1000) * 2 + pcm_frame(0) * 2)

    assert run_once(bus, state, transcriber) == 0
    assert read_utterances(bus) == []


def test_health_reports_processing_failure_without_audio_payload():
    bus = FakeBus()
    state = make_state()
    state.last_error = "RuntimeError: model unavailable"
    bus.ensure_group("health", "test")

    assert maybe_emit_health(bus, state, None, 100.0) == 100.0

    _, health = bus.read("health", "test", "test-1")[0]
    assert isinstance(health, Health)
    assert health.ok is False
    assert health.detail == "RuntimeError: model unavailable"


def test_transcription_failure_emits_one_attention_notification():
    bus = FakeBus()
    state = make_state(end_silence_ms=60, min_speech_ms=60)
    bus.ensure_group("notify", "test")
    publish_session(bus, "ENGAGED")
    audio = pcm_frame(1000) * 2 + pcm_frame(0) * 2
    publish_audio(bus, audio)

    assert run_once(bus, state, FailingTranscriber()) == 0

    notifications = [event for _, event in bus.read("notify", "test", "test-1")]
    assert len(notifications) == 1
    assert notifications[0].level == "attention"
    assert "speech" in notifications[0].title.lower()
    assert state.last_error == "RuntimeError: model unavailable"

    publish_audio(bus, audio)
    run_once(bus, state, FailingTranscriber())
    assert bus.read("notify", "test", "test-1") == []


def test_audio_from_before_session_is_not_transcribed_after_restart():
    bus = FakeBus()
    state = make_state(end_silence_ms=60, min_speech_ms=60)
    transcriber = FakeTranscriber()
    old = datetime(2026, 1, 1, tzinfo=UTC)
    bus.publish(
        AudioChunk(
            source="embodiment",
            ts=old,
            pcm16=pcm_frame(1000) * 2 + pcm_frame(0) * 2,
            sample_rate=SAMPLE_RATE,
        ),
        maxlen=50,
    )
    bus.publish(
        SessionState(
            source="agent",
            ts=old + timedelta(seconds=1),
            session_id="session-1",
            phase="OBSERVING",
            goal="return_to_bed",
            strategy_index=0,
        )
    )

    assert run_once(bus, state, transcriber) == 0
    assert transcriber.calls == []


def test_session_backlog_is_drained_before_audio():
    bus = FakeBus()
    state = make_state(end_silence_ms=60, min_speech_ms=60)
    transcriber = FakeTranscriber()
    for index in range(12):
        publish_session(bus, "OBSERVING", f"session-{index}")
    publish_session(bus, "IDLE", None)
    publish_audio(bus, pcm_frame(1000) * 2 + pcm_frame(0) * 2)

    assert run_once(bus, state, transcriber, count=10) == 0
    assert state.phase == "IDLE"
    assert transcriber.calls == []
