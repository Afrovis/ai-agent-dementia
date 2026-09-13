"""Session-gated VAD and local speech-to-text service (issue #18).

``listen`` consumes ephemeral 16 kHz PCM ``AudioChunk`` events from the
browser media bridge and current ``SessionState`` events from ``agent``. Audio
is discarded while the phase is ``IDLE``. During ``OBSERVING`` and every later
session phase, WebRTC VAD forms bounded in-memory utterances and faster-whisper
``small.en`` transcribes them locally into ``Utterance`` events.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import redis
from nc_shared.bus import Bus
from nc_shared.events import AudioChunk, Health, Notify, SessionState, Utterance

from listen.config import ListenConfig
from listen.transcribe import FasterWhisperTranscriber, Transcriber
from listen.vad import SpeechSegmenter, WebRtcVoiceDetector

SERVICE_NAME = "listen"
GROUP = "listen"
SESSION_GROUP = "listen-session"
CONSUMER = "listen-1"
AUDIO_STREAM = "audio_in"
SESSION_STREAM = "session"
HEALTH_INTERVAL_S = 30.0
ACTIVE_PHASES = {"OBSERVING", "ENGAGED", "COOLDOWN", "ESCALATED"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, level: int = logging.INFO, **fields: object) -> None:
    """Log metadata only; never include audio bytes or transcript text."""
    logger.log(level, json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


@dataclass
class ListenState:
    """Mutable session and error state for the bounded consume loop."""

    segmenter: SpeechSegmenter
    phase: str = "IDLE"
    session_id: str | None = None
    active_since: datetime | None = None
    last_error: str | None = None
    fault_notified: bool = False

    @property
    def active(self) -> bool:
        return self.phase in ACTIVE_PHASES and self.session_id is not None

    def update_session(self, event: SessionState) -> None:
        session_changed = event.session_id != self.session_id
        if session_changed or event.phase == "IDLE":
            self.segmenter.reset()
        if event.phase == "IDLE":
            self.active_since = None
        elif session_changed or self.active_since is None:
            self.active_since = event.ts
        self.phase = event.phase
        self.session_id = event.session_id


def run_once(bus, state: ListenState, transcriber: Transcriber, *, count: int = 10) -> int:
    """Consume one batch and return the number of published utterances.

    The function is deliberately injectable and bounded so tests require no
    Redis, microphone, VAD library, or Whisper model.
    """
    bus.ensure_group(SESSION_STREAM, SESSION_GROUP)
    bus.ensure_group(AUDIO_STREAM, GROUP)

    while True:
        session_messages = bus.read(
            SESSION_STREAM, SESSION_GROUP, CONSUMER, count=count, block_ms=1
        )
        for msg_id, event in session_messages:
            if isinstance(event, SessionState):
                state.update_session(event)
            bus.ack(SESSION_STREAM, SESSION_GROUP, msg_id)
        if len(session_messages) < count:
            break

    published = 0
    for msg_id, event in bus.read(AUDIO_STREAM, GROUP, CONSUMER, count=count, block_ms=100):
        try:
            is_current_audio = state.active_since is None or event.ts >= state.active_since
            if state.active and is_current_audio and isinstance(event, AudioChunk):
                for pcm16 in state.segmenter.accept(event.pcm16):
                    transcript = transcriber.transcribe(pcm16, event.sample_rate)
                    text = transcript.text.strip()
                    if not text:
                        continue
                    duration_s = len(pcm16) / (event.sample_rate * 2)
                    bus.publish(
                        Utterance(
                            source=SERVICE_NAME,
                            session_id=state.session_id,
                            text=text,
                            confidence=transcript.confidence,
                            duration_s=duration_s,
                        )
                    )
                    state.last_error = None
                    state.fault_notified = False
                    published += 1
                    _log(
                        "published Utterance",
                        event_type="Utterance",
                        session_id=state.session_id,
                        duration_s=round(duration_s, 3),
                        confidence=round(transcript.confidence, 3),
                        text_chars=len(text),
                    )
        except Exception as exc:  # noqa: BLE001 - keep the live audio loop running.
            state.last_error = f"{type(exc).__name__}: {exc}"
            state.segmenter.reset()
            _log("audio processing failed", level=logging.ERROR, error=state.last_error)
            if not state.fault_notified:
                bus.publish(
                    Notify(
                        source=SERVICE_NAME,
                        session_id=state.session_id,
                        level="attention",
                        title="Night Companion cannot understand speech",
                        body="Local speech recognition failed; check the listen service.",
                        repeat_until_ack=False,
                    )
                )
                state.fault_notified = True
        finally:
            # The stream is capped and raw audio must not be retained. A failure
            # is surfaced through Health rather than leaving audio pending.
            bus.ack(AUDIO_STREAM, GROUP, msg_id)
    return published


def maybe_emit_health(
    bus,
    state: ListenState,
    last_emitted_at: float | None,
    now: float,
    *,
    interval: float = HEALTH_INTERVAL_S,
) -> float | None:
    """Publish a heartbeat, reporting the most recent processing failure."""
    if last_emitted_at is not None and now - last_emitted_at < interval:
        return last_emitted_at
    ok = state.last_error is None
    detail = "running" if ok else state.last_error
    bus.publish(Health(source=SERVICE_NAME, service=SERVICE_NAME, ok=ok, detail=detail))
    _log("published Health", event_type="Health", ok=ok, detail=detail)
    return now


def build_state(config: ListenConfig) -> ListenState:
    """Build the real WebRTC VAD segmenter from configuration."""
    detector = WebRtcVoiceDetector(config.vad_aggressiveness)
    return ListenState(
        segmenter=SpeechSegmenter(
            detector,
            end_silence_ms=config.end_silence_ms,
            min_speech_ms=config.min_speech_ms,
            max_utterance_seconds=config.max_utterance_seconds,
        )
    )


def run(*, clock: Callable[[], float] = time.time) -> None:
    """Connect to Redis and continuously transcribe session-gated speech."""
    config = ListenConfig.from_env()
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    bus = Bus(redis.Redis.from_url(redis_url))
    state = build_state(config)
    transcriber = FasterWhisperTranscriber(
        config.model,
        config.device,
        config.compute_type,
        download_root=config.model_cache,
    )
    _log(
        "listen starting",
        model=config.model,
        model_cache=config.model_cache,
        device=config.device,
        compute_type=config.compute_type,
    )

    last_health_at: float | None = None
    while True:
        run_once(bus, state, transcriber)
        last_health_at = maybe_emit_health(bus, state, last_health_at, clock())


if __name__ == "__main__":
    run()
