"""Session-gated VAD and local speech-to-text service (issue #18).

``listen`` consumes ephemeral 16 kHz PCM ``AudioChunk`` events from the
browser media bridge and current ``SessionState`` events from ``agent``. Audio
is discarded while the phase is ``IDLE``. During ``OBSERVING`` and every later
session phase, WebRTC VAD forms bounded in-memory utterances and faster-whisper
``small.en`` transcribes them locally into ``Utterance`` events. It also emits
a payload-free ``SpeechStarted`` at VAD onset so interruptible bedside speech
can stop before transcription finishes (issue #20).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import redis
from nc_shared.bus import Bus
from nc_shared.events import (
    Activity,
    AudioChunk,
    Health,
    Notify,
    SessionState,
    SpeechStarted,
    Utterance,
)

from listen.config import ListenConfig
from listen.transcribe import FasterWhisperTranscriber, Transcriber
from listen.vad import FRAME_MS, SpeechSegmenter, WebRtcVoiceDetector

SERVICE_NAME = "listen"
GROUP = "listen"
SESSION_GROUP = "listen-session"
CONSUMER = "listen-1"
AUDIO_STREAM = "audio_in"
SESSION_STREAM = "session"
ACTIVITY_STREAM = "activity"
ACTIVITY_GROUP = "listen-activity"
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
    barge_in_ms: int = 600
    barge_in_while_speaking_ms: int = 1500
    verify_speech: Callable[[bytes], bool] | None = None
    playback_until: float = 0.0
    speech_frames: list[bytes] = field(default_factory=list)
    barge_checked: bool = False
    barge_silence_ms: int = 0

    def reset_barge(self) -> None:
        self.speech_frames.clear()
        self.barge_checked = False
        self.barge_silence_ms = 0

    def on_playback(self, event: Activity, now: float) -> None:
        if event.service != "embodiment" or event.kind != "playback":
            return
        if event.phase == "start" and event.detail in {"requested", "playing"}:
            self.playback_until = now + 20
        elif event.phase == "end":
            self.playback_until = 0.0

    @property
    def active(self) -> bool:
        return self.phase in ACTIVE_PHASES and self.session_id is not None

    def update_session(self, event: SessionState) -> None:
        session_changed = event.session_id != self.session_id
        if session_changed or event.phase == "IDLE":
            self.segmenter.reset()
            self.reset_barge()
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
    bus.ensure_group(ACTIVITY_STREAM, ACTIVITY_GROUP)

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

    while True:
        activity_messages = bus.read(
            ACTIVITY_STREAM, ACTIVITY_GROUP, CONSUMER, count=count, block_ms=None
        )
        for msg_id, event in activity_messages:
            if isinstance(event, Activity):
                state.on_playback(event, time.monotonic())
            bus.ack(ACTIVITY_STREAM, ACTIVITY_GROUP, msg_id)
        if len(activity_messages) < count:
            break

    def check_frame(frame: bytes, speech: bool) -> None:
        if not speech:
            if state.speech_frames and not state.barge_checked:
                _log(
                    "suppressed barge-in",
                    reason="insufficient_duration",
                    while_speaking=time.monotonic() < state.playback_until,
                    vad_ms=len(state.speech_frames) * FRAME_MS,
                )
            state.speech_frames.clear()
            state.barge_silence_ms += FRAME_MS
            if state.barge_silence_ms >= state.segmenter.end_silence_ms:
                state.reset_barge()
            return
        state.barge_silence_ms = 0
        if state.barge_checked:
            return
        state.speech_frames.append(frame)
        speaking = time.monotonic() < state.playback_until
        threshold = state.barge_in_while_speaking_ms if speaking else state.barge_in_ms
        if len(state.speech_frames) * FRAME_MS < threshold:
            return
        state.barge_checked = True
        window = b"".join(state.speech_frames)
        confirmed = state.verify_speech(window) if state.verify_speech is not None else False
        if confirmed:
            bus.publish(SpeechStarted(source=SERVICE_NAME, session_id=state.session_id))
            _log("published SpeechStarted", event_type="SpeechStarted", session_id=state.session_id)
        else:
            _log(
                "suppressed barge-in",
                reason="silero_rejected",
                while_speaking=speaking,
                vad_ms=len(state.speech_frames) * FRAME_MS,
            )

    published = 0
    for msg_id, event in bus.read(AUDIO_STREAM, GROUP, CONSUMER, count=count, block_ms=100):
        try:
            is_current_audio = state.active_since is None or event.ts >= state.active_since
            if state.active and is_current_audio and isinstance(event, AudioChunk):
                state.segmenter.on_frame = check_frame
                state.segmenter.on_segment_end = state.reset_barge
                utterances = state.segmenter.accept(event.pcm16)
                state.segmenter.take_speech_starts()
                for pcm16 in utterances:
                    bus.publish(
                        Activity(
                            source=SERVICE_NAME,
                            service=SERVICE_NAME,
                            kind="transcribe",
                            phase="start",
                        ),
                        maxlen=200,
                    )
                    _log(
                        "published Activity",
                        event_type="Activity",
                        kind="transcribe",
                        phase="start",
                    )
                    started = time.perf_counter()
                    text = ""
                    error = None
                    try:
                        transcript = transcriber.transcribe(pcm16, event.sample_rate)
                        text = transcript.text.strip()
                    except Exception as exc:
                        error = type(exc).__name__
                        raise
                    finally:
                        bus.publish(
                            Activity(
                                source=SERVICE_NAME,
                                service=SERVICE_NAME,
                                kind="transcribe",
                                phase="end",
                                ok=bool(text) and error is None,
                                duration_ms=(time.perf_counter() - started) * 1000,
                                detail=error or f"{len(text)} chars",
                            ),
                            maxlen=200,
                        )
                        _log(
                            "published Activity",
                            event_type="Activity",
                            kind="transcribe",
                            phase="end",
                            ok=bool(text) and error is None,
                            detail=error or f"{len(text)} chars",
                        )
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
            state.reset_barge()
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
        ),
        barge_in_ms=config.barge_in_ms,
        barge_in_while_speaking_ms=config.barge_in_while_speaking_ms,
        verify_speech=SileroVerifier(),
    )


class SileroVerifier:
    """Use the Silero model bundled with faster-whisper, loaded on first speech."""

    def __call__(self, pcm16: bytes) -> bool:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        return bool(
            get_speech_timestamps(
                audio, vad_options=VadOptions(min_speech_duration_ms=100), sampling_rate=16000
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
