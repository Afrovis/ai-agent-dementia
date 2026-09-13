"""Pydantic event schemas for the Night Companion Redis-stream bus.

Every event is a pydantic model that inherits from :class:`BaseEvent`, which
carries the three fields common to every event on the bus: ``ts`` (UTC,
ISO 8601, defaults to "now"), ``source`` (the producing service name), and
``session_id`` (nullable outside of a session). See HANDOFF.md section 5 for
the authoritative contract table; this module and that table must be kept
in sync.

``EVENT_STREAMS`` and ``EVENT_TYPES`` form a two-way registry between event
classes and the Redis stream name they are published on, used by
``nc_shared.bus`` to route publishes and reconstruct events on read.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, JsonValue, PlainSerializer


def _utcnow() -> datetime:
    """Return the current time, timezone-aware, in UTC."""
    return datetime.now(UTC)


def _decode_base64_if_str(value: object) -> object:
    """Decode base64 text back into raw bytes; pass raw bytes through unchanged.

    Used as a validator so that constructing an event in Python takes plain
    ``bytes`` (e.g. `Frame(jpeg=raw_jpeg_bytes, ...)`), while loading the same
    field back from JSON (where it was serialised as base64 text, see
    `_encode_base64`) reconstructs the original bytes.
    """
    if isinstance(value, str):
        return base64.b64decode(value)
    return value


def _encode_base64(value: bytes) -> str:
    """Encode raw bytes as base64 text for JSON output.

    JSON has no binary type, so binary fields (`Frame.jpeg`,
    `AudioChunk.pcm16`) are base64-encoded inside the JSON payload rather
    than the payload itself carrying raw bytes.
    """
    return base64.b64encode(value).decode("ascii")


BytesAsBase64 = Annotated[
    bytes,
    BeforeValidator(_decode_base64_if_str),
    PlainSerializer(_encode_base64, return_type=str, when_used="json"),
]
"""A `bytes` field that round-trips as plain bytes in Python and as base64
text when serialised to JSON (``model_dump_json``/``model_validate_json``)."""


class BaseEvent(BaseModel):
    """Fields common to every event on the bus.

    ``ts`` is UTC and ISO 8601 when serialised. ``source`` is the name of
    the service that produced the event. ``session_id`` is ``None`` when the
    event is not associated with an active session.
    """

    ts: datetime = Field(default_factory=_utcnow)
    source: str
    session_id: str | None = None


class Frame(BaseEvent):
    """A single camera frame, gated by `capture`'s motion gate. Produced by
    `capture`. Never stored on disk."""

    jpeg: BytesAsBase64
    width: int
    height: int
    source_kind: Literal["browser", "usb", "rtsp"]


class RawFrame(BaseEvent):
    """A single camera frame straight from a source, before `capture`'s motion
    gate has had a chance to look at it. Produced by the `embodiment` browser
    bridge (issue #28), which is the only source that has to cross a process
    boundary to reach `capture`; the `usb`/`rtsp` sources are read inside
    `capture` itself and never take this detour. `capture` reads this stream,
    applies the motion gate, and republishes the frames it admits as `Frame`
    on `frames`. Never reaches disk and is never gated on rate or motion.
    Same shape as `Frame` on purpose: gating is the only thing that
    distinguishes the two streams."""

    jpeg: BytesAsBase64
    width: int
    height: int
    source_kind: Literal["browser", "usb", "rtsp"]


class PersonState(BaseEvent):
    """Pose/location classification for the tracked person. Produced by `perceive`."""

    state: Literal["in_bed", "sitting_up", "standing", "walking", "on_floor", "absent"]
    confidence: float
    zone: Literal["bed", "door", "bathroom_path", "other"]
    scene_note: str | None = None


class Utterance(BaseEvent):
    """A transcribed utterance from the person. Produced by `listen`."""

    text: str
    confidence: float
    duration_s: float


class SpeechStarted(BaseEvent):
    """Voice activity onset from the person. Produced by `listen`.

    This deliberately carries no audio or transcript.  It exists so the
    bedside browser can stop interruptible speech as soon as VAD fires,
    without waiting for the utterance to end and Whisper to transcribe it.
    """


class SessionState(BaseEvent):
    """Current agent session phase. Produced by `agent`."""

    phase: Literal["IDLE", "OBSERVING", "ENGAGED", "COOLDOWN", "ESCALATED"]
    goal: str
    strategy_index: int


class GoalChanged(BaseEvent):
    """A change of goal within a session. Produced by `agent`."""

    from_goal: str
    to_goal: str
    reason: str


class CloudCall(BaseEvent):
    """Audit record for an opted-in, text-only cloud LLM request.

    ``payload`` is the exact structured user data sent to Claude. ``JsonValue``
    deliberately excludes bytes, which makes image/audio data invalid at the
    event-contract boundary rather than relying on caller discipline alone.
    """

    task: Literal["interpret", "plan"]
    model: str
    payload: dict[str, JsonValue]


class Say(BaseEvent):
    """Text-to-speech instruction. Produced by `agent`."""

    text: str
    strategy: str
    interruptible: bool


class Show(BaseEvent):
    """On-screen embodiment instruction. Produced by `agent`."""

    face: Literal["asleep", "awake", "speaking", "listening"]
    headline: str
    body: str
    photo_id: str | None = None
    brightness: float


class Notify(BaseEvent):
    """Caregiver notification. Produced by `agent` or any service on fault."""

    level: Literal["info", "attention", "critical"]
    title: str
    body: str
    repeat_until_ack: bool


class Ack(BaseEvent):
    """Acknowledgement of a notification. Produced by `dashboard` or `notify`."""

    notify_id: str


class LightCommand(BaseEvent):
    """Idempotent command for a configured room light. Produced by `agent`.

    Commands carry the desired state rather than a toggle so Redis stream
    redelivery cannot accidentally reverse the light.
    """

    light: Literal["hallway"]
    state: Literal["on", "off"]
    reason: str


class AudioChunk(BaseEvent):
    """A chunk of raw PCM audio. Produced by `embodiment` (browser bridge)."""

    pcm16: BytesAsBase64
    sample_rate: Literal[16000]


class Health(BaseEvent):
    """Liveness heartbeat. Produced by every service, every 30 s."""

    service: str
    ok: bool
    detail: str


EVENT_STREAMS: dict[type[BaseEvent], str] = {
    Frame: "frames",
    RawFrame: "frames_raw",
    PersonState: "person",
    SpeechStarted: "speech_in",
    Utterance: "speech_in",
    SessionState: "session",
    GoalChanged: "session",
    CloudCall: "cloud",
    Say: "say",
    Show: "show",
    Notify: "notify",
    Ack: "ack",
    LightCommand: "light",
    AudioChunk: "audio_in",
    Health: "health",
}

EVENT_TYPES: dict[str, type[BaseEvent]] = {
    event_cls.__name__: event_cls for event_cls in EVENT_STREAMS
}
