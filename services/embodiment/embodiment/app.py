"""FastAPI application for the `embodiment` service (issue #3).

Serves the fullscreen embodiment page (face plus big text) over HTTP/HTTPS
and pushes `Show`/`Say`/`SpeechStarted` events to connected browsers over a
WebSocket, so display changes and barge-in arrive live with no polling.

The bus-reading side is deliberately split into small, injectable pieces so
it can be tested with `nc_shared.bus.FakeBus` and no real Redis or network:

- `ConnectionManager` tracks connected websockets and the last-known `Show`,
  `PersonState`, and `SessionState` for late joiners.
- `broadcast_loop(bus, manager, ...)` is a plain async function that reads
  display, speech, person, session, pose, and activity streams, then forwards
  browser messages to `manager.broadcast`. It accepts `max_iterations` and/or a
  `stop_event` so tests can bound it instead of looping forever.
- `create_app(bus)` wires a `ConnectionManager` into the routes and starts
  `broadcast_loop` as a background task on startup.

The browser media bridge (issue #28) is the reverse direction: the `/media`
websocket receives JSON messages from the browser (captured webcam frames
and mic audio) and publishes them onto the bus as `RawFrame`/`AudioChunk`
events. `publish_frame`/`publish_audio_chunk` do the actual decode-and-
publish work and are plain functions so tests can call them directly with a
`FakeBus`, without going through a websocket at all. Frames go on
`frames_raw`, ungated, as `RawFrame`: `capture` (issue #7) is the one
service that reads `frames_raw`, applies the motion gate, and republishes
what it admits as `Frame` on `frames` (HANDOFF.md section 5).

`GET /photos/{photo_id}` serves the optional photo a `Show` event points at.
`resolve_photo` maps an id to a file, preferring caregiver uploads under
`PHOTO_DIR` over the demo images bundled in `demo_photos/`. A missing photo
answers 404 and the page hides its photo layer, which is the normal case on
a fresh install where nothing has been uploaded yet.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from nc_shared.events import (
    Ack,
    Activity,
    AudioChunk,
    BedZoneStatus,
    CalibrateBed,
    DebugControl,
    Gaze,
    Health,
    Notify,
    PersonState,
    PoseDebug,
    RawFrame,
    ResetSession,
    Say,
    SessionState,
    Show,
    SpeechStarted,
    Utterance,
)
from nc_shared.replay import CAPPED_MAXLEN

from embodiment.eyes import EyesState

SERVICE_NAME = "embodiment"
GROUP = "embodiment"
CONSUMER = "embodiment-1"

STATIC_DIR = Path(__file__).parent / "static"

# Caregiver-uploaded photos live under the gitignored `data/` tree
# (HANDOFF.md section 6), so a fresh install legitimately has none.
DEFAULT_PHOTO_DIR = Path("data/photos")
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_VOICE_CLIP_DIR = Path("/app/data/voice-clips")

# Placeholder images shipped with the service, so the fake agent's `demo_*`
# photo ids resolve on a fresh checkout. Searched only after `photo_dir`, so
# a real uploaded photo always wins over a demo one of the same name.
DEMO_PHOTO_DIR = Path(__file__).parent / "demo_photos"

# A `photo_id` is an opaque id the dashboard assigns on upload. Restricting
# it to this character set is what keeps a crafted id such as
# `../../etc/passwd` from escaping the photo directory.
PHOTO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
VOICE_CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
PLAYBACK_PHASES = {
    "received": "start",
    "requested": "start",
    "playing": "start",
    "unlocked": "start",
    "ended": "end",
    "failed": "end",
    "interrupted": "end",
    "no_audio": "end",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


@dataclass
class ClientSocket:
    channel: str
    peer: str
    connected_at: float
    page_id: str | None = None
    device_id: str | None = None
    user_agent: str = "unknown"
    browser: str = "unknown"
    platform: str = "unknown"
    screen: str = "unknown"
    page_load_time: str = "unknown"
    visibility: str = "unknown"
    audio_unlocked: bool = False
    last_heartbeat: float | None = None
    stale: bool = False


def _browser(user_agent: str) -> str:
    for name in ("Edg", "Firefox", "Chrome", "Safari"):
        if name in user_agent:
            return "Edge" if name == "Edg" else name
    return "unknown"


def _client_state(message: dict) -> dict | None:
    """Accept only small, typed identity messages from a browser."""
    if not isinstance(message, dict) or message.get("type") not in {
        "hello",
        "heartbeat",
        "visibility",
    }:
        return None
    for key, limit in (
        ("page_id", 64),
        ("device_id", 64),
        ("user_agent", 256),
        ("platform", 80),
        ("page_load_time", 64),
    ):
        value = message.get(key)
        if not isinstance(value, str) or not 1 <= len(value) <= limit:
            return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", message["page_id"]):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", message["device_id"]):
        return None
    screen = message.get("screen")
    if not isinstance(screen, dict) or set(screen) != {"width", "height"}:
        return None
    if any(type(screen[key]) is not int or not 1 <= screen[key] <= 16384 for key in screen):
        return None
    if message.get("visibility") not in {"visible", "hidden"}:
        return None
    if type(message.get("audio_unlocked")) is not bool:
        return None
    return message


class ConnectionManager:
    """Tracks connected websocket clients and the last-known `Show` state."""

    def __init__(
        self,
        bus=None,
        *,
        night_start: str = "20:00",
        night_end: str = "07:00",
        clock_24h: bool = False,
    ) -> None:
        self._connections: list[WebSocket] = []
        self._clients: dict[WebSocket, ClientSocket] = {}
        self.bus = bus
        self._last_page_count = 0
        self.last_show: dict | None = None
        self.last_person: dict | None = None
        self.last_session: dict | None = None
        self.last_eyes: dict = EyesState().state
        self.config = {
            "type": "config",
            "night_start": night_start,
            "night_end": night_end,
            "clock_24h": clock_24h,
        }
        self.last_debug_state: dict | None = None
        self.last_bed_zone: dict | None = None
        self.debug_controls_enabled = False

    async def connect(self, websocket: WebSocket, channel: str = "ws") -> None:
        """Accept `websocket` and register it for future broadcasts."""
        await websocket.accept()
        if channel == "ws":
            self._connections.append(websocket)
        forwarded = websocket.headers.get("x-forwarded-for", "").split(",")[0].strip()
        peer = forwarded or (websocket.client.host if websocket.client else "unknown")
        self._clients[websocket] = ClientSocket(channel, peer, time.monotonic())
        logger.info(
            json.dumps(
                {
                    "service": SERVICE_NAME,
                    "event_type": "client connected",
                    "channel": channel,
                    "peer": peer,
                }
            )
        )

    async def disconnect(
        self, websocket: WebSocket, code: int | None = None, reason: str = ""
    ) -> None:
        """Remove `websocket` from the connected set, if present."""
        if websocket in self._connections:
            self._connections.remove(websocket)
        client = self._clients.pop(websocket, None)
        if client is None:
            return
        duration = round(time.monotonic() - client.connected_at, 3)
        logger.info(
            json.dumps(
                {
                    "service": SERVICE_NAME,
                    "event_type": "client disconnected",
                    "channel": client.channel,
                    "peer": client.peer,
                    "page_id": client.page_id,
                    "device_id": client.device_id,
                    "duration_s": duration,
                    "close_code": code,
                    "close_reason": reason[:120],
                }
            )
        )
        self._activity(client, "end", code in {1000, 1001, None}, duration * 1000)
        await self._clients_changed()

    def _activity(
        self, client: ClientSocket, phase: str, ok: bool, duration_ms: float | None = None
    ) -> None:
        if self.bus is None or client.page_id is None:
            return
        detail = (
            f"{client.page_id[:8]} {client.browser} {client.visibility} "
            f"audio:{'ok' if client.audio_unlocked else 'blocked'}"
        )[:100]
        self.bus.publish(
            Activity(
                source=SERVICE_NAME,
                service=SERVICE_NAME,
                kind="client",
                phase=phase,
                ok=ok,
                duration_ms=duration_ms,
                detail=detail,
            ),
            maxlen=CAPPED_MAXLEN["activity"],
        )

    def pages(self) -> list[dict]:
        pages: dict[str, ClientSocket] = {}
        for client in self._clients.values():
            if client.page_id and not client.stale:
                if client.page_id not in pages or client.channel == "ws":
                    pages[client.page_id] = client
        now = time.monotonic()
        return [
            {
                "page_id": client.page_id[:8],
                "browser": client.browser,
                "visible": client.visibility == "visible",
                "audio": client.audio_unlocked,
                "age": round(now - client.connected_at),
            }
            for client in pages.values()
        ]

    async def _clients_changed(self) -> None:
        pages = self.pages()
        count = len(pages)
        if self.bus is not None and count != self._last_page_count:
            self.bus.publish(
                Health(
                    source=SERVICE_NAME,
                    service=SERVICE_NAME,
                    ok=True,
                    detail=f"{count} page{'s' if count != 1 else ''} connected",
                )
            )
        self._last_page_count = count
        raw = json.dumps({"type": "clients", "pages": pages})
        failed = []
        for socket in list(self._connections):
            if socket not in self._clients:
                continue
            try:
                await socket.send_text(raw)
            except Exception:  # noqa: BLE001 - socket may have closed during update
                failed.append(socket)
        for socket in failed:
            await self.disconnect(socket, reason="send failed")

    async def update(self, websocket: WebSocket, message: dict) -> None:
        client = self._clients.get(websocket)
        state = _client_state(message)
        if client is None or state is None:
            return
        if message["type"] != "hello" and client.page_id != state["page_id"]:
            return
        if message["type"] == "hello" and client.page_id is not None:
            return
        old_pages = {page["page_id"] for page in self.pages()}
        old_visibility = client.visibility
        old_audio = client.audio_unlocked
        was_stale = client.stale
        client.page_id = state["page_id"]
        client.device_id = state["device_id"]
        client.user_agent = state["user_agent"]
        client.browser = _browser(client.user_agent)
        client.platform = state["platform"]
        client.screen = f"{state['screen']['width']}x{state['screen']['height']}"
        client.page_load_time = state["page_load_time"]
        client.visibility = state["visibility"]
        client.audio_unlocked = state["audio_unlocked"]
        client.last_heartbeat = time.monotonic()
        client.stale = False
        if message["type"] == "hello":
            logger.info(
                json.dumps(
                    {
                        "service": SERVICE_NAME,
                        "event_type": "client hello",
                        "channel": client.channel,
                        "peer": client.peer,
                        "page_id": client.page_id,
                        "device_id": client.device_id,
                        "user_agent": client.user_agent,
                        "platform": client.platform,
                        "screen": client.screen,
                        "page_load_time": client.page_load_time,
                        "visibility": client.visibility,
                        "audio_unlocked": client.audio_unlocked,
                    }
                )
            )
            self._activity(client, "start", True)
        if old_visibility != client.visibility and message["type"] != "hello":
            logger.info(
                json.dumps(
                    {
                        "service": SERVICE_NAME,
                        "event_type": "client visibility",
                        "channel": client.channel,
                        "page_id": client.page_id,
                        "visibility": client.visibility,
                        "audio_unlocked": client.audio_unlocked,
                    }
                )
            )
        if (
            old_pages != {page["page_id"] for page in self.pages()}
            or old_visibility != client.visibility
            or old_audio != client.audio_unlocked
            or was_stale
            or (message["type"] == "hello" and client.channel == "ws")
        ):
            await self._clients_changed()

    async def mark_stale(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        changed = False
        for client in self._clients.values():
            if (
                client.page_id
                and not client.stale
                and client.last_heartbeat is not None
                and now - client.last_heartbeat > 30
            ):
                client.stale = True
                changed = True
                logger.info(
                    json.dumps(
                        {
                            "service": SERVICE_NAME,
                            "event_type": "client stale",
                            "channel": client.channel,
                            "page_id": client.page_id,
                            "seconds_since_heartbeat": round(now - client.last_heartbeat, 1),
                        }
                    )
                )
        if changed:
            await self._clients_changed()

    async def broadcast(self, message: dict) -> None:
        """Send `message` as JSON to every connected client, dropping dead ones.

        Also remembers the message if it is a `show` event, so a client that
        connects later can be caught up in `send_current_state`.
        """
        if message.get("type") == "show":
            self.last_show = message
        elif message.get("type") == "person":
            self.last_person = message
        elif message.get("type") == "session":
            self.last_session = message
        elif message.get("type") == "eyes":
            self.last_eyes = message
        elif message.get("type") == "debug_state":
            self.last_debug_state = message
        elif message.get("type") == "bed_zone":
            self.last_bed_zone = message
        dead: list[WebSocket] = []
        recipients = []
        for connection in list(self._connections):
            client = self._clients.get(connection)
            recipient = {
                "page_id": client.page_id if client else None,
                "visibility": client.visibility if client else "unknown",
                "audio_unlocked": client.audio_unlocked if client else False,
            }
            try:
                await connection.send_text(json.dumps(message))
                recipients.append(recipient)
            except Exception:  # noqa: BLE001 - best-effort broadcast
                dead.append(connection)
        if message.get("type") in {"say", "show", "speech_started"}:
            logger.info(
                json.dumps(
                    {
                        "service": SERVICE_NAME,
                        "event_type": "broadcast",
                        "type": message["type"],
                        "strategy": message.get("strategy"),
                        "recipients": recipients,
                        "failures": [
                            self._clients[s].page_id if s in self._clients else None for s in dead
                        ],
                    }
                )
            )
        for connection in dead:
            await self.disconnect(connection, reason="send failed")

    async def send_current_state(self, websocket: WebSocket) -> None:
        """Push the latest display, eyes and debug state to a newly connected page."""
        await websocket.send_text(
            json.dumps({"type": "debug_config", "enabled": self.debug_controls_enabled})
        )
        for message in (
            self.last_show,
            self.last_person,
            self.last_session,
            self.last_eyes,
            self.config,
            self.last_debug_state,
            self.last_bed_zone,
        ):
            if message is not None:
                await websocket.send_text(json.dumps(message))


def resolve_photo(
    photo_dir: Path,
    photo_id: str,
    demo_dir: Path | None = DEMO_PHOTO_DIR,
) -> Path | None:
    """Return the file backing `photo_id`, or `None` if there is not one.

    `photo_dir` (caregiver uploads) is searched first, then `demo_dir` (the
    images shipped with the service), so an uploaded photo always shadows a
    demo one. The id carries no extension, since the dashboard stores
    whatever the caregiver uploaded, so each of `PHOTO_EXTENSIONS` is tried
    in turn. Ids that do not match `PHOTO_ID_RE` are rejected before being
    joined onto any directory, so no id can point outside them.
    """
    if not PHOTO_ID_RE.match(photo_id):
        return None
    directories = [photo_dir] if demo_dir is None else [photo_dir, demo_dir]
    for directory in directories:
        for extension in PHOTO_EXTENSIONS:
            candidate = directory / f"{photo_id}{extension}"
            if candidate.is_file():
                return candidate
    return None


def resolve_voice_clip(voice_clip_dir: Path, clip_id: str) -> Path | None:
    """Resolve one dashboard-assigned WAV id without allowing path traversal."""
    if (
        "/" in clip_id
        or "\\" in clip_id
        or ".." in clip_id
        or VOICE_CLIP_ID_RE.fullmatch(clip_id) is None
    ):
        return None
    candidate = voice_clip_dir / f"{clip_id}.wav"
    return candidate if candidate.is_file() else None


def _show_to_message(event: Show) -> dict:
    return {
        "type": "show",
        "face": event.face,
        "headline": event.headline,
        "body": event.body,
        "photo_id": event.photo_id,
        "brightness": event.brightness,
    }


def _say_to_message(
    event: Say, audio_id: str | None = None, *, audio_url: str | None = None
) -> dict:
    message = {
        "type": "say",
        "text": event.text,
        "strategy": event.strategy,
        "interruptible": event.interruptible,
        "emphasis": event.emphasis,
        "session_id": event.session_id,
    }
    if audio_url is not None:
        message["audio_url"] = audio_url
    elif audio_id is not None:
        message["audio_url"] = f"/speech/{audio_id}.wav"
    return message


def _speech_started_to_message(event: SpeechStarted) -> dict:
    return {"type": "speech_started", "session_id": event.session_id}


def _debug_to_message(event) -> dict:
    fields = {
        PersonState: ("person", ("state", "confidence", "zone", "scene_note")),
        SessionState: ("session", ("phase", "goal", "strategy_index", "session_id")),
        Utterance: ("utterance", ("text", "confidence", "duration_s")),
        PoseDebug: (
            "pose",
            (
                "landmarks",
                "bbox",
                "confidence",
                "detected",
                "candidate_state",
                "state",
                "zone",
                "frame_ts",
                "latency_ms",
            ),
        ),
        Activity: ("activity", ("service", "kind", "phase", "ok", "duration_ms", "detail")),
    }
    kind, names = fields[type(event)]
    return {
        "type": kind,
        **{name: getattr(event, name) for name in names},
        "ts": event.ts.isoformat(),
    }


async def broadcast_loop(
    bus,
    manager: ConnectionManager,
    *,
    speech=None,
    consumer: str = CONSUMER,
    count: int = 10,
    block_ms: int = 100,
    max_iterations: int | None = None,
    stop_event: asyncio.Event | None = None,
    voice_clip_dir: Path | str = DEFAULT_VOICE_CLIP_DIR,
    eyes: EyesState | None = None,
) -> None:
    """Broadcast display, speech, and early voice-activity events to browsers.

    `bus` is synchronous (`nc_shared.bus.Bus` or `FakeBus`), so each blocking
    read happens in a worker thread via `asyncio.to_thread`. Runs until
    `stop_event` is set or `max_iterations` iterations have run (either can
    be left `None`/unset for the production "run forever" case).
    """
    bus.ensure_group("show", GROUP)
    bus.ensure_group("say", GROUP)
    for stream in (
        "speech_in",
        "person",
        "gaze",
        "session",
        "notify",
        "ack",
        "pose_debug",
        "activity",
        "debug",
    ):
        bus.ensure_group(stream, GROUP)

    eyes = eyes or EyesState()
    voice_clip_dir = Path(voice_clip_dir)
    iterations = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if max_iterations is not None and iterations >= max_iterations:
            return
        iterations += 1

        streams = (
            ("show", _show_to_message),
            ("say", _say_to_message),
            ("speech_in", _speech_started_to_message),
            ("person", _debug_to_message),
            ("gaze", None),
            ("session", _debug_to_message),
            ("notify", None),
            ("ack", None),
            ("pose_debug", _debug_to_message),
            ("activity", _debug_to_message),
            ("debug", None),
        )
        # One blocking read over every stream: it wakes on the first message
        # anywhere. Reading them in turn, each blocking 100 ms, held a
        # SpeechStarted or Say for up to 0.9 s (TT-4, 2026-09-24T1311-live).
        batch = await asyncio.to_thread(
            bus.read_many,
            [stream for stream, _ in streams],
            GROUP,
            consumer,
            count=count,
            block_ms=block_ms,
        )
        for stream, to_message in streams:
            messages = batch.get(stream, [])
            for msg_id, event in messages:
                if stream == "debug":
                    if isinstance(event, DebugControl) and event.source == "agent":
                        await manager.broadcast(
                            {
                                "type": "debug_state",
                                "time_offset_hours": event.time_offset_hours,
                                "force_in_bed": event.force_in_bed,
                            }
                        )
                    elif isinstance(event, BedZoneStatus):
                        await manager.broadcast(
                            {
                                "type": "bed_zone",
                                "has_bed": event.has_bed,
                                "polygon": event.polygon,
                                "calibration": event.calibration,
                                "detail": event.detail,
                            }
                        )
                    bus.ack(stream, GROUP, msg_id)
                    continue
                if stream == "session" and not isinstance(event, SessionState):
                    bus.ack(stream, GROUP, msg_id)
                    continue
                eyes_message = eyes.consume(event, msg_id)
                if eyes_message is not None:
                    await manager.broadcast(eyes_message)
                if isinstance(event, (Gaze, Notify, Ack)):
                    bus.ack(stream, GROUP, msg_id)
                    continue
                # Complete utterances share `speech_in` with the onset signal.
                if isinstance(event, Utterance):
                    message = _debug_to_message(event)
                    await manager.broadcast(message)
                    bus.ack(stream, GROUP, msg_id)
                    continue
                if isinstance(event, Say) and event.clip_id is not None:
                    clip_path = resolve_voice_clip(voice_clip_dir, event.clip_id)
                    if clip_path is None:
                        logger.warning(
                            json.dumps(
                                {
                                    "service": SERVICE_NAME,
                                    "message": "caregiver voice clip unavailable; text only",
                                    "clip_id": event.clip_id,
                                }
                            )
                        )
                        message = _say_to_message(event)
                    else:
                        message = _say_to_message(event, audio_url=f"/voice/{event.clip_id}.wav")
                elif isinstance(event, Say) and speech is not None:
                    await manager.broadcast(
                        {
                            "type": "activity",
                            "service": SERVICE_NAME,
                            "kind": "tts",
                            "phase": "start",
                            "ok": True,
                            "duration_ms": None,
                            "detail": None,
                        }
                    )
                    logger.info(
                        json.dumps(
                            {
                                "service": SERVICE_NAME,
                                "event_type": "Activity",
                                "kind": "tts",
                                "phase": "start",
                            }
                        )
                    )
                    started = time.perf_counter()
                    ok = True
                    try:
                        audio_id = await asyncio.to_thread(
                            speech.synthesize, event.text, loud=event.emphasis == "loud"
                        )
                        message = _say_to_message(event, audio_id)
                    except Exception:  # noqa: BLE001 - keep the calm visual fallback alive
                        ok = False
                        logger.exception("Piper could not synthesize a Say event")
                        bus.publish(
                            Notify(
                                source=SERVICE_NAME,
                                session_id=event.session_id,
                                level="attention",
                                title="Night Companion speech failed",
                                body="The bedside display is showing text without voice.",
                                repeat_until_ack=False,
                            )
                        )
                        message = _say_to_message(event)
                    finally:
                        duration_ms = (time.perf_counter() - started) * 1000
                        await manager.broadcast(
                            {
                                "type": "activity",
                                "service": SERVICE_NAME,
                                "kind": "tts",
                                "phase": "end",
                                "ok": ok,
                                "duration_ms": duration_ms,
                                "detail": "ok" if ok else "error",
                            }
                        )
                        logger.info(
                            json.dumps(
                                {
                                    "service": SERVICE_NAME,
                                    "event_type": "Activity",
                                    "kind": "tts",
                                    "phase": "end",
                                    "ok": ok,
                                    "duration_ms": duration_ms,
                                }
                            )
                        )
                else:
                    message = to_message(event)
                await manager.broadcast(message)
                bus.ack(stream, GROUP, msg_id)
        eyes_message = eyes.tick()
        if eyes_message is not None:
            await manager.broadcast(eyes_message)


def publish_frame(bus, message: dict, session_id: str | None = None) -> RawFrame:
    """Decode a `{"type": "frame", ...}` browser message and publish it as `RawFrame`.

    Published ungated onto `frames_raw`, not `frames`: this is a raw source
    frame, and `capture` (issue #7) is the service that owns the motion gate
    and rate limit that turn it into the `Frame` events `perceive` consumes
    (HANDOFF.md section 5).

    Raises `ValueError` (with a message safe to log) if `message` is missing
    a required field or `jpeg_b64` is not valid base64; the caller decides
    whether to disconnect or just skip the message.
    """
    try:
        jpeg = base64.b64decode(message["jpeg_b64"], validate=True)
    except (KeyError, binascii.Error) as exc:
        raise ValueError(f"malformed frame message: {exc}") from exc
    try:
        width = int(message["width"])
        height = int(message["height"])
        source_width = int(message["source_width"])
        source_height = int(message["source_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed frame message: {exc}") from exc

    if min(width, height, source_width, source_height) <= 0:
        raise ValueError("malformed frame message: dimensions must be positive")

    event = RawFrame(
        source=SERVICE_NAME,
        session_id=session_id,
        jpeg=jpeg,
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
        source_kind="browser",
    )
    bus.publish(event, maxlen=CAPPED_MAXLEN["frames_raw"])
    return event


def publish_audio_chunk(bus, message: dict, session_id: str | None = None) -> AudioChunk:
    """Decode a `{"type": "audio", ...}` browser message and publish it as `AudioChunk`.

    Raises `ValueError` (with a message safe to log) if `message` is missing
    a required field, `pcm16_b64` is not valid base64, or `sample_rate` is
    not the fixed `16000` the event schema requires.
    """
    try:
        pcm16 = base64.b64decode(message["pcm16_b64"], validate=True)
    except (KeyError, binascii.Error) as exc:
        raise ValueError(f"malformed audio message: {exc}") from exc
    try:
        sample_rate = int(message["sample_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed audio message: {exc}") from exc

    event = AudioChunk(
        source=SERVICE_NAME,
        session_id=session_id,
        pcm16=pcm16,
        sample_rate=sample_rate,
    )
    bus.publish(event, maxlen=CAPPED_MAXLEN["audio_in"])
    return event


async def handle_media_message(bus, message: dict, session_id: str | None = None) -> None:
    """Dispatch one decoded `/media` websocket message to the right publisher.

    Unknown or malformed messages are logged and dropped rather than raised,
    so one bad message from the browser never tears down the connection.
    """
    message_type = message.get("type")
    try:
        if message_type == "frame":
            await asyncio.to_thread(publish_frame, bus, message, session_id)
        elif message_type == "audio":
            await asyncio.to_thread(publish_audio_chunk, bus, message, session_id)
        else:
            logger.warning("ignoring /media message with unknown type: %r", message_type)
    except ValueError:
        logger.warning("ignoring malformed /media message of type %r", message_type)


def publish_playback(bus, message: dict) -> Activity | None:
    """Validate one bounded browser playback report and publish transient telemetry."""
    if not isinstance(message, dict) or message.get("type") != "playback":
        return None
    phase = message.get("phase")
    if not isinstance(phase, str) or phase not in PLAYBACK_PHASES:
        return None
    for name, limit in (
        ("strategy", 128),
        ("session_id", 128),
        ("audio_id", 24),
        ("detail", 160),
        ("error_name", 64),
        ("error_message", 160),
    ):
        value = message.get(name)
        if value is not None and (not isinstance(value, str) or len(value) > limit):
            return None
    latency_ms = message.get("latency_ms")
    if (
        isinstance(latency_ms, bool)
        or not isinstance(latency_ms, (int, float))
        or not math.isfinite(latency_ms)
        or not 0 <= latency_ms <= 3_600_000
    ):
        return None
    detail = message.get("detail") or message.get("error_name") or phase
    if phase == "failed" and message.get("error_name"):
        detail = message["error_name"]
    event = Activity(
        source=SERVICE_NAME,
        service=SERVICE_NAME,
        session_id=message.get("session_id"),
        kind="playback",
        phase=PLAYBACK_PHASES[phase],
        ok=phase not in {"failed", "interrupted", "no_audio"},
        duration_ms=latency_ms,
        detail=detail,
    )
    logger.info(
        json.dumps(
            {
                "service": SERVICE_NAME,
                "event_type": "playback",
                "playback_phase": phase,
                "strategy": message.get("strategy"),
                "session_id": message.get("session_id"),
                "audio_id": message.get("audio_id"),
                "latency_ms": latency_ms,
                "detail": message.get("detail"),
                "error_name": message.get("error_name"),
                "error_message": message.get("error_message"),
            }
        )
    )
    bus.publish(event, maxlen=CAPPED_MAXLEN["activity"])
    return event


def create_app(
    bus,
    photo_dir: Path | str = DEFAULT_PHOTO_DIR,
    *,
    speech=None,
    prerender_phrases: tuple[str, ...] = (),
    voice_clip_dir: Path | str = DEFAULT_VOICE_CLIP_DIR,
    night_start: str = "20:00",
    night_end: str = "07:00",
    clock_24h: bool = False,
) -> FastAPI:
    """Build the FastAPI app, wiring `bus` into the websocket broadcast loop.

    `photo_dir` is where caregiver-uploaded photos referenced by a `Show`
    event's `photo_id` are read from.
    """
    manager = ConnectionManager(
        bus, night_start=night_start, night_end=night_end, clock_24h=clock_24h
    )
    manager.debug_controls_enabled = os.getenv("EMBODIMENT_DEBUG_CONTROLS", "").lower() in {
        "true",
        "1",
        "yes",
    }
    photo_dir = Path(photo_dir)
    voice_clip_dir = Path(voice_clip_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if speech is not None and prerender_phrases:
            try:
                count = await asyncio.to_thread(speech.pre_render, prerender_phrases)
                logger.info("pre-rendered %d Piper phrases", count)
            except Exception:  # noqa: BLE001 - visual display remains the safe fallback
                logger.exception("Piper startup phrase pre-render failed")
                bus.publish(
                    Notify(
                        source=SERVICE_NAME,
                        level="attention",
                        title="Night Companion speech failed",
                        body="The bedside display started without a ready voice.",
                        repeat_until_ack=False,
                    )
                )
        task = asyncio.create_task(
            broadcast_loop(bus, manager, speech=speech, voice_clip_dir=voice_clip_dir)
        )

        async def stale_loop() -> None:
            while True:
                await asyncio.sleep(5)
                await manager.mark_stale()

        stale_task = asyncio.create_task(stale_loop())
        try:
            yield
        finally:
            task.cancel()
            stale_task.cancel()

    app = FastAPI(title="Night Companion embodiment", lifespan=lifespan)
    app.state.manager = manager
    app.state.bus = bus

    @app.middleware("http")
    async def revalidate_page(request, call_next):
        # The bedside page stays open for days; make a plain reload pick up a
        # redeployed script instead of a heuristically cached one.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/photos/{photo_id}")
    async def photo(photo_id: str) -> FileResponse:
        """Serve the caregiver photo a `Show` event referenced by `photo_id`.

        A missing photo is an ordinary state, not a failure: the caregiver
        may not have uploaded one yet, or may have deleted it. This answers
        404 and the page drops its photo layer, so the night display never
        shows a broken image.
        """
        path = resolve_photo(photo_dir, photo_id)
        if path is None:
            raise HTTPException(status_code=404, detail="photo not found")
        return FileResponse(path)

    @app.get("/speech/{audio_id}.wav")
    async def speech_audio(audio_id: str) -> FileResponse:
        """Serve one locally synthesized WAV by its opaque cache digest."""
        path = None if speech is None else speech.resolve(audio_id)
        if path is None:
            raise HTTPException(status_code=404, detail="speech audio not found")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/voice/{clip_id}.wav")
    async def voice_clip(clip_id: str) -> FileResponse:
        """Serve one validated caregiver-uploaded, consented WAV clip."""
        path = resolve_voice_clip(voice_clip_dir, clip_id)
        if path is None:
            raise HTTPException(status_code=404, detail="voice clip not found")
        return FileResponse(path, media_type="audio/wav")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket, "ws")
        try:
            await manager.send_current_state(websocket)
            while True:
                raw = await websocket.receive_text()
                if len(raw) > 2048:
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict) and message.get("type") in {
                    "hello",
                    "heartbeat",
                    "visibility",
                }:
                    await manager.update(websocket, message)
                elif isinstance(message, dict) and message.get("type") in {
                    "debug_control",
                    "calibrate_bed",
                    "reset_session",
                }:
                    kind = message["type"]
                    if not manager.debug_controls_enabled:
                        logger.warning(
                            json.dumps(
                                {
                                    "service": SERVICE_NAME,
                                    "event_type": kind,
                                    "detail": "controls disabled",
                                }
                            )
                        )
                        continue
                    if kind == "debug_control":
                        offset = message.get("time_offset_hours")
                        forced = message.get("force_in_bed")
                        if (
                            type(offset) not in (int, float)
                            or not math.isfinite(offset)
                            or type(forced) is not bool
                        ):
                            logger.warning(
                                json.dumps(
                                    {
                                        "service": SERVICE_NAME,
                                        "event_type": kind,
                                        "detail": "invalid control values",
                                    }
                                )
                            )
                            continue
                        event = DebugControl(
                            source=SERVICE_NAME,
                            time_offset_hours=max(-23, min(23, offset)),
                            force_in_bed=forced,
                        )
                    elif kind == "calibrate_bed":
                        event = CalibrateBed(source=SERVICE_NAME)
                    else:
                        event = ResetSession(source=SERVICE_NAME)
                    bus.publish(event, maxlen=100)
                    logger.warning(
                        json.dumps(
                            {
                                "service": SERVICE_NAME,
                                "event_type": type(event).__name__,
                                "time_offset_hours": getattr(event, "time_offset_hours", None),
                                "force_in_bed": getattr(event, "force_in_bed", None),
                            }
                        )
                    )
                else:
                    publish_playback(bus, message)
        except WebSocketDisconnect as exc:
            await manager.disconnect(websocket, exc.code, exc.reason)
        except Exception:
            await manager.disconnect(websocket, reason="socket error")
            raise

    @app.websocket("/media")
    async def media_endpoint(websocket: WebSocket) -> None:
        """Receive captured webcam frames and mic audio from the browser.

        The browser sends JSON text messages shaped as
        `{"type": "frame", "jpeg_b64": ..., "width": ..., "height": ...,
        "source_width": ..., "source_height": ...}` or
        `{"type": "audio", "pcm16_b64": ..., "sample_rate": 16000}`. Each is
        published onto the bus as a `RawFrame`/`AudioChunk` event; a malformed
        or unrecognised message is logged and skipped, not fatal to the
        connection (issue #28).
        """
        await manager.connect(websocket, "media")
        try:
            while True:
                raw = await websocket.receive_text()
                if len(raw) > 2_000_000:
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("ignoring non-JSON /media message")
                    continue
                if isinstance(message, dict) and message.get("type") in {
                    "hello",
                    "heartbeat",
                    "visibility",
                }:
                    if len(raw) <= 2048:
                        await manager.update(websocket, message)
                elif isinstance(message, dict):
                    await handle_media_message(bus, message)
        except WebSocketDisconnect as exc:
            await manager.disconnect(websocket, exc.code, exc.reason)
        except Exception:
            await manager.disconnect(websocket, reason="socket error")
            raise

    return app
