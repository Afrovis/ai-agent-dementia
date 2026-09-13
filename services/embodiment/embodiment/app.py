"""FastAPI application for the `embodiment` service (issue #3).

Serves the fullscreen embodiment page (face plus big text) over HTTP/HTTPS
and pushes `Show`/`Say`/`SpeechStarted` events to connected browsers over a
WebSocket, so display changes and barge-in arrive live with no polling.

The bus-reading side is deliberately split into small, injectable pieces so
it can be tested with `nc_shared.bus.FakeBus` and no real Redis or network:

- `ConnectionManager` tracks connected websockets and the last-known `Show`
  (so a client that connects mid-night still gets the current state).
- `broadcast_loop(bus, manager, ...)` is a plain async function that reads
  `show`, `say`, and the barge-in signal on `speech_in`, then forwards browser
  messages to `manager.broadcast`. It accepts `max_iterations` and/or a
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
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from nc_shared.events import AudioChunk, Notify, RawFrame, Say, Show, SpeechStarted
from nc_shared.replay import CAPPED_MAXLEN

SERVICE_NAME = "embodiment"
GROUP = "embodiment"
CONSUMER = "embodiment-1"

STATIC_DIR = Path(__file__).parent / "static"

# Caregiver-uploaded photos live under the gitignored `data/` tree
# (HANDOFF.md section 6), so a fresh install legitimately has none.
DEFAULT_PHOTO_DIR = Path("data/photos")
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Placeholder images shipped with the service, so the fake agent's `demo_*`
# photo ids resolve on a fresh checkout. Searched only after `photo_dir`, so
# a real uploaded photo always wins over a demo one of the same name.
DEMO_PHOTO_DIR = Path(__file__).parent / "demo_photos"

# A `photo_id` is an opaque id the dashboard assigns on upload. Restricting
# it to this character set is what keeps a crafted id such as
# `../../etc/passwd` from escaping the photo directory.
PHOTO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


class ConnectionManager:
    """Tracks connected websocket clients and the last-known `Show` state."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self.last_show: dict | None = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept `websocket` and register it for future broadcasts."""
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove `websocket` from the connected set, if present."""
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        """Send `message` as JSON to every connected client, dropping dead ones.

        Also remembers the message if it is a `show` event, so a client that
        connects later can be caught up in `send_current_state`.
        """
        if message.get("type") == "show":
            self.last_show = message
        dead: list[WebSocket] = []
        for connection in self._connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:  # noqa: BLE001 - best-effort broadcast
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)

    async def send_current_state(self, websocket: WebSocket) -> None:
        """Push the last-known `Show` state to a newly connected `websocket`."""
        if self.last_show is not None:
            await websocket.send_text(json.dumps(self.last_show))


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


def _show_to_message(event: Show) -> dict:
    return {
        "type": "show",
        "face": event.face,
        "headline": event.headline,
        "body": event.body,
        "photo_id": event.photo_id,
        "brightness": event.brightness,
    }


def _say_to_message(event: Say, audio_id: str | None = None) -> dict:
    message = {
        "type": "say",
        "text": event.text,
        "strategy": event.strategy,
        "interruptible": event.interruptible,
        "session_id": event.session_id,
    }
    if audio_id is not None:
        message["audio_url"] = f"/speech/{audio_id}.wav"
    return message


def _speech_started_to_message(event: SpeechStarted) -> dict:
    return {"type": "speech_started", "session_id": event.session_id}


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
) -> None:
    """Broadcast display, speech, and early voice-activity events to browsers.

    `bus` is synchronous (`nc_shared.bus.Bus` or `FakeBus`), so each blocking
    read happens in a worker thread via `asyncio.to_thread`. Runs until
    `stop_event` is set or `max_iterations` iterations have run (either can
    be left `None`/unset for the production "run forever" case).
    """
    bus.ensure_group("show", GROUP)
    bus.ensure_group("say", GROUP)
    bus.ensure_group("speech_in", GROUP)

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
        )
        for stream, to_message in streams:
            messages = await asyncio.to_thread(
                bus.read, stream, GROUP, consumer, count=count, block_ms=block_ms
            )
            for msg_id, event in messages:
                # Complete Utterance events share `speech_in` with the onset
                # signal but are for the agent, not the bedside browser.
                if stream == "speech_in" and not isinstance(event, SpeechStarted):
                    bus.ack(stream, GROUP, msg_id)
                    continue
                if isinstance(event, Say) and speech is not None:
                    try:
                        audio_id = await asyncio.to_thread(speech.synthesize, event.text)
                        message = _say_to_message(event, audio_id)
                    except Exception:  # noqa: BLE001 - keep the calm visual fallback alive
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
                else:
                    message = to_message(event)
                await manager.broadcast(message)
                bus.ack(stream, GROUP, msg_id)


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


def create_app(
    bus,
    photo_dir: Path | str = DEFAULT_PHOTO_DIR,
    *,
    speech=None,
    prerender_phrases: tuple[str, ...] = (),
) -> FastAPI:
    """Build the FastAPI app, wiring `bus` into the websocket broadcast loop.

    `photo_dir` is where caregiver-uploaded photos referenced by a `Show`
    event's `photo_id` are read from.
    """
    manager = ConnectionManager()
    photo_dir = Path(photo_dir)

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
        task = asyncio.create_task(broadcast_loop(bus, manager, speech=speech))
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Night Companion embodiment", lifespan=lifespan)
    app.state.manager = manager
    app.state.bus = bus

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

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await manager.connect(websocket)
        await manager.send_current_state(websocket)
        try:
            while True:
                # The page does not send anything meaningful; just keep the
                # connection open and notice disconnects.
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

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
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("ignoring non-JSON /media message")
                    continue
                await handle_media_message(bus, message)
        except WebSocketDisconnect:
            pass

    return app
