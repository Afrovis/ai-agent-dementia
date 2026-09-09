"""FastAPI application for the `embodiment` service (issue #3).

Serves the fullscreen embodiment page (face plus big text) over HTTP/HTTPS
and pushes `Show`/`Say` events to connected browsers over a WebSocket, so a
change of face state or text appears live with no polling.

The bus-reading side is deliberately split into small, injectable pieces so
it can be tested with `nc_shared.bus.FakeBus` and no real Redis or network:

- `ConnectionManager` tracks connected websockets and the last-known `Show`
  (so a client that connects mid-night still gets the current state).
- `broadcast_loop(bus, manager, ...)` is a plain async function that reads
  the `show` and `say` streams and forwards them to `manager.broadcast`. It
  accepts `max_iterations` and/or a `stop_event` so tests can bound it
  instead of looping forever.
- `create_app(bus)` wires a `ConnectionManager` into the routes and starts
  `broadcast_loop` as a background task on startup.

The browser media bridge (issue #28) is the reverse direction: the `/media`
websocket receives JSON messages from the browser (captured webcam frames
and mic audio) and publishes them onto the bus as `Frame`/`AudioChunk`
events. `publish_frame`/`publish_audio_chunk` do the actual decode-and-
publish work and are plain functions so tests can call them directly with a
`FakeBus`, without going through a websocket at all.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from nc_shared.events import AudioChunk, Frame, Say, Show
from nc_shared.replay import CAPPED_MAXLEN

SERVICE_NAME = "embodiment"
GROUP = "embodiment"
CONSUMER = "embodiment-1"

STATIC_DIR = Path(__file__).parent / "static"

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


def _show_to_message(event: Show) -> dict:
    return {
        "type": "show",
        "face": event.face,
        "headline": event.headline,
        "body": event.body,
        "photo_id": event.photo_id,
        "brightness": event.brightness,
    }


def _say_to_message(event: Say) -> dict:
    return {
        "type": "say",
        "text": event.text,
        "strategy": event.strategy,
        "interruptible": event.interruptible,
    }


async def broadcast_loop(
    bus,
    manager: ConnectionManager,
    *,
    consumer: str = CONSUMER,
    count: int = 10,
    block_ms: int = 1000,
    max_iterations: int | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Read `show` and `say` events from `bus` and broadcast them via `manager`.

    `bus` is synchronous (`nc_shared.bus.Bus` or `FakeBus`), so each blocking
    read happens in a worker thread via `asyncio.to_thread`. Runs until
    `stop_event` is set or `max_iterations` iterations have run (either can
    be left `None`/unset for the production "run forever" case).
    """
    bus.ensure_group("show", GROUP)
    bus.ensure_group("say", GROUP)

    iterations = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if max_iterations is not None and iterations >= max_iterations:
            return
        iterations += 1

        for stream, to_message in (("show", _show_to_message), ("say", _say_to_message)):
            messages = await asyncio.to_thread(
                bus.read, stream, GROUP, consumer, count=count, block_ms=block_ms
            )
            for msg_id, event in messages:
                await manager.broadcast(to_message(event))
                bus.ack(stream, GROUP, msg_id)


def publish_frame(bus, message: dict, session_id: str | None = None) -> Frame:
    """Decode a `{"type": "frame", ...}` browser message and publish it as `Frame`.

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
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed frame message: {exc}") from exc

    event = Frame(
        source=SERVICE_NAME,
        session_id=session_id,
        jpeg=jpeg,
        width=width,
        height=height,
        source_kind="browser",
    )
    bus.publish(event, maxlen=CAPPED_MAXLEN["frames"])
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


def create_app(bus) -> FastAPI:
    """Build the FastAPI app, wiring `bus` into the websocket broadcast loop."""
    manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(broadcast_loop(bus, manager))
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
        `{"type": "frame", "jpeg_b64": ..., "width": ..., "height": ...}` or
        `{"type": "audio", "pcm16_b64": ..., "sample_rate": 16000}`. Each is
        published onto the bus as a `Frame`/`AudioChunk` event; a malformed
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
