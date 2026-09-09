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
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from nc_shared.events import Say, Show

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

    return app
