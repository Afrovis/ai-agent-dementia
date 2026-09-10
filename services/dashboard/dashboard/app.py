"""FastAPI application for the `dashboard` service (issue #10).

The dashboard's first real page is the Zones editor: PLAN.md section 6.2
("the bed zone, door zone, and bathroom-direction zone are drawn once in
the dashboard on a daylight frame") and section 9 ("Zones (draw bed and
door zones)"). Everything else on PLAN.md section 9's page list (Tonight,
History, Person profile, Strategies, System) is later issues; this module
only implements `/zones`.

Two things make this route more sensitive than the rest of the dashboard
will be, so they are both non-negotiable here rather than left for later:

- `GET /zones/frame.jpg` serves a live image from inside the bedroom. The
  dashboard subscribes to the `frames` stream under its own consumer group
  (`FRAME_GROUP`) and keeps only the single most recent frame in memory --
  never written to disk, per HANDOFF.md rule 2. `frame_consume_loop` is the
  background task that keeps it current; `consume_frame_once` is the
  single-iteration function under it, same shape as
  `perceive.main.run_once` and `embodiment.app.broadcast_loop`, so it can
  be driven directly in tests against a `FakeBus` with no event loop
  timing to race.
- Every route requires HTTP Basic auth against `DASHBOARD_PASSWORD`
  (PLAN.md section 9: "protected by a single password"). `check_auth` fails
  closed: if the password is not configured, every route answers 503
  naming the variable, never serving a bedroom frame or accepting a zones
  write with no password set at all.

`POST /zones` validates a caregiver's drawn polygons server-side
(`dashboard.zones_store.validate_zones` -- never trust the browser) and
writes them atomically to `config/zones.yaml`
(`dashboard.zones_store.save_zones`). `perceive` reads that file once at
startup and does not reread it (`perceive.main.run`), so the save response
tells the caregiver to restart `perceive` rather than implying the change
is already live.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from dashboard.zones_store import load_existing_zones, save_zones, validate_zones

SERVICE_NAME = "dashboard"

FRAME_STREAM = "frames"
FRAME_GROUP = "dashboard"
CONSUMER = "dashboard-1"

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

# `auto_error=False` so a request with no `Authorization` header at all
# reaches `check_auth` instead of FastAPI short-circuiting to a 401 --
# with no `DASHBOARD_PASSWORD` configured the right answer is 503 even
# with no credentials offered.
_security = HTTPBasic(auto_error=False)


def _log(message: str, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4).

    Never pass frame bytes here: the hard rule that camera frames are
    never logged as bytes applies just as much to the dashboard's frame
    preview as it does to `capture` and `perceive`.
    """
    logger.info(json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


class LatestFrame:
    """Holds the single most recent `Frame` jpeg, in memory only.

    No history, no disk: `set` simply replaces whatever was there, so at
    most one frame's worth of bytes ever exists in this process, and none
    of it survives a restart. This is the "camera preview for setup only"
    PLAN.md section 9 allows, and nothing more.
    """

    def __init__(self) -> None:
        self._jpeg: bytes | None = None

    def set(self, jpeg: bytes) -> None:
        self._jpeg = jpeg

    def get(self) -> bytes | None:
        return self._jpeg


def consume_frame_once(
    bus,
    latest: LatestFrame,
    *,
    consumer: str = CONSUMER,
    count: int = 10,
    block_ms: int = 200,
) -> int:
    """Read any pending `Frame` messages and remember only the most recent one.

    Acks every message read, even ones whose bytes are immediately
    discarded in favour of a later one in the same batch: nothing else
    reads `frames` against the `dashboard` consumer group, so there is
    nothing to keep them pending for, matching `perceive.main.run_once`'s
    reasoning for acking as soon as a frame has been looked at. Returns the
    number of messages consumed, mainly so tests can assert something
    happened without reaching into `latest`.
    """
    messages = bus.read(FRAME_STREAM, FRAME_GROUP, consumer, count=count, block_ms=block_ms)
    for msg_id, _event in messages:
        bus.ack(FRAME_STREAM, FRAME_GROUP, msg_id)
    if messages:
        _msg_id, frame = messages[-1]
        latest.set(frame.jpeg)
    return len(messages)


async def frame_consume_loop(
    bus,
    latest: LatestFrame,
    *,
    consumer: str = CONSUMER,
    count: int = 10,
    block_ms: int = 1000,
    max_iterations: int | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Repeatedly call `consume_frame_once` in a worker thread.

    Same shape as `embodiment.app.broadcast_loop`: `bus` is synchronous, so
    each blocking read happens via `asyncio.to_thread`, and the loop
    accepts `max_iterations`/`stop_event` so tests can bound it instead of
    running forever.
    """
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)

    iterations = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if max_iterations is not None and iterations >= max_iterations:
            return
        iterations += 1
        await asyncio.to_thread(
            consume_frame_once, bus, latest, consumer=consumer, count=count, block_ms=block_ms
        )


def check_auth(password: str | None, credentials: HTTPBasicCredentials | None) -> None:
    """Raise the right `HTTPException` for `password`/`credentials`, or return.

    Fails closed: an unset or empty `password` always raises 503 naming
    `DASHBOARD_PASSWORD`, before credentials are even considered -- there
    is no way to reach a dashboard route, including the frame preview, by
    supplying *any* credentials while the app is unconfigured. Any
    username is accepted; PLAN.md section 9 describes one password, not
    per-caregiver accounts, so only the password is checked, and with
    `secrets.compare_digest` rather than `==` to avoid a timing side
    channel on it.
    """
    if not password:
        raise HTTPException(
            status_code=503,
            detail=(
                "DASHBOARD_PASSWORD is not set; the dashboard is disabled until it is configured"
            ),
        )
    # Compare as bytes: `secrets.compare_digest` rejects str arguments that
    # are not ASCII-only, so a caregiver who picks a password with an
    # accented character in it, or a request that sends one, would raise
    # TypeError and answer 500 instead of 401.
    if credentials is None or not secrets.compare_digest(
        credentials.password.encode("utf-8"), password.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


def create_app(
    bus,
    *,
    password: str | None = None,
    zones_path: str | Path | None = None,
) -> FastAPI:
    """Build the FastAPI app, wiring `bus` into the frame-consuming background task.

    `password` is the caregiver's `DASHBOARD_PASSWORD` (or `None`/empty to
    exercise the fail-closed 503 path). `zones_path` overrides where
    `config/zones.yaml` is read from and written to, same resolution order
    as `perceive.zones.load_zones` (see `dashboard.zones_store`).

    Raises `ValueError` on a non-ASCII password. HTTP Basic carries
    credentials as ASCII, and FastAPI's `HTTPBasic` answers 401 on anything
    else before any of this module's code runs, so such a password can
    never authenticate however carefully it is compared. An unset password
    is a not-yet-configured system and gets a helpful 503 per route; a
    non-ASCII one is a misconfiguration that will never work, and refusing
    at startup is the loud failure HANDOFF.md rule 4 asks for rather than
    leaving a caregiver typing the right password into an endless 401.
    """
    if password and not password.isascii():
        raise ValueError(
            "DASHBOARD_PASSWORD must contain only ASCII characters; "
            "HTTP Basic authentication cannot carry anything else"
        )

    latest_frame = LatestFrame()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(frame_consume_loop(bus, latest_frame))
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Night Companion dashboard", lifespan=lifespan)
    app.state.bus = bus
    app.state.latest_frame = latest_frame
    app.state.password = password
    app.state.zones_path = zones_path

    def auth_dependency(
        credentials: HTTPBasicCredentials | None = Depends(_security),
    ) -> None:
        check_auth(app.state.password, credentials)

    @app.get("/zones", response_class=HTMLResponse)
    async def zones_page(_: None = Depends(auth_dependency)) -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text())

    @app.get("/static/style.css")
    async def static_style(_: None = Depends(auth_dependency)) -> FileResponse:
        return FileResponse(STATIC_DIR / "style.css", media_type="text/css")

    @app.get("/static/script.js")
    async def static_script(_: None = Depends(auth_dependency)) -> FileResponse:
        return FileResponse(STATIC_DIR / "script.js", media_type="application/javascript")

    @app.get("/static/htmx.min.js")
    async def static_htmx(_: None = Depends(auth_dependency)) -> FileResponse:
        # Vendored rather than loaded from a CDN: this is a LAN-only device
        # that may have no internet access at 3am, and PLAN.md's "no cloud
        # account" spirit extends to not depending on one to render the
        # editor page at all.
        return FileResponse(STATIC_DIR / "htmx.min.js", media_type="application/javascript")

    @app.get("/zones/frame.jpg")
    async def frame_jpg(_: None = Depends(auth_dependency)) -> Response:
        """Serve the most recent `Frame` in memory, or 503 if none has arrived.

        No frame yet is the ordinary state on a fresh install: nobody has
        opened the bedside embodiment page with camera permission granted,
        so `capture`/`perceive` have nothing to publish. 503 with a plain
        explanation, not a broken image.
        """
        jpeg = latest_frame.get()
        if jpeg is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "no camera frame received yet -- open the bedside embodiment "
                    "page and grant camera permission, then reload this page"
                ),
            )
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/zones/current")
    async def zones_current(_: None = Depends(auth_dependency)) -> dict:
        """Return the zones currently in effect, for the editor to pre-populate."""
        return load_existing_zones(app.state.zones_path)

    @app.post("/zones", response_class=HTMLResponse)
    async def zones_save(
        zones_json: str = Form(...),
        _: None = Depends(auth_dependency),
    ) -> HTMLResponse:
        """Validate and persist a caregiver's drawn zones (HTMX form submission).

        The browser sends the three polygons as a single JSON string in the
        `zones_json` form field (`services/dashboard/dashboard/static/script.js`
        keeps a hidden input's value in sync with the in-progress drawing),
        since a plain HTML form has no native way to carry nested per-point
        data. Validated server-side regardless -- `validate_zones` never
        trusts what the browser sent.
        """
        try:
            payload = json.loads(zones_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=["zones_json must be valid JSON"]) from None

        parsed, errors = validate_zones(payload)
        if errors:
            raise HTTPException(status_code=400, detail=errors)

        target = save_zones(parsed, app.state.zones_path)
        _log(
            "saved zones",
            path=str(target),
            zone_point_counts={name: len(points) for name, points in parsed.items()},
        )
        return HTMLResponse(
            "<p>Zones saved. <strong>perceive</strong> reads <code>zones.yaml</code> once "
            "at startup and does not reload it automatically -- restart the "
            "<code>perceive</code> service (<code>docker compose restart perceive</code>) "
            "for this change to take effect.</p>"
        )

    return app
