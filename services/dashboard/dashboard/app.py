"""FastAPI application for the caregiver dashboard (issues #10 and #22).

The dashboard's first real page is the Zones editor: PLAN.md section 6.2
("the bed zone, door zone, and bathroom-direction zone are drawn once in
the dashboard on a daylight frame") and section 9 ("Zones (draw bed and
door zones)"). Everything else on PLAN.md section 9's page list (Tonight,
History, Person profile, Strategies, System) is extended by issue #22 with
Tonight, retained per-night History, and caregiver acknowledgement.

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
import html
import json
import logging
import re
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from nc_shared.events import Ack

from dashboard.history import current_night_key, format_night_label, load_nights
from dashboard.live import LiveSnapshot, LiveState, consume_live_once
from dashboard.zones_store import load_existing_zones, save_zones, validate_zones

SERVICE_NAME = "dashboard"

FRAME_STREAM = "frames"
FRAME_GROUP = "dashboard"
CONSUMER = "dashboard-1"

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_DB_PATH = "data/night.db"
NOTIFY_ID_RE = re.compile(r"^[0-9]+-[0-9]+$")

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


async def live_consume_loop(
    bus,
    state: LiveState,
    *,
    max_iterations: int | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Continuously refresh the bounded, text-only Tonight projection."""
    iterations = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if max_iterations is not None and iterations >= max_iterations:
            return
        iterations += 1
        consumed = await asyncio.to_thread(consume_live_once, bus, state)
        if consumed == 0:
            await asyncio.sleep(0.5)


def _page(title: str, content: str, *, active: str) -> str:
    """Wrap dashboard content in shared navigation and the safety notice."""
    links = "".join(
        f'<a href="{path}" class="{"active" if name == active else ""}">{label}</a>'
        for name, label, path in (
            ("tonight", "Tonight", "/tonight"),
            ("history", "History", "/history"),
            ("zones", "Zones", "/zones"),
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Night Companion — {html.escape(title)}</title>
<link rel="stylesheet" href="/static/style.css"><script src="/static/htmx.min.js" defer></script>
</head><body><header class="site-header"><a class="brand" href="/tonight">Night Companion</a>
<nav aria-label="Dashboard">{links}</nav></header><main id="page">{content}</main>
<footer>Night Companion is an assistive tool, not a medical device or a substitute
for supervision.</footer>
</body></html>"""


def _format_time(value, timezone: str) -> str:
    try:
        from zoneinfo import ZoneInfo

        return value.astimezone(ZoneInfo(timezone)).strftime("%-I:%M %p")
    except (ValueError, KeyError):
        return value.strftime("%H:%M")


def _render_timeline(events, timezone: str) -> str:
    if not events:
        return '<p class="empty">No session activity has been retained for this night.</p>'
    items = []
    for event in events:
        detail = f"<p>{html.escape(event.detail)}</p>" if event.detail else ""
        items.append(
            f'<li class="timeline-{html.escape(event.kind)}">'
            f"<time>{_format_time(event.ts, timezone)}</time>"
            f"<div><strong>{html.escape(event.summary)}</strong>{detail}</div></li>"
        )
    return f'<ol class="timeline">{"".join(items)}</ol>'


def _render_live(snapshot: LiveSnapshot, night, timezone: str, *, ack_message: str = "") -> str:
    session = snapshot.session
    if session is None:
        phase = "Waiting for agent status"
        goal = "The dashboard has not received a session update yet."
        session_id = ""
    else:
        phase = session.phase.replace("_", " ").title()
        goal = f"Goal: {session.goal.replace('_', ' ')}"
        session_id = session.session_id or ""

    person = snapshot.person
    person_text = (
        f"{person.state.replace('_', ' ').title()} · {person.zone.replace('_', ' ')}"
        if person
        else "Waiting for perception status"
    )
    alerts = []
    for alert in snapshot.alerts:
        alerts.append(
            f'<article class="alert alert-{html.escape(alert.event.level)}"><div>'
            f"<strong>{html.escape(alert.event.title)}</strong>"
            f"<p>{html.escape(alert.event.body)}</p></div>"
            f'<form hx-post="/ack/{html.escape(alert.notify_id)}" '
            'hx-target="#tonight-live" hx-swap="innerHTML">'
            '<button type="submit">Acknowledge</button></form></article>'
        )
    alert_html = "".join(alerts) or '<p class="empty">No alerts awaiting acknowledgement.</p>'

    heard = "Nothing heard in this session yet."
    if snapshot.last_heard and snapshot.last_heard.session_id == session_id:
        heard = snapshot.last_heard.text
    said = "The agent has not spoken in this session yet."
    if snapshot.last_said and snapshot.last_said.session_id == session_id:
        said = snapshot.last_said.text

    events = night.events if night else ()
    message = (
        f'<p class="success" role="status">{html.escape(ack_message)}</p>' if ack_message else ""
    )
    return f"""{message}<section class="status-grid" aria-label="Live status">
<article class="status-card primary"><span>Current phase</span>
<strong>{html.escape(phase)}</strong><p>{html.escape(goal)}</p></article>
<article class="status-card"><span>Person</span>
<strong>{html.escape(person_text)}</strong></article>
</section><section><h2>Alerts</h2>{alert_html}</section>
<section class="conversation-grid"><article><h2>Last heard</h2>
<blockquote>{html.escape(heard)}</blockquote></article>
<article><h2>Last said</h2><blockquote>{html.escape(said)}</blockquote></article></section>
<section><div class="section-heading"><h2>Tonight’s timeline</h2>
<a href="/history">All nights</a></div>{_render_timeline(events, timezone)}</section>"""


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
    db_path: str | Path = DEFAULT_DB_PATH,
    timezone: str = "UTC",
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
    live_state = LiveState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        tasks = (
            asyncio.create_task(frame_consume_loop(bus, latest_frame)),
            asyncio.create_task(live_consume_loop(bus, live_state)),
        )
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()

    app = FastAPI(title="Night Companion dashboard", lifespan=lifespan)
    app.state.bus = bus
    app.state.latest_frame = latest_frame
    app.state.live_state = live_state
    app.state.password = password
    app.state.zones_path = zones_path
    app.state.db_path = db_path
    app.state.timezone = timezone

    def auth_dependency(
        credentials: HTTPBasicCredentials | None = Depends(_security),
    ) -> None:
        check_auth(app.state.password, credentials)

    def tonight_timeline():
        key = current_night_key(app.state.timezone)
        return next(
            (
                night
                for night in load_nights(app.state.db_path, timezone=app.state.timezone)
                if night.key == key
            ),
            None,
        )

    @app.get("/", include_in_schema=False)
    async def root(_: None = Depends(auth_dependency)) -> RedirectResponse:
        return RedirectResponse("/tonight", status_code=303)

    @app.get("/tonight", response_class=HTMLResponse)
    async def tonight_page(_: None = Depends(auth_dependency)) -> HTMLResponse:
        live_content = _render_live(
            app.state.live_state.snapshot(), tonight_timeline(), app.state.timezone
        )
        content = (
            '<div class="page-heading"><div><h1>Tonight</h1>'
            "<p>Live status and current session activity.</p></div>"
            '<span class="live-badge">Live</span></div><div id="tonight-live" '
            'hx-get="/partials/tonight" hx-trigger="every 3s" hx-swap="innerHTML">'
            f"{live_content}"
            "</div>"
        )
        return HTMLResponse(_page("Tonight", content, active="tonight"))

    @app.get("/partials/tonight", response_class=HTMLResponse)
    async def tonight_partial(_: None = Depends(auth_dependency)) -> HTMLResponse:
        return HTMLResponse(
            _render_live(app.state.live_state.snapshot(), tonight_timeline(), app.state.timezone)
        )

    @app.post("/ack/{notify_id}", response_class=HTMLResponse)
    async def acknowledge(notify_id: str, _: None = Depends(auth_dependency)) -> HTMLResponse:
        if not NOTIFY_ID_RE.fullmatch(notify_id):
            raise HTTPException(status_code=400, detail="invalid notification id")
        snapshot = app.state.live_state.snapshot()
        session_id = snapshot.session.session_id if snapshot.session else None
        app.state.bus.publish(Ack(source=SERVICE_NAME, session_id=session_id, notify_id=notify_id))
        app.state.live_state.acknowledge(notify_id)
        _log("notification acknowledged", session_id=session_id, notify_id=notify_id)
        return HTMLResponse(
            _render_live(
                app.state.live_state.snapshot(),
                tonight_timeline(),
                app.state.timezone,
                ack_message="Alert acknowledged. Repeating notifications will stop.",
            )
        )

    @app.get("/history", response_class=HTMLResponse)
    async def history_page(_: None = Depends(auth_dependency)) -> HTMLResponse:
        nights = load_nights(app.state.db_path, timezone=app.state.timezone)
        if nights:
            cards = "".join(
                f'<a class="night-card" href="/history/{night.key}">'
                f"<strong>{html.escape(format_night_label(night.key))}</strong>"
                f"<span>{night.session_count} session{'s' if night.session_count != 1 else ''}"
                f" · {len(night.events)} events</span></a>"
                for night in nights
            )
        else:
            cards = '<p class="empty">No session history has been retained yet.</p>'
        content = (
            "<h1>History</h1><p>Session timelines are grouped at noon, so activity "
            f'after midnight stays with the night before.</p><div class="night-list">{cards}</div>'
        )
        return HTMLResponse(_page("History", content, active="history"))

    @app.get("/history/{night_key}", response_class=HTMLResponse)
    async def history_night(night_key: str, _: None = Depends(auth_dependency)) -> HTMLResponse:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", night_key):
            raise HTTPException(status_code=404, detail="night not found")
        night = next(
            (
                item
                for item in load_nights(app.state.db_path, timezone=app.state.timezone)
                if item.key == night_key
            ),
            None,
        )
        if night is None:
            raise HTTPException(status_code=404, detail="night not found")
        plural = "s" if night.session_count != 1 else ""
        content = (
            f'<p><a href="/history">← All nights</a></p>'
            f"<h1>{html.escape(format_night_label(night.key))}</h1>"
            f"<p>{night.session_count} session{plural}</p>"
            f"{_render_timeline(night.events, app.state.timezone)}"
        )
        return HTMLResponse(_page(format_night_label(night.key), content, active="history"))

    @app.get("/zones", response_class=HTMLResponse)
    async def zones_page(_: None = Depends(auth_dependency)) -> HTMLResponse:
        return HTMLResponse(_page("Zones", (STATIC_DIR / "index.html").read_text(), active="zones"))

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
