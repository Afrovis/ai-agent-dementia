"""FastAPI app factory for `web` (HANDOFF.md section 5.5).

V1 scaffolds `/healthz` and the security headers required by rule 5.5; the
submissions API, static pages and Turnstile wiring land in V5/V6.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from volunteer_web.config import WebConfig

STATIC_DIR = Path(__file__).parent / "static"

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; "
        "frame-src https://challenges.cloudflare.com; media-src 'self' blob:; "
        "img-src 'self' data:; connect-src 'self'"
    ),
    "Permissions-Policy": "camera=(self), microphone=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def create_app(config: WebConfig | None = None) -> FastAPI:
    config = config or WebConfig.from_env()
    app = FastAPI(title="volunteer-web")
    app.state.config = config

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app
