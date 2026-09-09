"""Entry point for the `embodiment` service (issue #3).

Fullscreen embodiment web page: animated face, big text, TTS playback
(HANDOFF.md section 4). Serves the FastAPI app in `embodiment.app` over
HTTPS with a mkcert LAN certificate when one is available, since
`getUserMedia` (needed by the browser media bridge, issue #28) only works
on localhost or over HTTPS. Falls back to plain HTTP on the same port for
local development without mkcert.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import redis
import uvicorn
from nc_shared.bus import Bus

from embodiment.app import create_app

SERVICE_NAME = "embodiment"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4)."""
    logger.info(json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


def ssl_kwargs_for(cert_file: str, cert_key: str) -> dict[str, str]:
    """Return `uvicorn.run` ssl kwargs if both `cert_file` and `cert_key` exist.

    Returns an empty dict (plain HTTP) otherwise, so local development
    without mkcert still works (HANDOFF.md section 9).
    """
    if Path(cert_file).exists() and Path(cert_key).exists():
        return {"ssl_certfile": cert_file, "ssl_keyfile": cert_key}
    return {}


def run() -> None:
    """Start the embodiment FastAPI app, over HTTPS when a certificate exists."""
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    port = int(os.environ.get("EMBODIMENT_PORT", "8443"))
    cert_file = os.environ.get("CERT_FILE", "data/certs/lan.pem")
    cert_key = os.environ.get("CERT_KEY", "data/certs/lan-key.pem")
    # `or` rather than a `get` default: compose passes a variable that is
    # missing from `.env` through as an empty string, and an empty path
    # would silently resolve no photos at all.
    photo_dir = os.environ.get("PHOTO_DIR") or "data/photos"

    bus = Bus(redis.Redis.from_url(redis_url))
    app = create_app(bus, photo_dir=photo_dir)

    ssl_kwargs = ssl_kwargs_for(cert_file, cert_key)
    if ssl_kwargs:
        _log("starting over HTTPS", port=port, cert_file=cert_file)
    else:
        _log(
            "no mkcert certificate found, falling back to plain HTTP "
            "(getUserMedia will not work off localhost)",
            port=port,
            cert_file=cert_file,
            cert_key=cert_key,
        )

    uvicorn.run(app, host="0.0.0.0", port=port, **ssl_kwargs)  # noqa: S104 - LAN-only device


if __name__ == "__main__":
    run()
