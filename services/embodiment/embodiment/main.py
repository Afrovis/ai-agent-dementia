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
import re
from pathlib import Path

import redis
import uvicorn
from nc_shared.bus import Bus
from nc_shared.events import Notify

from embodiment.app import create_app
from embodiment.tts import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SPEED,
    DEFAULT_VOICE_MODEL,
    PiperSpeech,
    load_prerender_phrases,
)

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


def _clock_time(value: str, name: str) -> str:
    """Validate one 24-hour HH:MM boundary for the bedside clock."""
    if re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is None:
        raise ValueError(f"{name} must be HH:MM in 24-hour time")
    return value


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
    voice_clip_dir = os.environ.get("VOICE_CLIP_DIR") or "/app/data/voice-clips"
    night_start = _clock_time(
        os.environ.get("EMBODIMENT_NIGHT_START") or "20:00", "EMBODIMENT_NIGHT_START"
    )
    night_end = _clock_time(
        os.environ.get("EMBODIMENT_NIGHT_END") or "07:00", "EMBODIMENT_NIGHT_END"
    )
    clock_24h = (os.environ.get("EMBODIMENT_CLOCK_24H") or "false").lower() == "true"

    strategies_path = os.environ.get("STRATEGIES_PATH") or "/app/config/strategies.yaml"
    person_path = os.environ.get("PERSON_PATH") or "/app/config/person.yaml"

    bus = Bus(redis.Redis.from_url(redis_url))
    speech = None
    try:
        voice_model = os.environ.get("PIPER_VOICE_MODEL") or str(DEFAULT_VOICE_MODEL)
        speech_cache = os.environ.get("PIPER_CACHE_DIR") or str(DEFAULT_CACHE_DIR)
        speed = float(os.environ.get("PIPER_SPEED") or DEFAULT_SPEED)
        speech = PiperSpeech.from_model(voice_model, speech_cache, speed=speed)
    except Exception:  # noqa: BLE001 - keep the visual bedside fallback running
        logger.exception("Piper voice unavailable; starting with text-only speech")
        bus.publish(
            Notify(
                source=SERVICE_NAME,
                level="attention",
                title="Night Companion speech unavailable",
                body="The bedside display started in text-only mode.",
                repeat_until_ack=False,
            )
        )
    phrases = load_prerender_phrases(strategies_path, person_path)
    app = create_app(
        bus,
        photo_dir=photo_dir,
        speech=speech,
        prerender_phrases=phrases,
        voice_clip_dir=voice_clip_dir,
        night_start=night_start,
        night_end=night_end,
        clock_24h=clock_24h,
    )

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
