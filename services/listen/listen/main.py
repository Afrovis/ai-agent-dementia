"""Placeholder entry point for the `listen` service.

Voice activity detection and speech to text, publishes Utterance events (HANDOFF.md section 4).
Part of M3.

This module is a stub: it logs that it is not implemented yet, publishes
one `Health(ok=False, detail="placeholder")` event so the bus and `store`
pipeline can be exercised end to end, and then idles so the container
stays up under `docker compose up`.
"""

from __future__ import annotations

import json
import logging
import os
import time

import redis
from nc_shared.bus import Bus
from nc_shared.events import Health

SERVICE_NAME = "listen"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4)."""
    logger.info(json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


def emit_placeholder_health(bus) -> None:
    """Publish one `Health(ok=False, detail="placeholder")` event on `bus`."""
    health = Health(source=SERVICE_NAME, service=SERVICE_NAME, ok=False, detail="placeholder")
    bus.publish(health)
    _log("published placeholder Health event", event_type="Health")


def run() -> None:
    """Start the placeholder: log a warning, emit one Health event, then idle."""
    _log(f"{SERVICE_NAME} is not implemented yet")
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            bus = Bus(redis.Redis.from_url(redis_url))
            emit_placeholder_health(bus)
        except redis.RedisError:
            _log("redis unavailable, skipping Health event")
    while True:
        time.sleep(30)


if __name__ == "__main__":
    run()
