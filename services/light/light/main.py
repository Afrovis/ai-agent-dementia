"""Consume idempotent light commands and operate the configured smart plug."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass

import redis
from nc_shared.bus import Bus
from nc_shared.events import Health, Notify

from light.backends import DisabledBackend, LightBackend, ShellyBackend, UnavailableBackend

SERVICE_NAME = "light"
LIGHT_STREAM = "light"
GROUP = "light"
HEALTH_INTERVAL_S = 30.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, **fields: object) -> None:
    logger.info(json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


@dataclass
class LightState:
    ok: bool = True
    detail: str = "path light ready"
    fault_notified: bool = False


def make_backend(env: Mapping[str, str] | None = None) -> LightBackend:
    """Build the feature-flagged local backend from documented env vars."""
    values = os.environ if env is None else env
    enabled = values.get("LIGHT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return DisabledBackend()
    backend = values.get("LIGHT_BACKEND", "shelly").strip().lower()
    if backend != "shelly":
        raise ValueError(f"unsupported LIGHT_BACKEND: {backend}")
    device_url = values.get("LIGHT_DEVICE_URL", "").strip()
    if not device_url:
        raise ValueError("LIGHT_DEVICE_URL is required when LIGHT_ENABLED=true")
    return ShellyBackend(
        device_url,
        switch_id=int(values.get("LIGHT_SWITCH_ID", "0")),
        timeout=float(values.get("LIGHT_TIMEOUT_SECONDS", "3")),
    )


def _publish_fault(bus, state: LightState) -> None:
    if state.fault_notified:
        return
    bus.publish(
        Notify(
            source=SERVICE_NAME,
            level="attention",
            title="Path light unavailable",
            body="The hallway smart plug did not accept a light command; please check it.",
            repeat_until_ack=False,
        )
    )
    state.fault_notified = True


def consume_once(
    bus,
    backend: LightBackend,
    state: LightState,
    *,
    consumer: str = "light-1",
    count: int = 10,
) -> int:
    """Apply and acknowledge one batch of absolute-state light commands."""
    bus.ensure_group(LIGHT_STREAM, GROUP)
    handled = 0
    for msg_id, event in bus.read(LIGHT_STREAM, GROUP, consumer, count=count, block_ms=100):
        handled += 1
        if backend.set_state(event.state == "on"):
            state.ok = True
            state.detail = f"hallway light {event.state}"
            state.fault_notified = False
            _log("applied LightCommand", event_type="LightCommand", state=event.state)
        else:
            state.ok = False
            state.detail = "smart plug command failed"
            _publish_fault(bus, state)
            _log("LightCommand failed", event_type="LightCommand", state=event.state)
        # The command is an absolute desired state, so retrying it is safe,
        # but an unreachable plug must not pin the Redis consumer forever.
        bus.ack(LIGHT_STREAM, GROUP, msg_id)
    return handled


def publish_health(bus, state: LightState) -> None:
    bus.publish(Health(source=SERVICE_NAME, service=SERVICE_NAME, ok=state.ok, detail=state.detail))


def run() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    bus = Bus(redis.Redis.from_url(redis_url))
    state = LightState()
    try:
        backend = make_backend()
    except (TypeError, ValueError) as exc:
        backend = UnavailableBackend()
        state.ok = False
        state.detail = str(exc)
        _publish_fault(bus, state)

    _log("light starting", redis_url=redis_url, backend=type(backend).__name__)
    last_health = 0.0
    while True:
        consume_once(bus, backend, state)
        now = time.monotonic()
        if now - last_health >= HEALTH_INTERVAL_S:
            publish_health(bus, state)
            last_health = now
        time.sleep(0.2)


if __name__ == "__main__":
    run()
