"""Entry point for the `notify` service: sends caregiver alerts.

`notify` consumes the `Notify` event stream (HANDOFF.md section 5) through
a Redis consumer group, same pattern as `store` and `embodiment`, and hands
each event to a pluggable `notify.backends.NotifyBackend` (ntfy by
default). HANDOFF.md rule 4, "fail loud to the caregiver", and PLAN.md
section 9 require that `critical` notifications repeat every 60 s until
acknowledged.

Acknowledgement mechanism (deviation worth flagging explicitly): the only
event that references a notification's identity is
`nc_shared.events.Ack(notify_id)`. `Notify` itself carries no id field, so
`notify_id` is taken to be the Redis stream message id `Bus.publish`
returns for the `Notify` message (the one identifier that actually exists
today). There is no producer of `Ack` yet — the dashboard/caregiver-app
that lets a caregiver acknowledge an alert is a later issue — so this
module only *consumes* `Ack` events from the `ack` stream if and when they
show up; until then, critical notifications simply keep repeating, which
matches "fail loud" being the safe default.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import redis
from nc_shared.bus import Bus
from nc_shared.events import Notify

from notify.backends import DryRunBackend, LoggingBackend, NotifyBackend, NtfyBackend

SERVICE_NAME = "notify"
GROUP = "notify"
NOTIFY_STREAM = "notify"
ACK_STREAM = "ack"

# How often a `critical` notification with `repeat_until_ack=True` resends,
# per PLAN.md section 9 ("critical ... repeats every 60 s until
# acknowledged").
CRITICAL_REPEAT_INTERVAL_S = 60.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def _log(message: str, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4)."""
    logger.info(json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


@dataclass
class _Outstanding:
    """A critical notification awaiting acknowledgement."""

    event: Notify
    next_due_at: float


@dataclass
class NotifyState:
    """In-memory state for the consume loop: outstanding critical notifications.

    Keyed by the Redis stream message id of the original `Notify` message,
    which doubles as the `notify_id` an `Ack` event refers back to. Kept as
    an explicit object (rather than module globals) so tests can construct
    a fresh one and inspect it directly.
    """

    outstanding: dict[str, _Outstanding] = field(default_factory=dict)

    def ack(self, notify_id: str) -> bool:
        """Acknowledge `notify_id`, stopping its repeats. Returns True if it was tracked."""
        return self.outstanding.pop(notify_id, None) is not None


def consume_once(
    bus,
    backend: NotifyBackend,
    state: NotifyState,
    consumer: str = "notify-1",
    count: int = 10,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run one pass: new `Notify` events, new `Ack` events, then due repeats.

    Returns the number of `backend.send` calls made, for tests and idle
    detection in `run`. `clock` is injectable so tests can fast-forward
    "time" for the repeat check without a real sleep.
    """
    sent = 0
    bus.ensure_group(NOTIFY_STREAM, GROUP)
    bus.ensure_group(ACK_STREAM, GROUP)

    for msg_id, event in bus.read(NOTIFY_STREAM, GROUP, consumer, count=count, block_ms=100):
        if backend.send(event.level, event.title, event.body):
            sent += 1
        else:
            _log("backend send failed", notify_id=msg_id, level=event.level)
        if event.level == "critical" and event.repeat_until_ack:
            state.outstanding[msg_id] = _Outstanding(
                event=event, next_due_at=clock() + CRITICAL_REPEAT_INTERVAL_S
            )
        bus.ack(NOTIFY_STREAM, GROUP, msg_id)

    for msg_id, ack_event in bus.read(ACK_STREAM, GROUP, consumer, count=count, block_ms=100):
        if state.ack(ack_event.notify_id):
            _log("critical notification acknowledged", notify_id=ack_event.notify_id)
        bus.ack(ACK_STREAM, GROUP, msg_id)

    now = clock()
    for notify_id, item in list(state.outstanding.items()):
        if item.next_due_at <= now:
            event = item.event
            if backend.send(event.level, event.title, event.body):
                sent += 1
            item.next_due_at = now + CRITICAL_REPEAT_INTERVAL_S

    return sent


def ack(state: NotifyState, notify_id: str) -> bool:
    """Acknowledge an outstanding critical notification.

    Exposed as a plain function so a future dashboard/caregiver-app can
    call it directly (or, once that issue lands, publish an `Ack` event
    that `consume_once` picks up on the `ack` stream instead).
    """
    return state.ack(notify_id)


def make_backend(env: Mapping[str, str] | None = None) -> NotifyBackend:
    """Build the configured backend from env vars.

    `NTFY_URL` is the full topic URL, e.g. `https://ntfy.sh/my-topic` or a
    self-hosted server's topic URL (HANDOFF.md: "ntfy by default"). If
    unset, falls back to `LoggingBackend` so local dev without a topic
    configured still runs (and logs) instead of crashing.
    """
    values = os.environ if env is None else env
    dry_run_value = values.get("DRY_RUN", "false").strip().lower()
    if dry_run_value not in TRUE_VALUES | FALSE_VALUES:
        raise ValueError("DRY_RUN must be true or false")
    dry_run = dry_run_value in TRUE_VALUES
    if dry_run:
        _log("dry run enabled, suppressing all outbound notifications")
        return DryRunBackend()
    ntfy_url = values.get("NTFY_URL")
    if ntfy_url:
        return NtfyBackend(ntfy_url)
    _log("NTFY_URL not set, using LoggingBackend")
    return LoggingBackend()


def run() -> None:
    """Loop forever, consuming `Notify` events and delivering/repeating alerts."""
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    try:
        backend = make_backend()
    except ValueError as exc:
        _log("refusing to start", error=str(exc))
        raise SystemExit(1) from exc
    _log(
        "notify starting",
        redis_url=redis_url,
        backend=type(backend).__name__,
        dry_run=isinstance(backend, DryRunBackend),
    )

    bus = Bus(redis.Redis.from_url(redis_url))
    state = NotifyState()

    while True:
        consume_once(bus, backend, state)
        time.sleep(1)


__all__ = ["NotifyState", "ack", "consume_once", "make_backend", "run"]

if __name__ == "__main__":
    run()
