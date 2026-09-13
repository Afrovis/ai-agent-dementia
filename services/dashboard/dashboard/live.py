"""Live caregiver-dashboard state reconstructed from retained bus streams.

The SQLite store is the source for History.  Tonight also needs the latest
state without waiting for the store's next polling pass, and it needs the
Redis message id of a repeating notification so the caregiver can publish
the matching :class:`Ack`.  This module therefore keeps a small, text-only
projection in memory.  It never subscribes to frame or audio streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from nc_shared.events import Ack, Health, Notify, PersonState, Say, SessionState, Utterance

LIVE_STREAMS = ("session", "person", "speech_in", "say", "notify", "ack", "health")
LIVE_GROUP = "dashboard-live"
LIVE_CONSUMER = "dashboard-1"


@dataclass(frozen=True)
class ActiveAlert:
    """A repeating notification and the Redis id an ``Ack`` must reference."""

    notify_id: str
    event: Notify


@dataclass(frozen=True)
class LiveSnapshot:
    session: SessionState | None
    person: PersonState | None
    last_heard: Utterance | None
    last_said: Say | None
    alerts: tuple[ActiveAlert, ...]
    health: tuple[Health, ...]


@dataclass
class LiveState:
    """Thread-safe, bounded live projection used by dashboard request handlers."""

    _session: SessionState | None = None
    _person: PersonState | None = None
    _last_heard: Utterance | None = None
    _last_said: Say | None = None
    _alerts: dict[str, Notify] = field(default_factory=dict)
    _health: dict[str, Health] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def apply(self, msg_id: str, event: object) -> None:
        with self._lock:
            if isinstance(event, SessionState):
                self._session = event
            elif isinstance(event, PersonState):
                self._person = event
            elif isinstance(event, Utterance):
                self._last_heard = event
            elif isinstance(event, Say):
                self._last_said = event
            elif isinstance(event, Notify) and event.repeat_until_ack:
                self._alerts[msg_id] = event
            elif isinstance(event, Ack):
                self._alerts.pop(event.notify_id, None)
            elif isinstance(event, Health):
                self._health[event.service] = event

    def acknowledge(self, notify_id: str) -> None:
        with self._lock:
            self._alerts.pop(notify_id, None)

    def snapshot(self) -> LiveSnapshot:
        with self._lock:
            return LiveSnapshot(
                session=self._session,
                person=self._person,
                last_heard=self._last_heard,
                last_said=self._last_said,
                alerts=tuple(
                    ActiveAlert(notify_id=notify_id, event=event)
                    for notify_id, event in self._alerts.items()
                ),
                health=tuple(sorted(self._health.values(), key=lambda event: event.service)),
            )


def consume_live_once(
    bus,
    state: LiveState,
    *,
    consumer: str = LIVE_CONSUMER,
    count: int = 100,
    block_ms: int = 100,
) -> int:
    """Apply and acknowledge one batch from each text-only retained stream."""
    consumed = 0
    for stream in LIVE_STREAMS:
        bus.ensure_group(stream, LIVE_GROUP)
        messages = bus.read(stream, LIVE_GROUP, consumer, count=count, block_ms=block_ms)
        for msg_id, event in messages:
            state.apply(msg_id, event)
            bus.ack(stream, LIVE_GROUP, msg_id)
            consumed += 1
    return consumed
