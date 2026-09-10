"""Redis-streams bus wrapper with at-least-once delivery.

Events are pydantic models (see :mod:`nc_shared.events`). On publish, an
event is serialised to a JSON string via ``model_dump_json`` (bytes fields
such as ``Frame.jpeg`` and ``AudioChunk.pcm16`` use the
``nc_shared.events.BytesAsBase64`` type, so they round-trip as base64 text
inside that JSON string -- there is no separate binary encoding step here).
The JSON string is stored as the ``data`` field of an XADD entry, alongside an
``event_type`` field naming the concrete event class, so a reader can look
up the right pydantic model in ``EVENT_TYPES`` and reconstruct it.

Consumer groups (``XREADGROUP`` / ``XACK``) give at-least-once delivery: a
message stays pending until it is acked, so a crashed or slow consumer will
have it redelivered (by this consumer or another one that claims it).

Two classes share the same public interface (``publish``, ``ensure_group``,
``read``, ``ack``):

- :class:`Bus` talks to a real Redis server.
- :class:`FakeBus` is an in-memory stand-in with no external dependencies,
  for services and tests to run with no Redis, per HANDOFF.md section 4.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import redis

from nc_shared.events import EVENT_STREAMS, EVENT_TYPES, BaseEvent


class Bus:
    """A thin wrapper over a synchronous `redis.Redis` client for streams."""

    def __init__(self, client: redis.Redis) -> None:
        """Wrap an existing `redis.Redis` client.

        The client must not use `decode_responses=True`: this wrapper reads
        and writes raw JSON text itself and decodes fields as needed.
        """
        self._client = client

    def publish(self, event: BaseEvent, maxlen: int | None = None) -> str:
        """Publish `event` to its configured stream, returning the message id.

        `maxlen` caps the stream with approximate trimming (`XADD ... MAXLEN
        ~ maxlen`), as required for the capped `frames` and `audio_in`
        streams (HANDOFF.md: `MAXLEN ~ 50`). Leave it `None` for streams that
        are persisted to SQLite by `store` instead.
        """
        stream = EVENT_STREAMS[type(event)]
        fields = {
            "event_type": type(event).__name__,
            "data": event.model_dump_json(),
        }
        if maxlen is not None:
            msg_id = self._client.xadd(stream, fields, maxlen=maxlen, approximate=True)
        else:
            msg_id = self._client.xadd(stream, fields)
        return msg_id.decode() if isinstance(msg_id, bytes) else msg_id

    def ensure_group(self, stream: str, group: str) -> None:
        """Create consumer group `group` on `stream` if it does not exist yet."""
        try:
            self._client.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def read(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, BaseEvent]]:
        """Read up to `count` new messages for `consumer` in `group`.

        Returns a list of `(msg_id, event)` pairs. Messages are delivered but
        not removed from the stream; call `ack` once they are fully handled,
        otherwise they stay pending and will be redelivered.
        """
        response = self._client.xreadgroup(
            group, consumer, {stream: ">"}, count=count, block=block_ms
        )
        results: list[tuple[str, BaseEvent]] = []
        for _stream_name, messages in response or []:
            for msg_id, fields in messages:
                results.append((_decode(msg_id), _event_from_fields(fields)))
        return results

    def ack(self, stream: str, group: str, msg_id: str) -> None:
        """Acknowledge `msg_id` in `group`, removing it from the pending list."""
        self._client.xack(stream, group, msg_id)


def _decode(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _event_from_fields(fields: dict) -> BaseEvent:
    decoded = {_decode(k): _decode(v) for k, v in fields.items()}
    event_cls = EVENT_TYPES[decoded["event_type"]]
    return event_cls.model_validate_json(decoded["data"])


@dataclass
class _StreamEntry:
    """One message stored in a `FakeBus` stream."""

    msg_id: str
    event_type: str
    data: str


@dataclass
class _GroupState:
    """Per-consumer-group read cursor and pending set for a `FakeBus` stream."""

    next_index: int = 0
    pending: dict[str, _StreamEntry] = field(default_factory=dict)


class FakeBus:
    """An in-memory stand-in for `Bus` with the same public methods.

    Backed by plain Python lists and dicts, so it requires no Redis and no
    network access, matching HANDOFF.md's testing convention that every
    service must be testable with no hardware and no Redis. Consumer groups
    track their own read cursor and pending (unacked) messages, so an unacked
    message is still visible to redelivery logic even after `read` has
    returned it once.
    """

    def __init__(self) -> None:
        self._streams: dict[str, list[_StreamEntry]] = {}
        self._groups: dict[tuple[str, str], _GroupState] = {}
        self._id_counter = itertools.count(1)

    def publish(self, event: BaseEvent, maxlen: int | None = None) -> str:
        """Append `event` to its configured stream, returning the message id.

        Trimming a capped stream shifts every surviving entry down the list,
        so each group's `next_index` -- a positional cursor, unlike Redis's
        opaque message ids -- is rebased by the number of entries dropped.
        Without that, a consumer of a capped stream (`frames`, `frames_raw`,
        `audio_in`) goes permanently deaf the moment the stream first
        reaches `maxlen`: its cursor sits past the end of a list that never
        grows again.
        """
        stream = EVENT_STREAMS[type(event)]
        entries = self._streams.setdefault(stream, [])
        msg_id = f"{next(self._id_counter)}-0"
        entries.append(
            _StreamEntry(
                msg_id=msg_id,
                event_type=type(event).__name__,
                data=event.model_dump_json(),
            )
        )
        if maxlen is not None and len(entries) > maxlen:
            dropped = len(entries) - maxlen
            del entries[:dropped]
            for (entry_stream, _group), state in self._groups.items():
                if entry_stream == stream:
                    state.next_index = max(0, state.next_index - dropped)
        return msg_id

    def ensure_group(self, stream: str, group: str) -> None:
        """Create consumer group `group` on `stream` if it does not exist yet."""
        self._streams.setdefault(stream, [])
        self._groups.setdefault((stream, group), _GroupState())

    def read(
        self,
        stream: str,
        group: str,
        consumer: str,  # noqa: ARG002 - kept for interface parity with Bus
        count: int = 10,
        block_ms: int = 1000,  # noqa: ARG002 - no blocking needed in-memory
    ) -> list[tuple[str, BaseEvent]]:
        """Return up to `count` new messages for `group`, marking them pending."""
        state = self._groups.setdefault((stream, group), _GroupState())
        entries = self._streams.get(stream, [])
        new_entries = entries[state.next_index : state.next_index + count]
        state.next_index += len(new_entries)
        results: list[tuple[str, BaseEvent]] = []
        for entry in new_entries:
            state.pending[entry.msg_id] = entry
            event_cls = EVENT_TYPES[entry.event_type]
            results.append((entry.msg_id, event_cls.model_validate_json(entry.data)))
        return results

    def ack(self, stream: str, group: str, msg_id: str) -> None:
        """Acknowledge `msg_id` in `group`, removing it from the pending set."""
        state = self._groups.get((stream, group))
        if state is not None:
            state.pending.pop(msg_id, None)

    def pending(self, stream: str, group: str) -> list[str]:
        """Return the ids of messages read but not yet acked for `group`.

        Not part of the `Bus`/`FakeBus` shared interface; a test-only
        convenience for asserting at-least-once redelivery semantics.
        """
        state = self._groups.get((stream, group))
        return list(state.pending) if state is not None else []
