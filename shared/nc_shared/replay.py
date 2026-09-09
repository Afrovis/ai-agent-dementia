"""Event recording and replay tooling.

Records every event flowing over the bus to a JSONL file, and replays a
JSONL file back onto a bus at a configurable speed, preserving the relative
timing between events. This is what lets every service be tested without a
camera or mic (HANDOFF.md section 4: "every service must be testable with
no camera, mic, Ollama, or Redis by using the replay fixtures and a fake
bus"; section 8, M0 done criteria: "replay tooling works").

One JSONL line per recorded event, shaped as::

    {
        "stream": "frames",
        "event_type": "Frame",
        "ts": "2026-09-08T02:14:00+00:00",
        "recorded_at": 1757296440.123,
        "payload": {...event.model_dump(mode="json")...}
    }

``ts`` is the event's own timestamp (used to preserve relative timing on
replay); ``recorded_at`` is the wall-clock `time.time()` at which the
recorder observed it (diagnostic only, not used for timing). ``payload`` is
the full event, including any binary fields, which round-trip as base64
text through `model_dump(mode="json")` / `model_validate` because they use
`nc_shared.events.BytesAsBase64` (see events.py).

Usable as a library (`record_once`, `replay`) or from the command line::

    python -m nc_shared.replay record redis://localhost:6379 out.jsonl
    python -m nc_shared.replay play redis://localhost:6379 out.jsonl --speed 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import TextIO

import redis

from nc_shared.bus import Bus
from nc_shared.events import EVENT_STREAMS, EVENT_TYPES, BaseEvent

RECORD_GROUP = "replay-recorder"
"""Consumer group name the recorder reads under. Every stream is recorded,
including the capped `frames` and `audio_in` streams -- the issue asks for
*all* bus events, unlike `store` which skips those two (services/store
persists everything but the capped streams; the recorder is not `store`,
it exists so a captured session can be replayed frame-for-frame)."""

ALL_STREAMS: list[str] = sorted(set(EVENT_STREAMS.values()))

CAPPED_MAXLEN: dict[str, int] = {"frames": 50, "audio_in": 50}
"""Approximate MAXLEN to apply when replaying onto capped streams, matching
HANDOFF.md's `MAXLEN ~ 50` for `frames` and `audio_in`."""


def record_once(
    bus,
    out: TextIO,
    consumer: str = "recorder-1",
    count: int = 10,
    block_ms: int = 100,
    now_fn: Callable[[], float] = time.time,
) -> int:
    """Read up to `count` new messages from every stream and append them to `out`.

    Creates the `RECORD_GROUP` consumer group on each stream if needed. Each
    message is written as one JSON line and acked only after the write
    succeeds. Returns the number of events recorded.
    """
    written = 0
    for stream in ALL_STREAMS:
        bus.ensure_group(stream, RECORD_GROUP)
        for msg_id, event in bus.read(
            stream, RECORD_GROUP, consumer, count=count, block_ms=block_ms
        ):
            line = {
                "stream": stream,
                "event_type": type(event).__name__,
                "ts": event.ts.isoformat(),
                "recorded_at": now_fn(),
                "payload": event.model_dump(mode="json"),
            }
            out.write(json.dumps(line))
            out.write("\n")
            bus.ack(stream, RECORD_GROUP, msg_id)
            written += 1
    return written


def run_recorder(
    bus,
    output_path: str,
    consumer: str = "recorder-1",
    poll_interval: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Record forever, appending to `output_path` until interrupted.

    Used by the `record` CLI command against a real `Bus`. Polls every
    stream, and sleeps `poll_interval` seconds whenever a pass records
    nothing, to avoid a busy loop.
    """
    with open(output_path, "a") as out:
        while True:
            written = record_once(bus, out, consumer=consumer)
            out.flush()
            if written == 0:
                sleep_fn(poll_interval)


def _iter_lines(input_path: str) -> Iterator[dict]:
    """Yield parsed JSON objects from each non-blank line of `input_path`."""
    with open(input_path) as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            yield json.loads(raw_line)


def _event_from_line(line: dict) -> BaseEvent:
    """Reconstruct the pydantic event described by one recorded JSONL line."""
    event_cls = EVENT_TYPES[line["event_type"]]
    return event_cls.model_validate(line["payload"])


def replay(
    bus,
    input_path: str,
    speed: float | None = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Publish every event in the JSONL file at `input_path` onto `bus`.

    Events are republished onto the stream named in each line and
    reconstructed as the correct pydantic type via `EVENT_TYPES`.

    Relative timing between consecutive events' original `ts` values is
    preserved by sleeping (via the injectable `sleep_fn`, default
    `time.sleep`) between publishes, scaled by `speed`: `speed=1.0` is
    real-time, `speed=10.0` is ten times faster. `speed` of `0` or `None`
    means "as fast as possible": no sleeping at all, regardless of the
    original gaps.

    Returns the number of events published.
    """
    previous_ts: datetime | None = None
    published = 0
    for line in _iter_lines(input_path):
        ts = datetime.fromisoformat(line["ts"])
        if speed and previous_ts is not None:
            gap_s = (ts - previous_ts).total_seconds()
            if gap_s > 0:
                sleep_fn(gap_s / speed)
        previous_ts = ts

        event = _event_from_line(line)
        maxlen = CAPPED_MAXLEN.get(line["stream"])
        bus.publish(event, maxlen=maxlen)
        published += 1
    return published


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nc_shared.replay",
        description="Record bus events to JSONL, or replay a JSONL file onto the bus.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="record all bus events to a JSONL file")
    record_parser.add_argument("redis_url", help="e.g. redis://localhost:6379")
    record_parser.add_argument("output", help="path to the JSONL file to append to")

    play_parser = subparsers.add_parser("play", help="replay a JSONL file onto the bus")
    play_parser.add_argument("redis_url", help="e.g. redis://localhost:6379")
    play_parser.add_argument("input", help="path to the JSONL file to read")
    play_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier; 0 means as fast as possible (default: 1.0)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m nc_shared.replay`."""
    args = _build_arg_parser().parse_args(argv)
    bus = Bus(redis.Redis.from_url(args.redis_url))

    if args.command == "record":
        run_recorder(bus, args.output)
    elif args.command == "play":
        count = replay(bus, args.input, speed=args.speed)
        print(f"replayed {count} events")


if __name__ == "__main__":
    main()
