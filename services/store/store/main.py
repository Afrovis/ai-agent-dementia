"""Entry point for the `store` service: persistence and morning summaries.

`store` reads every stream except the capped `frames`, `audio_in`, and
`frames_raw` streams (per HANDOFF.md section 5: "Everything else is
persisted to SQLite by `store`") via a Redis consumer group, and writes
each event as one row in the generic `events` table (see
`store.models.EventRow`).

`PERSISTED_STREAMS` is derived from `nc_shared.events.EVENT_STREAMS`
rather than hard-coded, so a new event added to `events.py` is picked up
automatically unless it targets `frames`, `audio_in`, or `frames_raw`.

Issue #24 also schedules a once-per-night caregiver summary from that event
history. The summary itself is an informational `Notify` event and therefore
follows the same delivery and persistence path as every other notification.
"""

from __future__ import annotations

import logging
import os
import time

import redis
from nc_shared.bus import Bus
from nc_shared.events import EVENT_STREAMS
from sqlmodel import Session, SQLModel, create_engine

from store.models import EventRow
from store.summary import SummaryConfig, maybe_publish_morning_summary

SERVICE_NAME = "store"
GROUP = "store"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

CAPPED_STREAMS = {"frames", "audio_in", "frames_raw"}
PERSISTED_STREAMS = sorted(set(EVENT_STREAMS.values()) - CAPPED_STREAMS)


def make_engine(db_path: str):
    """Create a SQLModel engine for the sqlite file at `db_path`, creating tables."""
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    return engine


def consume_once(bus, engine, consumer: str = "store-1", count: int = 10) -> int:
    """Read up to `count` new messages from each persisted stream and store them.

    Returns the number of events persisted. Each message is acked only after
    its row has been committed to SQLite, so a mid-batch write failure leaves
    the unwritten messages unacked (and therefore redelivered) rather than
    silently dropped.
    """
    written = 0
    with Session(engine) as session:
        for stream in PERSISTED_STREAMS:
            bus.ensure_group(stream, GROUP)
            for msg_id, event in bus.read(stream, GROUP, consumer, count=count, block_ms=100):
                row = EventRow(
                    stream=stream,
                    event_type=type(event).__name__,
                    session_id=event.session_id,
                    ts=event.ts,
                    payload_json=event.model_dump_json(),
                )
                session.add(row)
                session.commit()
                bus.ack(stream, GROUP, msg_id)
                written += 1
    return written


def run() -> None:
    """Loop forever, consuming from the bus and persisting to SQLite."""
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    db_path = os.environ.get("DB_PATH", "data/night.db")
    summary_config = SummaryConfig.from_env()
    logger.info(
        "store starting: redis_url=%s db_path=%s morning_summary_time=%s timezone=%s",
        redis_url,
        db_path,
        summary_config.send_at.strftime("%H:%M"),
        summary_config.timezone.key,
    )

    bus = Bus(redis.Redis.from_url(redis_url))
    engine = make_engine(db_path)

    while True:
        written = consume_once(bus, engine)
        summary = maybe_publish_morning_summary(bus, engine, summary_config)
        if summary is not None:
            logger.info(
                "morning summary published: night_key=%s wake_up_count=%s fault_count=%s",
                summary.night_key,
                summary.wake_up_count,
                len(summary.faults),
            )
        if written == 0:
            time.sleep(1)


if __name__ == "__main__":
    run()
