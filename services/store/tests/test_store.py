"""Tests for the `store` service's consume loop, using FakeBus and a temp SQLite file.

No real Redis is used: `nc_shared.bus.FakeBus` stands in for the bus, per
HANDOFF.md's testing convention (shared/tests/test_bus.py follows the same
pattern).
"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from nc_shared.bus import FakeBus
from nc_shared.events import Activity, Health, Notify, PersonState, PoseDebug
from sqlmodel import Session, select

from store.main import consume_once, make_engine
from store.models import EventRow, MorningSummaryRow
from store.retention import RetentionConfig, prune_expired_history


def test_consume_once_persists_a_published_event(tmp_path):
    db_path = tmp_path / "night.db"
    engine = make_engine(str(db_path))
    bus = FakeBus()

    event = PersonState(
        source="perceive",
        session_id="session-1",
        state="standing",
        confidence=0.9,
        zone="bed",
    )
    bus.publish(event)

    written = consume_once(bus, engine)
    assert written == 1

    with Session(engine) as session:
        rows = session.exec(select(EventRow)).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.stream == "person"
    assert row.event_type == "PersonState"
    assert row.session_id == "session-1"
    payload = json.loads(row.payload_json)
    assert payload["state"] == "standing"
    assert payload["zone"] == "bed"


def test_consume_once_persists_events_from_multiple_streams(tmp_path):
    db_path = tmp_path / "night.db"
    engine = make_engine(str(db_path))
    bus = FakeBus()

    bus.publish(Health(source="agent", service="agent", ok=True, detail="fine"))
    bus.publish(Notify(source="agent", level="info", title="t", body="b", repeat_until_ack=False))

    written = consume_once(bus, engine)
    assert written == 2

    with Session(engine) as session:
        rows = session.exec(select(EventRow)).all()
    streams = {row.stream for row in rows}
    assert streams == {"health", "notify"}


def test_consume_once_does_not_persist_capped_streams(tmp_path):
    from nc_shared.events import Frame

    db_path = tmp_path / "night.db"
    engine = make_engine(str(db_path))
    bus = FakeBus()

    bus.publish(Frame(source="capture", jpeg=b"abc", width=1, height=1, source_kind="usb"))
    bus.publish(
        PoseDebug(
            source="perceive", landmarks={}, bbox=None, confidence=0, detected=False, latency_ms=1
        ),
        maxlen=50,
    )
    bus.publish(
        Activity(source="listen", service="listen", kind="transcribe", phase="start"), maxlen=200
    )

    written = consume_once(bus, engine)
    assert written == 0

    with Session(engine) as session:
        rows = session.exec(select(EventRow)).all()
    assert rows == []


def test_consume_once_is_idempotent_on_repeated_calls_with_no_new_events(tmp_path):
    db_path = tmp_path / "night.db"
    engine = make_engine(str(db_path))
    bus = FakeBus()

    bus.publish(Health(source="notify", service="notify", ok=True, detail="fine"))

    first = consume_once(bus, engine)
    second = consume_once(bus, engine)

    assert first == 1
    assert second == 0


def test_retention_defaults_to_ninety_days_and_validates_env():
    assert RetentionConfig.from_env({}).days == 90
    assert RetentionConfig.from_env({"DATA_RETENTION_DAYS": "30"}).days == 30

    for value in ("0", "-1", "forever"):
        with pytest.raises(ValueError, match="DATA_RETENTION_DAYS"):
            RetentionConfig.from_env({"DATA_RETENTION_DAYS": value})


def test_prune_expired_history_removes_only_rows_older_than_cutoff(tmp_path):
    now = datetime(2026, 9, 13, 12, tzinfo=UTC)
    engine = make_engine(str(tmp_path / "night.db"))
    with Session(engine) as session:
        for age_days in (91, 90, 89):
            session.add(
                EventRow(
                    stream="health",
                    event_type="Health",
                    ts=now - timedelta(days=age_days),
                    payload_json="{}",
                )
            )
        session.add(MorningSummaryRow(night_key="old", created_at=now - timedelta(days=91)))
        session.add(MorningSummaryRow(night_key="kept", created_at=now - timedelta(days=5)))
        session.commit()

    assert prune_expired_history(engine, 90, now=now) == 1

    with Session(engine) as session:
        events = session.exec(select(EventRow).order_by(EventRow.ts)).all()
        markers = session.exec(select(MorningSummaryRow)).all()
    assert len(events) == 2
    assert events[0].ts == (now - timedelta(days=90)).replace(tzinfo=None)
    assert [marker.night_key for marker in markers] == ["kept"]
