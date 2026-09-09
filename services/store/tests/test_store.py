"""Tests for the `store` service's consume loop, using FakeBus and a temp SQLite file.

No real Redis is used: `nc_shared.bus.FakeBus` stands in for the bus, per
HANDOFF.md's testing convention (shared/tests/test_bus.py follows the same
pattern).
"""

import json

from nc_shared.bus import FakeBus
from nc_shared.events import Health, Notify, PersonState
from sqlmodel import Session, select

from store.main import consume_once, make_engine
from store.models import EventRow


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
