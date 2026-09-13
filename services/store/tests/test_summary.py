from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import pytest
from nc_shared.bus import FakeBus
from nc_shared.events import Notify
from sqlmodel import Session, select

from store.main import make_engine
from store.models import EventRow, MorningSummaryRow
from store.summary import SummaryConfig, build_morning_summary, maybe_publish_morning_summary


def _row(
    event_type: str,
    ts: str,
    payload: dict[str, object],
    *,
    session_id: str | None = None,
    stream: str = "session",
) -> EventRow:
    return EventRow(
        stream=stream,
        event_type=event_type,
        session_id=session_id,
        ts=datetime.fromisoformat(ts),
        payload_json=json.dumps(payload),
    )


def _insert(engine, *rows: EventRow) -> None:
    with Session(engine) as database:
        database.add_all(rows)
        database.commit()


def test_summary_config_reads_local_time_and_timezone() -> None:
    config = SummaryConfig.from_env({"MORNING_SUMMARY_TIME": "07:45", "TZ": "America/New_York"})

    assert config.send_at == time(7, 45)
    assert config.timezone.key == "America/New_York"


@pytest.mark.parametrize("value", ["breakfast", "8", "8:00", "08:00:01", "25:00"])
def test_summary_config_rejects_invalid_time(value: str) -> None:
    with pytest.raises(ValueError, match="MORNING_SUMMARY_TIME"):
        SummaryConfig.from_env({"MORNING_SUMMARY_TIME": value, "TZ": "UTC"})


def test_summary_config_rejects_unknown_timezone() -> None:
    with pytest.raises(ValueError, match="known IANA timezone"):
        SummaryConfig.from_env({"MORNING_SUMMARY_TIME": "08:00", "TZ": "not/a-zone"})


def test_build_summary_reports_wakeups_durations_help_faults_and_escalations(tmp_path) -> None:
    engine = make_engine(str(tmp_path / "night.db"))
    _insert(
        engine,
        _row(
            "SessionState",
            "2026-09-12T22:00:00+00:00",
            {"phase": "OBSERVING"},
            session_id="resolved",
        ),
        _row(
            "Say",
            "2026-09-12T22:01:00+00:00",
            {"strategy": "soft_greeting"},
            session_id="resolved",
            stream="say",
        ),
        _row(
            "Say",
            "2026-09-12T22:03:00+00:00",
            {"strategy": "guided_return"},
            session_id="resolved",
            stream="say",
        ),
        _row(
            "SessionState",
            "2026-09-12T22:05:00+00:00",
            {"phase": "COOLDOWN"},
            session_id="resolved",
        ),
        _row(
            "SessionState",
            "2026-09-13T02:00:00+00:00",
            {"phase": "ENGAGED"},
            session_id="escalated",
        ),
        _row(
            "SessionState",
            "2026-09-13T02:02:00+00:00",
            {"phase": "ESCALATED"},
            session_id="escalated",
        ),
        _row(
            "SessionState",
            "2026-09-13T02:10:00+00:00",
            {"phase": "COOLDOWN"},
            session_id="escalated",
        ),
        _row(
            "Health",
            "2026-09-13T03:00:00+00:00",
            {"service": "listen", "ok": False, "detail": "transcription unavailable"},
            stream="health",
        ),
        # Repeated unhealthy heartbeats are one fault in the human summary.
        _row(
            "Health",
            "2026-09-13T03:01:00+00:00",
            {"service": "listen", "ok": False, "detail": "transcription unavailable"},
            stream="health",
        ),
        # Outside the noon-to-noon night window and therefore excluded.
        _row(
            "Health",
            "2026-09-12T10:00:00+00:00",
            {"service": "capture", "ok": False, "detail": "camera unavailable"},
            stream="health",
        ),
    )

    summary = build_morning_summary(
        engine,
        date(2026, 9, 12),
        cutoff=datetime(2026, 9, 13, 8, tzinfo=UTC),
    )

    assert summary.wake_up_count == 2
    assert summary.durations_seconds == (300, 600)
    assert summary.helped == (("guided_return", 1),)
    assert summary.faults == ("listen: transcription unavailable",)
    assert summary.escalation_count == 1
    assert "Durations: 5 min, 10 min" in summary.body
    assert "What helped: guided return (1)" in summary.body
    assert "Faults: listen: transcription unavailable" in summary.body


def test_scheduler_waits_until_due_then_publishes_only_once_across_restart(tmp_path) -> None:
    engine = make_engine(str(tmp_path / "night.db"))
    bus = FakeBus()
    config = SummaryConfig(send_at=time(8), timezone=ZoneInfo("America/New_York"))

    before = datetime(2026, 9, 13, 11, 59, tzinfo=UTC)  # 07:59 local
    assert maybe_publish_morning_summary(bus, engine, config, now=before) is None

    due = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)  # 08:00 local
    summary = maybe_publish_morning_summary(bus, engine, config, now=due)
    assert summary is not None
    assert summary.night_key == "2026-09-12"

    # No in-memory scheduler state is involved: the persisted marker alone
    # prevents another process/restart from publishing the same night.
    assert maybe_publish_morning_summary(bus, engine, config, now=due) is None
    with Session(engine) as database:
        assert database.get(MorningSummaryRow, "2026-09-12") is not None

    bus.ensure_group("notify", "test")
    messages = bus.read("notify", "test", "test-1")
    assert len(messages) == 1
    event = messages[0][1]
    assert isinstance(event, Notify)
    assert event.source == "store"
    assert event.level == "info"
    assert event.repeat_until_ack is False


def test_empty_night_still_sends_an_explicit_all_clear_summary(tmp_path) -> None:
    engine = make_engine(str(tmp_path / "night.db"))

    summary = build_morning_summary(
        engine,
        date(2026, 9, 12),
        cutoff=datetime(2026, 9, 13, 8, tzinfo=UTC),
    )

    assert summary.wake_up_count == 0
    assert summary.body == (
        "0 wake-ups. Durations: none. What helped: none recorded. Faults: none. Escalations: 0."
    )


def test_delivery_marker_table_is_created_with_the_event_store(tmp_path) -> None:
    engine = make_engine(str(tmp_path / "night.db"))

    with Session(engine) as database:
        assert database.exec(select(MorningSummaryRow)).all() == []
