from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from dashboard.history import current_night_key, format_night_label, load_nights


def _database(path, rows):
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE events (
            id INTEGER PRIMARY KEY, stream TEXT, event_type TEXT,
            session_id TEXT, ts TEXT, payload_json TEXT
        )"""
    )
    connection.executemany(
        "INSERT INTO events(stream,event_type,session_id,ts,payload_json) VALUES(?,?,?,?,?)",
        rows,
    )
    connection.commit()
    connection.close()


def test_missing_database_is_an_empty_history(tmp_path):
    assert load_nights(tmp_path / "missing.db") == []
    assert not (tmp_path / "missing.db").exists()


def test_groups_after_midnight_activity_with_the_previous_night(tmp_path):
    path = tmp_path / "night.db"
    _database(
        path,
        [
            (
                "say",
                "Say",
                "session-1",
                "2026-09-13T02:00:00+00:00",
                json.dumps({"text": "It is time to rest."}),
            ),
            (
                "speech_in",
                "Utterance",
                "session-1",
                "2026-09-13T03:00:00+00:00",
                json.dumps({"text": "All right."}),
            ),
        ],
    )

    nights = load_nights(path, timezone="UTC")

    assert [night.key for night in nights] == ["2026-09-12"]
    assert nights[0].session_count == 1
    assert [(event.summary, event.detail) for event in nights[0].events] == [
        ("Agent said", "It is time to rest."),
        ("Heard", "All right."),
    ]


def test_history_skips_media_and_events_without_a_session(tmp_path):
    path = tmp_path / "night.db"
    _database(
        path,
        [
            ("health", "Health", None, "2026-09-12T22:00:00+00:00", "{}"),
            ("show", "Show", "session-1", "2026-09-12T22:01:00+00:00", "{}"),
            (
                "session",
                "SessionState",
                "session-1",
                "2026-09-12T22:02:00+00:00",
                json.dumps({"phase": "ENGAGED", "goal": "return_to_bed"}),
            ),
        ],
    )

    nights = load_nights(path)

    assert len(nights) == 1
    assert len(nights[0].events) == 1
    assert nights[0].events[0].summary == "Engaged"


def test_current_night_key_uses_the_same_noon_boundary():
    before_noon = datetime(2026, 9, 13, 3, tzinfo=UTC)
    after_noon = datetime(2026, 9, 13, 22, tzinfo=UTC)

    assert current_night_key("UTC", before_noon) == "2026-09-12"
    assert current_night_key("UTC", after_noon) == "2026-09-13"
    assert format_night_label("2026-09-12") == "Night of September 12, 2026"


def test_cloud_call_shows_the_exact_sent_payload_in_history(tmp_path):
    path = tmp_path / "night.db"
    sent = {
        "task": "Classify intent",
        "input": {"utterance": "Where am I?", "profile": {"name": "Jean"}},
    }
    _database(
        path,
        [
            (
                "cloud",
                "CloudCall",
                "session-1",
                "2026-09-12T22:00:00+00:00",
                json.dumps({"task": "interpret", "model": "claude-opus-5", "payload": sent}),
            )
        ],
    )

    event = load_nights(path)[0].events[0]
    assert event.summary == "Cloud interpret request"
    assert json.loads(event.detail) == sent
