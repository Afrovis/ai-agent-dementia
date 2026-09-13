"""System-page tests for issue #26 data export and deletion."""

from __future__ import annotations

import json
import sqlite3

from fastapi.testclient import TestClient
from nc_shared.bus import FakeBus

from dashboard.app import create_app
from dashboard.data_management import delete_history, history_stats

AUTH = ("caregiver", "secret123")


def _history_db(path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """CREATE TABLE events (
                 id INTEGER PRIMARY KEY,
                 stream TEXT NOT NULL,
                 event_type TEXT NOT NULL,
                 session_id TEXT,
                 ts TEXT NOT NULL,
                 payload_json TEXT NOT NULL
               );
               CREATE TABLE morning_summaries (
                 night_key TEXT PRIMARY KEY,
                 created_at TEXT NOT NULL
               );"""
        )
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (
                1,
                "speech_in",
                "Utterance",
                "session-1",
                "2026-09-13T03:00:00+00:00",
                json.dumps({"text": "I need the restroom"}),
            ),
        )
        connection.execute(
            "INSERT INTO morning_summaries VALUES (?, ?)",
            ("2026-09-12", "2026-09-13T08:00:00+00:00"),
        )


def test_system_page_shows_configured_retention_and_count(tmp_path):
    db_path = tmp_path / "night.db"
    _history_db(db_path)
    app = create_app(FakeBus(), password="secret123", db_path=db_path, data_retention_days=45)

    with TestClient(app) as client:
        response = client.get("/system", auth=AUTH)

    assert response.status_code == 200
    assert "45 days" in response.text
    assert "1 event" in response.text
    assert "System" in response.text
    assert "Live notification mode" in response.text


def test_system_page_makes_dry_run_notification_suppression_visible(tmp_path):
    db_path = tmp_path / "night.db"
    _history_db(db_path)
    app = create_app(FakeBus(), password="secret123", db_path=db_path, dry_run=True)

    with TestClient(app) as client:
        response = client.get("/system", auth=AUTH)

    assert response.status_code == 200
    assert "Dry run active" in response.text
    assert "outbound caregiver notifications are suppressed" in response.text


def test_export_downloads_no_cache_portable_json(tmp_path):
    db_path = tmp_path / "night.db"
    _history_db(db_path)
    app = create_app(FakeBus(), password="secret123", db_path=db_path)

    with TestClient(app) as client:
        response = client.get("/system/export", auth=AUTH)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"].startswith("attachment;")
    document = response.json()
    assert document["format"] == "night-companion-history-v1"
    assert document["event_count"] == 1
    assert document["events"][0]["payload"]["text"] == "I need the restroom"


def test_delete_clears_events_and_summary_markers_but_keeps_settings(tmp_path):
    db_path = tmp_path / "night.db"
    person_path = tmp_path / "person.yaml"
    person_path.write_text("name: Jean\n")
    _history_db(db_path)
    app = create_app(FakeBus(), password="secret123", db_path=db_path, person_path=person_path)

    with TestClient(app) as client:
        response = client.post(
            "/system/delete",
            data={"confirmation": "delete-retained-history"},
            auth=AUTH,
        )

    assert response.status_code == 200
    assert "Deleted 1 retained event" in response.text
    assert history_stats(db_path).event_count == 0
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM morning_summaries").fetchone()[0] == 0
    assert person_path.read_text() == "name: Jean\n"


def test_delete_rejects_wrong_confirmation_without_changing_history(tmp_path):
    db_path = tmp_path / "night.db"
    _history_db(db_path)
    app = create_app(FakeBus(), password="secret123", db_path=db_path)

    with TestClient(app) as client:
        response = client.post("/system/delete", data={"confirmation": "wrong"}, auth=AUTH)

    assert response.status_code == 400
    assert history_stats(db_path).event_count == 1


def test_system_data_routes_require_authentication(tmp_path):
    db_path = tmp_path / "night.db"
    _history_db(db_path)
    app = create_app(FakeBus(), password="secret123", db_path=db_path)

    with TestClient(app) as client:
        assert client.get("/system").status_code == 401
        assert client.get("/system/export").status_code == 401
        assert (
            client.post(
                "/system/delete", data={"confirmation": "delete-retained-history"}
            ).status_code
            == 401
        )

    assert history_stats(db_path).event_count == 1


def test_delete_missing_database_is_a_safe_no_op(tmp_path):
    assert delete_history(tmp_path / "missing.db") == 0
