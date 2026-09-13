"""Export and secure deletion helpers for retained SQLite event history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class HistoryStats:
    event_count: int = 0
    oldest: str | None = None
    newest: str | None = None


@dataclass(frozen=True)
class HistoryExport:
    event_count: int
    chunks: Iterator[bytes]


def history_stats(db_path: str | Path) -> HistoryStats:
    """Return a small read-only summary, or empty stats for an unavailable DB."""
    path = Path(db_path)
    if not path.is_file():
        return HistoryStats()
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2) as connection:
            row = connection.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM events").fetchone()
    except (OSError, sqlite3.Error):
        return HistoryStats()
    if row is None:
        return HistoryStats()
    return HistoryStats(event_count=int(row[0]), oldest=row[1], newest=row[2])


def export_history(db_path: str | Path, *, exported_at: datetime | None = None) -> HistoryExport:
    """Stream a portable JSON snapshot without loading 90 days into memory."""
    path = Path(db_path)
    timestamp = exported_at or datetime.now(UTC)
    metadata = {
        "format": "night-companion-history-v1",
        "exported_at": timestamp.astimezone(UTC).isoformat(),
    }
    if not path.is_file():
        metadata["event_count"] = 0
        document = json.dumps({**metadata, "events": []}, ensure_ascii=False).encode("utf-8")
        return HistoryExport(0, iter((document,)))

    try:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro", uri=True, timeout=5, check_same_thread=False
        )
        event_count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        cursor = connection.execute(
            """SELECT id, stream, event_type, session_id, ts, payload_json
               FROM events ORDER BY ts ASC, id ASC"""
        )
    except sqlite3.Error as exc:
        if "connection" in locals():
            connection.close()
        if "no such table" in str(exc).lower():
            metadata["event_count"] = 0
            document = json.dumps({**metadata, "events": []}, ensure_ascii=False).encode("utf-8")
            return HistoryExport(0, iter((document,)))
        raise RuntimeError("retained history could not be exported") from exc

    def chunks() -> Iterator[bytes]:
        metadata["event_count"] = event_count
        encoded_metadata = json.dumps(metadata, ensure_ascii=False)
        yield (encoded_metadata[:-1] + ', "events": [').encode("utf-8")
        first = True
        try:
            for row_id, stream, event_type, session_id, ts, payload_json in cursor:
                try:
                    payload = json.loads(str(payload_json))
                except (TypeError, json.JSONDecodeError):
                    payload = {"unparsed_payload": str(payload_json)}
                event = {
                    "id": row_id,
                    "stream": stream,
                    "event_type": event_type,
                    "session_id": session_id,
                    "ts": ts,
                    "payload": payload,
                }
                separator = "" if first else ","
                first = False
                yield (separator + json.dumps(event, ensure_ascii=False)).encode("utf-8")
            yield b"]}\n"
        finally:
            connection.close()

    return HistoryExport(event_count, chunks())


def delete_history(db_path: str | Path) -> int:
    """Securely delete all retained event rows and summary markers.

    The DB itself remains in place so the store service can continue writing new
    events without a restart. SQLite's secure-delete pragma overwrites deleted
    cells; VACUUM then rebuilds the file to remove unused pages.
    """
    path = Path(db_path)
    if not path.is_file():
        return 0
    try:
        with sqlite3.connect(path, timeout=10) as connection:
            connection.execute("PRAGMA secure_delete = ON")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "events" not in tables:
                return 0
            count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            connection.execute("DELETE FROM events")
            if "morning_summaries" in tables:
                connection.execute("DELETE FROM morning_summaries")
            connection.commit()
            connection.execute("VACUUM")
            return count
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError("retained history could not be deleted") from exc
