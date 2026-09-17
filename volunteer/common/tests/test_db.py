from __future__ import annotations

from pathlib import Path

import pytest

from volunteer_common import db


def _fresh(tmp_path: Path):
    conn = db.connect(tmp_path / "volunteer.db")
    db.create_submission(
        conn,
        submission_id="abc123",
        created_at="2026-01-01T00:00:00Z",
        consent_version="v1",
        upload_token_sha256="tok",
        deletion_token_sha256="del",
    )
    return conn


def test_create_and_get(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    row = db.get_submission(conn, "abc123")
    assert row["status"] == "recording"
    assert row["chunk_count"] == 0


def test_record_chunk_only_while_recording(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    db.record_chunk(conn, "abc123", bytes_written=100, now="2026-01-01T00:00:01Z")
    row = db.get_submission(conn, "abc123")
    assert row["chunk_count"] == 1
    assert row["bytes"] == 100

    db.transition(
        conn,
        "abc123",
        from_status="recording",
        to_status="finalized",
        now="2026-01-01T00:00:02Z",
    )
    db.record_chunk(conn, "abc123", bytes_written=50, now="2026-01-01T00:00:03Z")
    row = db.get_submission(conn, "abc123")
    assert row["chunk_count"] == 1


def test_transition_rejects_disallowed_edge(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    with pytest.raises(db.InvalidTransition):
        db.transition(
            conn,
            "abc123",
            from_status="recording",
            to_status="ready",
            now="2026-01-01T00:00:00Z",
        )


def test_transition_only_moves_matching_row(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    ok = db.transition(
        conn,
        "abc123",
        from_status="finalized",
        to_status="processing",
        now="2026-01-01T00:00:00Z",
    )
    assert ok is False
    row = db.get_submission(conn, "abc123")
    assert row["status"] == "recording"


def test_deleted_reachable_from_every_status(tmp_path: Path) -> None:
    for status in db.STATUSES - {"deleted"}:
        conn = _fresh(tmp_path / status)
        conn.execute("UPDATE submissions SET status = ? WHERE id = 'abc123'", (status,))
        ok = db.transition(
            conn,
            "abc123",
            from_status=status,
            to_status="deleted",
            now="2026-01-01T00:00:00Z",
        )
        assert ok is True


def test_claim_oldest_finalized_is_atomic_and_ordered(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "volunteer.db")
    for i, created in enumerate(["2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"]):
        db.create_submission(
            conn,
            submission_id=f"id{i}",
            created_at=created,
            consent_version="v1",
            upload_token_sha256="tok",
            deletion_token_sha256="del",
        )
        conn.execute("UPDATE submissions SET status = 'finalized' WHERE id = ?", (f"id{i}",))

    claimed = db.claim_oldest_finalized(conn, now="2026-01-01T00:00:03Z")
    assert claimed["id"] == "id1"  # earlier created_at, despite insertion order
    assert db.get_submission(conn, "id1")["status"] == "processing"

    second = db.claim_oldest_finalized(conn, now="2026-01-01T00:00:04Z")
    assert second["id"] == "id0"

    assert db.claim_oldest_finalized(conn, now="2026-01-01T00:00:05Z") is None


def test_finalize_stale_recordings(tmp_path: Path) -> None:
    conn = _fresh(tmp_path)
    ids = db.finalize_stale_recordings(
        conn, older_than="2026-01-01T00:15:00Z", now="2026-01-01T00:20:00Z"
    )
    assert ids == ["abc123"]
    assert db.get_submission(conn, "abc123")["status"] == "finalized"
