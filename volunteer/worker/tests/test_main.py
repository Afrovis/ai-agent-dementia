from __future__ import annotations

from pathlib import Path

from volunteer_common import db, paths

from volunteer_worker.main import run_once


def test_run_once_finalizes_stale_recordings_without_a_key(tmp_path: Path) -> None:
    submission_id = "sub-stale-000000000000001"
    conn = db.connect(paths.db_path(tmp_path))
    db.create_submission(
        conn,
        submission_id=submission_id,
        created_at="2000-01-01T00:00:00Z",
        consent_version="v1",
        upload_token_sha256="x",
        deletion_token_sha256="y",
    )

    result = run_once(tmp_path, private_key=None, yolo_model="m.pt")

    assert result is None  # no private.pem present, so no key is loaded
    assert db.get_submission(conn, submission_id)["status"] == "finalized"


def test_run_once_purges_deleted_submissions(tmp_path: Path) -> None:
    submission_id = "sub-deleted-00000000000001"
    conn = db.connect(paths.db_path(tmp_path))
    db.create_submission(
        conn,
        submission_id=submission_id,
        created_at="2026-01-01T00:00:00Z",
        consent_version="v1",
        upload_token_sha256="x",
        deletion_token_sha256="y",
    )
    conn.execute("UPDATE submissions SET status = 'deleted' WHERE id = ?", (submission_id,))
    raw_path = paths.raw_path(tmp_path, submission_id)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"x")

    run_once(tmp_path, private_key=None, yolo_model="m.pt")

    assert not raw_path.exists()
