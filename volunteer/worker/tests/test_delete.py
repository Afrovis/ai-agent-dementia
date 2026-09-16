from __future__ import annotations

from pathlib import Path

from volunteer_common import db, paths

from volunteer_worker.delete import main


def test_delete_marks_row_and_purges_files(tmp_path: Path) -> None:
    submission_id = "sub-delete-cli-000000001"
    conn = db.connect(paths.db_path(tmp_path))
    db.create_submission(
        conn,
        submission_id=submission_id,
        created_at="2026-01-01T00:00:00Z",
        consent_version="v1",
        upload_token_sha256="x",
        deletion_token_sha256="y",
    )
    raw_path = paths.raw_path(tmp_path, submission_id)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"data")

    exit_code = main([submission_id, "--data-dir", str(tmp_path)])

    assert exit_code == 0
    assert db.get_submission(conn, submission_id)["status"] == "deleted"
    assert not raw_path.exists()


def test_delete_unknown_id_fails(tmp_path: Path) -> None:
    db.connect(paths.db_path(tmp_path))
    assert main(["does-not-exist-0000000001", "--data-dir", str(tmp_path)]) == 1
