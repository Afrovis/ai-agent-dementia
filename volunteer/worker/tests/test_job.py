from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from volunteer_common import crypto, db, paths

from volunteer_worker import job, media, pipeline

AES_KEY = b"\x00" * crypto.AES_KEY_BYTES


@pytest.fixture
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _make_processing_submission(tmp_path: Path, public_key) -> tuple:
    submission_id = "sub-job-test-00000000001"
    conn = db.connect(paths.db_path(tmp_path))
    db.create_submission(
        conn,
        submission_id=submission_id,
        created_at="2026-01-01T00:00:00Z",
        consent_version="v1",
        upload_token_sha256="x",
        deletion_token_sha256="y",
    )
    conn.execute("UPDATE submissions SET status = 'processing' WHERE id = ?", (submission_id,))
    wrapped_key = crypto.wrap_key(public_key, AES_KEY)
    paths.key_wrapped_path(tmp_path, submission_id).parent.mkdir(parents=True, exist_ok=True)
    paths.key_wrapped_path(tmp_path, submission_id).write_bytes(wrapped_key)
    return conn, submission_id


def test_process_submission_success(tmp_path: Path, keypair, monkeypatch) -> None:
    private_key, public_key = keypair
    conn, submission_id = _make_processing_submission(tmp_path, public_key)

    monkeypatch.setattr(media, "decrypt_submission_to_raw", lambda *a, **k: tmp_path / "raw.mkv")

    calls = []

    def fake_run(subcommand, args, *, data_root, python="python"):
        calls.append(subcommand)
        if subcommand == "predict":
            return json.dumps({"tag": "yolo_fake-tag", "frames": 10}) + "\n"
        if subcommand == "visualize":
            analysis_path = paths.analysis_path(tmp_path, submission_id)
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_path.write_bytes(b"fake analysis mp4 bytes")
        return json.dumps({})

    monkeypatch.setattr(pipeline, "run", fake_run)

    job.process_submission(
        conn, tmp_path, submission_id, chunk_count=0, private_key=private_key, yolo_model="m.pt"
    )

    assert calls == ["prepare", "predict", "visualize"]
    row = db.get_submission(conn, submission_id)
    assert row["status"] == "ready"
    assert row["pipeline_tag"] == "yolo_fake-tag"

    result_bytes = paths.result_path(tmp_path, submission_id).read_bytes()
    decrypted = crypto.decrypt_result(AES_KEY, submission_id, result_bytes)
    assert decrypted == b"fake analysis mp4 bytes"


def test_process_submission_failure_sets_failed_status(
    tmp_path: Path, keypair, monkeypatch
) -> None:
    private_key, public_key = keypair
    conn, submission_id = _make_processing_submission(tmp_path, public_key)

    def boom(*a, **k):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(media, "decrypt_submission_to_raw", boom)

    job.process_submission(
        conn, tmp_path, submission_id, chunk_count=0, private_key=private_key, yolo_model="m.pt"
    )

    row = db.get_submission(conn, submission_id)
    assert row["status"] == "failed"
    assert row["error_public"] == job.PUBLIC_ERROR
    assert "ffmpeg exploded" in row["error_detail"]
