"""The per-submission analysis job (HANDOFF.md section 5.8)."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa
from volunteer_common import crypto, db, paths
from volunteer_common.log import log

from volunteer_worker import media, pipeline

SERVICE_NAME = "volunteer-worker"

PUBLIC_ERROR = "Processing failed. Your video is stored safely; Mathias will look at it."


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def process_submission(
    conn: sqlite3.Connection,
    data_root: Path,
    submission_id: str,
    chunk_count: int,
    private_key: rsa.RSAPrivateKey,
    *,
    yolo_model: str,
    python: str = "python",
) -> None:
    """Run steps 2-4 of HANDOFF.md section 5.8 for one already-claimed
    (`processing`) submission, transitioning it to `ready` or `failed`."""
    try:
        raw_path = media.decrypt_submission_to_raw(
            data_root, submission_id, chunk_count, private_key
        )

        pipeline.run(
            "prepare",
            ["--video", str(raw_path), "--clip", submission_id, "--variant", "letterbox640"],
            data_root=str(data_root),
            python=python,
        )
        predict_stdout = pipeline.run(
            "predict",
            [
                "--clip",
                submission_id,
                "--backend",
                "yolo",
                "--yolo-model",
                yolo_model,
                "--variant",
                "letterbox640",
            ],
            data_root=str(data_root),
            python=python,
        )
        tag = json.loads(predict_stdout.strip().splitlines()[-1])["tag"]

        pipeline.run(
            "visualize",
            ["--clip", submission_id, "--mode", "pipeline", "--pipeline-tag", tag, "--force"],
            data_root=str(data_root),
            python=python,
        )

        plaintext = paths.analysis_path(data_root, submission_id).read_bytes()

        wrapped_key = paths.key_wrapped_path(data_root, submission_id).read_bytes()
        key = crypto.unwrap_key(private_key, wrapped_key)
        stream = crypto.encrypt_result(
            key, submission_id, plaintext, iv_source=lambda: os.urandom(crypto.GCM_IV_BYTES)
        )

        result_path = paths.result_path(data_root, submission_id)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = result_path.with_name(result_path.name + ".tmp")
        tmp_path.write_bytes(stream)
        os.replace(tmp_path, result_path)

        db.transition(
            conn,
            submission_id,
            from_status="processing",
            to_status="ready",
            now=now_iso(),
            pipeline_tag=tag,
        )
        log(SERVICE_NAME, "submission ready", submission_id=submission_id, pipeline_tag=tag)
    except Exception as exc:  # must never crash the poll loop
        stderr = getattr(exc, "stderr", "") or ""
        detail = f"{exc}\n{stderr}"[-4000:]
        db.transition(
            conn,
            submission_id,
            from_status="processing",
            to_status="failed",
            now=now_iso(),
            error_public=PUBLIC_ERROR,
            error_detail=detail,
        )
        log(SERVICE_NAME, "submission failed", submission_id=submission_id, error=detail[-500:])
