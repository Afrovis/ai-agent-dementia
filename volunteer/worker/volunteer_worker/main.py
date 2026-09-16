"""Entry point for the `worker` service: the poll loop from HANDOFF.md
section 5.8, step 6 ("Every loop also finalizes stale `recording` rows and
runs pending purges")."""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from volunteer_common import db, paths
from volunteer_common.log import log

from volunteer_worker import job
from volunteer_worker.purge import purge_pending

SERVICE_NAME = "volunteer-worker"
POLL_INTERVAL_SECONDS = 5
STALE_RECORDING_MINUTES = 15

logging.basicConfig(level=logging.INFO)


def load_private_key(key_dir: Path) -> RSAPrivateKey:
    pem = (key_dir / "private.pem").read_bytes()
    return serialization.load_pem_private_key(pem, password=None)


def _minutes_ago(minutes: int) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_once(
    data_root: Path, private_key: RSAPrivateKey | None, *, yolo_model: str
) -> RSAPrivateKey | None:
    """One poll iteration. Returns the private key (loaded lazily, once it
    exists) so the caller can carry it into the next iteration."""
    key_dir = Path(os.environ.get("VOLUNTEER_KEY_DIR_MOUNT", "/keys"))
    if private_key is None and (key_dir / "private.pem").exists():
        private_key = load_private_key(key_dir)

    conn = db.connect(paths.db_path(data_root))
    try:
        now = job.now_iso()
        for submission_id in db.finalize_stale_recordings(
            conn, older_than=_minutes_ago(STALE_RECORDING_MINUTES), now=now
        ):
            log(SERVICE_NAME, "finalized stale recording", submission_id=submission_id)

        purge_pending(data_root, db.pending_deletions(conn))

        if private_key is not None:
            row = db.claim_oldest_finalized(conn, now=job.now_iso())
            if row is not None:
                log(SERVICE_NAME, "processing submission", submission_id=row["id"])
                job.process_submission(
                    conn,
                    data_root,
                    row["id"],
                    row["chunk_count"],
                    private_key,
                    yolo_model=yolo_model,
                )
    finally:
        conn.close()
    return private_key


def main() -> None:
    data_root = Path(os.environ.get("VOLUNTEER_DATA_DIR_MOUNT", "/data"))
    yolo_model = os.environ.get("VOLUNTEER_YOLO_MODEL", "/app/volunteer/worker/yolo11s-pose.pt")
    log(SERVICE_NAME, "starting", data_dir=str(data_root))

    private_key: RSAPrivateKey | None = None
    while True:
        private_key = run_once(data_root, private_key, yolo_model=yolo_model)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
