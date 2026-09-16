"""Submissions business logic (HANDOFF.md section 5.5), kept separate from
`app.py`'s routing so it can be unit tested without an HTTP client.

Nothing here ever imports the private key or decrypts anything -- `web`
only ever writes and reads ciphertext, `key.wrapped` and `submission.json`
(HANDOFF.md rule 2).
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from volunteer_common import db, paths

ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22}$")


def generate_submission_id() -> str:
    return secrets.token_urlsafe(16)


def generate_upload_token() -> str:
    return secrets.token_urlsafe(32)


def sha256_hex(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def free_disk_mb(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free // (1024 * 1024)


class ConsentRejected(Exception):
    pass


class TurnstileRejected(Exception):
    pass


class DiskLow(Exception):
    pass


class NotFound(Exception):
    pass


class Unauthorized(Exception):
    pass


class Conflict(Exception):
    pass


class TooLarge(Exception):
    pass


class IncompleteUpload(Exception):
    pass


@dataclass(frozen=True)
class CreatedSubmission:
    submission_id: str
    upload_token: str


class SubmissionsService:
    def __init__(self, data_root: Path, *, min_free_disk_mb: int, consent_version: str) -> None:
        self._data_root = data_root
        self._min_free_disk_mb = min_free_disk_mb
        self._consent_version = consent_version

    def _connect(self):
        return db.connect(paths.db_path(self._data_root))

    def create(
        self,
        *,
        consents: dict[str, bool],
        wrapped_key: bytes,
        deletion_token_sha256: str,
        mime_type: str | None,
    ) -> CreatedSubmission:
        if not all(consents.get(key) is True for key in ("adult", "understood", "alone")):
            raise ConsentRejected("all three consents must be true")
        if free_disk_mb(self._data_root) < self._min_free_disk_mb:
            raise DiskLow("not enough free disk space for a new submission")

        submission_id = generate_submission_id()
        upload_token = generate_upload_token()
        conn = self._connect()
        try:
            db.create_submission(
                conn,
                submission_id=submission_id,
                created_at=now_iso(),
                consent_version=self._consent_version,
                upload_token_sha256=sha256_hex(upload_token),
                deletion_token_sha256=deletion_token_sha256,
            )
        finally:
            conn.close()

        incoming = paths.incoming_dir(self._data_root, submission_id)
        incoming.mkdir(parents=True, exist_ok=True)
        paths.key_wrapped_path(self._data_root, submission_id).write_bytes(wrapped_key)
        paths.submission_json_path(self._data_root, submission_id).write_text(
            json.dumps({"consent_version": self._consent_version, "mime_type": mime_type})
        )
        return CreatedSubmission(submission_id=submission_id, upload_token=upload_token)

    def _require_row(self, conn, submission_id: str):
        if not ID_PATTERN.match(submission_id):
            raise NotFound(submission_id)
        row = db.get_submission(conn, submission_id)
        if row is None:
            raise NotFound(submission_id)
        return row

    def _check_upload_token(self, row, upload_token: str | None) -> None:
        if not upload_token or sha256_hex(upload_token) != row["upload_token_sha256"]:
            raise Unauthorized("bad or missing upload token")

    def write_chunk(
        self,
        submission_id: str,
        n: int,
        body: bytes,
        *,
        upload_token: str | None,
        chunk_max_bytes: int,
        max_upload_mb: int,
    ) -> None:
        conn = self._connect()
        try:
            row = self._require_row(conn, submission_id)
            self._check_upload_token(row, upload_token)
            if row["status"] != "recording":
                raise Conflict(f"submission is {row['status']}, not recording")
            if len(body) > chunk_max_bytes:
                raise TooLarge(f"chunk exceeds {chunk_max_bytes} bytes")
            if row["bytes"] + len(body) > max_upload_mb * 1024 * 1024:
                raise TooLarge(f"submission exceeds {max_upload_mb} MB total")

            path = paths.chunk_path(self._data_root, submission_id, n)
            already_present = path.exists()
            previous_size = path.stat().st_size if already_present else 0
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)

            if already_present:
                db.adjust_chunk_bytes(
                    conn, submission_id, bytes_delta=len(body) - previous_size, now=now_iso()
                )
            else:
                db.record_chunk(conn, submission_id, bytes_written=len(body), now=now_iso())
        finally:
            conn.close()

    def finalize(
        self,
        submission_id: str,
        *,
        upload_token: str | None,
        chunk_count: int,
        duration_s: float,
        mime_type: str | None,
        markers: list[dict],
    ) -> None:
        conn = self._connect()
        try:
            row = self._require_row(conn, submission_id)
            self._check_upload_token(row, upload_token)
            for n in range(chunk_count):
                if not paths.chunk_path(self._data_root, submission_id, n).exists():
                    raise IncompleteUpload(f"missing chunk {n}")

            info_path = paths.submission_json_path(self._data_root, submission_id)
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            info.update({"mime_type": mime_type, "duration_s": duration_s, "markers": markers})
            info_path.write_text(json.dumps(info))

            ok = db.transition(
                conn,
                submission_id,
                from_status="recording",
                to_status="finalized",
                now=now_iso(),
                duration_s=duration_s,
                mime_type=mime_type,
            )
            if not ok:
                raise Conflict(f"submission is {row['status']}, not recording")
        finally:
            conn.close()

    def get_status(self, submission_id: str) -> dict:
        conn = self._connect()
        try:
            row = self._require_row(conn, submission_id)
            queue_position = None
            if row["status"] == "finalized":
                queue_position = conn.execute(
                    """
                    SELECT COUNT(*) FROM submissions
                    WHERE status = 'finalized' AND created_at <= ?
                    """,
                    (row["created_at"],),
                ).fetchone()[0]
            return {
                "status": row["status"],
                "created_at": row["created_at"],
                "queue_position": queue_position,
                "error_public": row["error_public"],
            }
        finally:
            conn.close()

    def get_result_path(self, submission_id: str) -> Path:
        conn = self._connect()
        try:
            row = self._require_row(conn, submission_id)
            if row["status"] != "ready":
                raise NotFound(submission_id)
        finally:
            conn.close()
        result_path = paths.result_path(self._data_root, submission_id)
        if not result_path.exists():
            raise NotFound(submission_id)
        return result_path

    def delete(self, submission_id: str, *, deletion_token: str) -> None:
        conn = self._connect()
        try:
            row = self._require_row(conn, submission_id)
            if not secrets.compare_digest(sha256_hex(deletion_token), row["deletion_token_sha256"]):
                raise Unauthorized("bad deletion token")
            db.transition(
                conn,
                submission_id,
                from_status=row["status"],
                to_status="deleted",
                now=now_iso(),
            )
        finally:
            conn.close()
