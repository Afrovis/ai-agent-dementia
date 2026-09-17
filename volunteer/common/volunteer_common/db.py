"""SQLite queue for volunteer submissions (HANDOFF.md section 5.3).

Rollback journal, not WAL: WAL relies on shared memory that is unreliable
across containers on a macOS bind mount (PLAN.md, "Architecture").
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1

STATUSES = frozenset({"recording", "finalized", "processing", "ready", "failed", "deleted"})

# Allowed status -> status edges (HANDOFF.md section 5.3). "deleted" is
# reachable from every status, so it is added to each source's edge set
# below rather than listed once and forgotten.
_TRANSITIONS: dict[str, set[str]] = {
    "recording": {"finalized"},
    "finalized": {"processing"},
    "processing": {"ready", "failed"},
    "ready": set(),
    "failed": set(),
    "deleted": set(),
}
for _status in STATUSES:
    _TRANSITIONS.setdefault(_status, set()).add("deleted")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN
    ('recording','finalized','processing','ready','failed','deleted')),
  consent_version TEXT NOT NULL,
  upload_token_sha256 TEXT NOT NULL,
  deletion_token_sha256 TEXT NOT NULL,
  mime_type TEXT,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  bytes INTEGER NOT NULL DEFAULT 0,
  duration_s REAL,
  last_chunk_at TEXT,
  pipeline_tag TEXT,
  error_public TEXT,
  error_detail TEXT
);
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
"""


def connect(db_file: Path) -> sqlite3.Connection:
    """Open the database, creating the schema on first use.

    Rollback journal and a 5 s busy timeout (HANDOFF.md section 2), since
    `web` and `worker` open the same file from separate containers.
    """
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


class InvalidTransition(Exception):
    """Raised when a status change is not allowed from the row's current status."""


def create_submission(
    conn: sqlite3.Connection,
    *,
    submission_id: str,
    created_at: str,
    consent_version: str,
    upload_token_sha256: str,
    deletion_token_sha256: str,
) -> None:
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO submissions
                (id, created_at, updated_at, status, consent_version,
                 upload_token_sha256, deletion_token_sha256)
            VALUES (?, ?, ?, 'recording', ?, ?, ?)
            """,
            (
                submission_id,
                created_at,
                created_at,
                consent_version,
                upload_token_sha256,
                deletion_token_sha256,
            ),
        )


def get_submission(conn: sqlite3.Connection, submission_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()


def record_chunk(
    conn: sqlite3.Connection, submission_id: str, *, bytes_written: int, now: str
) -> None:
    """Record a *new* chunk: advances `chunk_count` as well as `bytes`."""
    with transaction(conn):
        conn.execute(
            """
            UPDATE submissions
            SET chunk_count = chunk_count + 1,
                bytes = bytes + ?,
                last_chunk_at = ?,
                updated_at = ?
            WHERE id = ? AND status = 'recording'
            """,
            (bytes_written, now, now, submission_id),
        )


def adjust_chunk_bytes(
    conn: sqlite3.Connection, submission_id: str, *, bytes_delta: int, now: str
) -> None:
    """Adjust `bytes` for a re-PUT of an already-recorded chunk (HANDOFF.md
    5.5: "Re-putting the same n overwrites it") without advancing
    `chunk_count`, which only counts distinct chunk indices."""
    with transaction(conn):
        conn.execute(
            """
            UPDATE submissions
            SET bytes = bytes + ?,
                last_chunk_at = ?,
                updated_at = ?
            WHERE id = ? AND status = 'recording'
            """,
            (bytes_delta, now, now, submission_id),
        )


def transition(
    conn: sqlite3.Connection,
    submission_id: str,
    *,
    from_status: str,
    to_status: str,
    now: str,
    **extra: object,
) -> bool:
    """Atomically move a row from `from_status` to `to_status`.

    Returns whether the row was actually updated -- false means the row was
    not in `from_status`, which the caller treats as a conflict (409/422)
    rather than a silent no-op.
    """
    if to_status not in _TRANSITIONS.get(from_status, set()):
        raise InvalidTransition(f"{from_status} -> {to_status} is not allowed")
    columns = ", ".join(f"{key} = ?" for key in extra)
    set_clause = "status = ?, updated_at = ?"
    if columns:
        set_clause = f"{set_clause}, {columns}"
    with transaction(conn):
        cursor = conn.execute(
            f"""
            UPDATE submissions
            SET {set_clause}
            WHERE id = ? AND status = ?
            """,
            (to_status, now, *extra.values(), submission_id, from_status),
        )
        return cursor.rowcount == 1


def claim_oldest_finalized(conn: sqlite3.Connection, *, now: str) -> sqlite3.Row | None:
    """Atomically claim the oldest `finalized` row and mark it `processing`."""
    with transaction(conn):
        row = conn.execute(
            """
            SELECT * FROM submissions
            WHERE status = 'finalized'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        cursor = conn.execute(
            """
            UPDATE submissions SET status = 'processing', updated_at = ?
            WHERE id = ? AND status = 'finalized'
            """,
            (now, row["id"]),
        )
        if cursor.rowcount != 1:
            return None
        return row


def finalize_stale_recordings(conn: sqlite3.Connection, *, older_than: str, now: str) -> list[str]:
    """Move `recording` rows with no chunk since `older_than` to `finalized`.

    Used both by `web` on explicit finalize and by `worker`'s stale sweep
    (HANDOFF.md 5.3: "or worker when stale for 15 minutes").
    """
    with transaction(conn):
        rows = conn.execute(
            """
            SELECT id FROM submissions
            WHERE status = 'recording'
              AND (last_chunk_at IS NULL OR last_chunk_at < ?)
              AND created_at < ?
            """,
            (older_than, older_than),
        ).fetchall()
        ids = [row["id"] for row in rows]
        for submission_id in ids:
            conn.execute(
                "UPDATE submissions SET status = 'finalized', updated_at = ? WHERE id = ?",
                (now, submission_id),
            )
        return ids


def pending_deletions(conn: sqlite3.Connection) -> list[str]:
    """Rows already marked `deleted` in the DB whose files a purge sweep should remove."""
    rows = conn.execute("SELECT id FROM submissions WHERE status = 'deleted'").fetchall()
    return [row["id"] for row in rows]
