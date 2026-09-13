"""Automatic expiry of persisted event history."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlmodel import Session

from store.models import EventRow, MorningSummaryRow

DEFAULT_RETENTION_DAYS = 90


@dataclass(frozen=True)
class RetentionConfig:
    """Retention period for personal event history."""

    days: int = DEFAULT_RETENTION_DAYS

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> RetentionConfig:
        values = os.environ if env is None else env
        raw = values.get("DATA_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS))
        try:
            days = int(raw)
        except ValueError as exc:
            raise ValueError("DATA_RETENTION_DAYS must be a whole number") from exc
        if days < 1:
            raise ValueError("DATA_RETENTION_DAYS must be at least 1")
        return cls(days=days)


def prune_expired_history(
    engine,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    *,
    now: datetime | None = None,
) -> int:
    """Delete event history older than the configured age and return its row count.

    Summary delivery markers expire on the same boundary. They do not contain a
    transcript, but retaining them indefinitely would violate the caregiver's
    expectation that all history metadata follows the displayed retention period.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    cutoff = current.astimezone(UTC) - timedelta(days=retention_days)

    with Session(engine) as session:
        result = session.exec(delete(EventRow).where(EventRow.ts < cutoff))
        session.exec(delete(MorningSummaryRow).where(MorningSummaryRow.created_at < cutoff))
        session.commit()
        return int(result.rowcount or 0)
