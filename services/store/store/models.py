"""SQLModel schema for the store service's SQLite database.

Every event read off the bus (excluding the capped `frames` and `audio_in`
streams, per HANDOFF.md section 5) is persisted as one row in a single
generic `events` table. No per-event-type tables yet; that can follow once
a concrete need for querying typed columns shows up.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class EventRow(SQLModel, table=True):
    """One event persisted from the bus."""

    __tablename__ = "events"

    id: int | None = Field(default=None, primary_key=True)
    stream: str
    event_type: str
    session_id: str | None = None
    ts: datetime
    payload_json: str


class MorningSummaryRow(SQLModel, table=True):
    """Durable once-per-night delivery marker for the caregiver summary."""

    __tablename__ = "morning_summaries"

    night_key: str = Field(primary_key=True)
    created_at: datetime
