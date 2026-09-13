"""Read-only views over the generic SQLite event store for dashboard history."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIMELINE_EVENT_TYPES = (
    "SessionState",
    "GoalChanged",
    "Say",
    "Utterance",
    "Notify",
    "CloudCall",
)


@dataclass(frozen=True)
class TimelineEvent:
    ts: datetime
    session_id: str
    kind: str
    summary: str
    detail: str = ""


@dataclass(frozen=True)
class NightTimeline:
    key: str
    events: tuple[TimelineEvent, ...]

    @property
    def session_count(self) -> int:
        return len({event.session_id for event in self.events})

    @property
    def first_at(self) -> datetime | None:
        return self.events[0].ts if self.events else None

    @property
    def last_at(self) -> datetime | None:
        return self.events[-1].ts if self.events else None


def load_nights(
    db_path: str | Path,
    *,
    timezone: str = "UTC",
    limit: int = 30,
) -> list[NightTimeline]:
    """Return newest-first night timelines without creating or changing the DB."""
    path = Path(db_path)
    if not path.is_file():
        return []

    placeholders = ",".join("?" for _ in TIMELINE_EVENT_TYPES)
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
        rows = connection.execute(
            f"""SELECT ts, session_id, event_type, payload_json
                FROM events
                WHERE session_id IS NOT NULL
                  AND event_type IN ({placeholders})
                ORDER BY ts ASC, id ASC""",  # noqa: S608 - placeholders are generated constants
            TIMELINE_EVENT_TYPES,
        ).fetchall()
    except (sqlite3.Error, OSError):
        return []
    finally:
        if "connection" in locals():
            connection.close()

    zone = _timezone(timezone)
    grouped: dict[str, list[TimelineEvent]] = {}
    for raw_ts, session_id, event_type, payload_json in rows:
        try:
            ts = _parse_timestamp(raw_ts)
            payload = json.loads(payload_json)
            event = _to_timeline_event(ts, str(session_id), event_type, payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        key = night_key(ts, zone)
        grouped.setdefault(key, []).append(event)

    keys = sorted(grouped, reverse=True)[:limit]
    return [NightTimeline(key=key, events=tuple(grouped[key])) for key in keys]


def night_key(ts: datetime, timezone: ZoneInfo) -> str:
    """Assign after-midnight events to the preceding night using a noon boundary."""
    return (ts.astimezone(timezone) - timedelta(hours=12)).date().isoformat()


def current_night_key(timezone: str, now: datetime | None = None) -> str:
    zone = _timezone(timezone)
    current = now or datetime.now(UTC)
    return night_key(current, zone)


def _timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _to_timeline_event(
    ts: datetime, session_id: str, event_type: str, payload: dict[str, object]
) -> TimelineEvent:
    if event_type == "Utterance":
        return TimelineEvent(ts, session_id, "heard", "Heard", str(payload.get("text", "")))
    if event_type == "Say":
        return TimelineEvent(ts, session_id, "said", "Agent said", str(payload.get("text", "")))
    if event_type == "SessionState":
        phase = str(payload.get("phase", "Unknown")).replace("_", " ").title()
        goal = str(payload.get("goal", "")).replace("_", " ")
        return TimelineEvent(ts, session_id, "phase", phase, f"Goal: {goal}" if goal else "")
    if event_type == "GoalChanged":
        old = str(payload.get("from_goal", "")).replace("_", " ")
        new = str(payload.get("to_goal", "")).replace("_", " ")
        return TimelineEvent(ts, session_id, "goal", "Goal changed", f"{old} → {new}")
    if event_type == "CloudCall":
        task = str(payload.get("task", "request")).replace("_", " ")
        sent = payload.get("payload", {})
        return TimelineEvent(
            ts,
            session_id,
            "cloud",
            f"Cloud {task} request",
            json.dumps(sent, ensure_ascii=False, sort_keys=True),
        )
    title = str(payload.get("title", "Alert"))
    return TimelineEvent(ts, session_id, "alert", title, str(payload.get("body", "")))


def format_night_label(key: str) -> str:
    try:
        parsed = date.fromisoformat(key)
    except ValueError:
        return key
    return f"Night of {parsed.strftime('%B')} {parsed.day}, {parsed.year}"
