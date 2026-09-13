"""Build and schedule the caregiver's once-per-night morning summary.

The store owns this job because it has the complete persisted event history.
Summaries contain only structured event data already retained in SQLite; media
never enters this path. A durable marker makes the job idempotent across
service restarts.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from os import environ
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nc_shared.events import Notify
from sqlmodel import Session, select

from store.models import EventRow, MorningSummaryRow

DEFAULT_SUMMARY_TIME = time(8, 0)
SUMMARY_EVENT_TYPES = ("Health", "Say", "SessionState")


@dataclass(frozen=True)
class SummaryConfig:
    """Local schedule and timezone for morning delivery."""

    send_at: time = DEFAULT_SUMMARY_TIME
    timezone: ZoneInfo = ZoneInfo("UTC")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> SummaryConfig:
        values = environ if env is None else env
        raw_time = values.get("MORNING_SUMMARY_TIME", "08:00").strip()
        if re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", raw_time) is None:
            raise ValueError("MORNING_SUMMARY_TIME must be HH:MM")
        try:
            send_at = time.fromisoformat(raw_time)
        except ValueError as exc:
            raise ValueError("MORNING_SUMMARY_TIME must be HH:MM") from exc

        timezone_name = values.get("TZ", "UTC").strip() or "UTC"
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"TZ is not a known IANA timezone: {timezone_name}") from exc
        return cls(send_at=send_at, timezone=timezone)


@dataclass(frozen=True)
class MorningSummary:
    """Rendered notification plus useful aggregate values for callers."""

    night_key: str
    wake_up_count: int
    durations_seconds: tuple[int, ...]
    helped: tuple[tuple[str, int], ...]
    faults: tuple[str, ...]
    escalation_count: int
    title: str
    body: str


@dataclass
class _SessionFacts:
    started_at: datetime | None = None
    resolved_at: datetime | None = None
    escalated: bool = False
    says: list[tuple[datetime, str]] | None = None

    def __post_init__(self) -> None:
        if self.says is None:
            self.says = []


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _night_bounds(night: date, timezone: ZoneInfo, cutoff: datetime) -> tuple[datetime, datetime]:
    start = datetime.combine(night, time(12), timezone)
    end = datetime.combine(night + timedelta(days=1), time(12), timezone)
    return start.astimezone(UTC), min(end.astimezone(UTC), _aware(cutoff))


def _payload(row: EventRow) -> dict[str, object] | None:
    try:
        value = json.loads(row.payload_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return "<1 min"
    minutes = seconds // 60
    hours, remaining = divmod(minutes, 60)
    if hours and remaining:
        return f"{hours}h {remaining}m"
    if hours:
        return f"{hours}h"
    return f"{minutes} min"


def _bounded_list(values: list[str], *, limit: int = 6) -> str:
    shown = values[:limit]
    suffix = f", plus {len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def build_morning_summary(
    engine,
    night: date,
    *,
    timezone: ZoneInfo = ZoneInfo("UTC"),
    cutoff: datetime | None = None,
) -> MorningSummary:
    """Aggregate wake-ups, time-to-settle, effective strategies, and faults."""
    effective_cutoff = cutoff or datetime.now(UTC)
    start, end = _night_bounds(night, timezone, effective_cutoff)
    with Session(engine) as database:
        rows = database.exec(
            select(EventRow)
            .where(EventRow.ts >= start, EventRow.ts < end)
            .where(EventRow.event_type.in_(SUMMARY_EVENT_TYPES))  # type: ignore[union-attr]
            .order_by(EventRow.ts, EventRow.id)  # type: ignore[arg-type]
        ).all()

    sessions: dict[str, _SessionFacts] = defaultdict(_SessionFacts)
    faults: set[str] = set()
    for row in rows:
        payload = _payload(row)
        if payload is None:
            continue
        if row.event_type == "Health" and payload.get("ok") is False:
            service = str(payload.get("service") or row.stream)
            detail = " ".join(str(payload.get("detail") or "unhealthy").split())[:100]
            faults.add(f"{service}: {detail}")
            continue
        if not row.session_id:
            continue
        facts = sessions[row.session_id]
        timestamp = _aware(row.ts)
        if row.event_type == "Say":
            strategy = str(payload.get("strategy") or "").strip()
            if strategy:
                assert facts.says is not None
                facts.says.append((timestamp, strategy))
            continue
        phase = str(payload.get("phase") or "")
        if phase in {"OBSERVING", "ENGAGED", "ESCALATED"} and facts.started_at is None:
            facts.started_at = timestamp
        if phase == "ESCALATED":
            facts.escalated = True
        if phase == "COOLDOWN" and facts.resolved_at is None:
            facts.resolved_at = timestamp

    active = [facts for facts in sessions.values() if facts.started_at is not None]
    durations = sorted(
        max(0, int((facts.resolved_at - facts.started_at).total_seconds()))
        for facts in active
        if facts.resolved_at is not None
    )
    helped_counts: Counter[str] = Counter()
    for facts in active:
        if facts.resolved_at is None or facts.escalated:
            continue
        assert facts.says is not None
        candidates = [
            strategy
            for timestamp, strategy in facts.says
            if timestamp <= facts.resolved_at and strategy != "escalate_phone"
        ]
        if candidates:
            helped_counts[candidates[-1]] += 1

    helped = tuple(sorted(helped_counts.items(), key=lambda item: (-item[1], item[0])))
    escalation_count = sum(facts.escalated for facts in active)
    duration_text = (
        _bounded_list([_format_duration(value) for value in durations]) if durations else "none"
    )
    helped_text = (
        _bounded_list([f"{name.replace('_', ' ')} ({count})" for name, count in helped])
        if helped
        else "none recorded"
    )
    fault_text = _bounded_list(sorted(faults)) if faults else "none"
    wake_text = f"{len(active)} wake-up{'s' if len(active) != 1 else ''}"
    body = (
        f"{wake_text}. Durations: {duration_text}. What helped: {helped_text}. "
        f"Faults: {fault_text}. Escalations: {escalation_count}."
    )
    title = f"Morning summary — {night.strftime('%b')} {night.day}"
    return MorningSummary(
        night_key=night.isoformat(),
        wake_up_count=len(active),
        durations_seconds=tuple(durations),
        helped=helped,
        faults=tuple(sorted(faults)),
        escalation_count=escalation_count,
        title=title,
        body=body,
    )


def maybe_publish_morning_summary(
    bus,
    engine,
    config: SummaryConfig,
    *,
    now: datetime | None = None,
) -> MorningSummary | None:
    """Publish the prior night's summary once the configured local time is due."""
    current = _aware(now or datetime.now(UTC))
    local_now = current.astimezone(config.timezone)
    if local_now.timetz().replace(tzinfo=None) < config.send_at:
        return None

    night = local_now.date() - timedelta(days=1)
    night_key = night.isoformat()
    with Session(engine) as database:
        if database.get(MorningSummaryRow, night_key) is not None:
            return None

    summary = build_morning_summary(engine, night, timezone=config.timezone, cutoff=current)
    bus.publish(
        Notify(
            source="store",
            level="info",
            title=summary.title,
            body=summary.body,
            repeat_until_ack=False,
        )
    )
    with Session(engine) as database:
        database.add(MorningSummaryRow(night_key=night_key, created_at=current))
        database.commit()
    return summary


__all__ = [
    "MorningSummary",
    "SummaryConfig",
    "build_morning_summary",
    "maybe_publish_morning_summary",
]
