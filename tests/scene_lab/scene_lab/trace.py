"""Normalize bus, offline runner, and operational log events without media payloads."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from .thresholds import Thresholds

INPUTS = {"PersonState", "SpeechStarted", "Utterance"}
OUTPUTS = {"SessionState", "GoalChanged", "Say", "Show", "Notify", "LightCommand"}


class TraceEvent(BaseModel):
    t: float
    ts: str | None = None
    kind: Literal["input", "output", "activity", "decision"]
    type: str
    data: dict


class Trace(BaseModel):
    id: str
    source: Literal["decision_bench", "session_replay", "export", "agent_log", "live"]
    events: list[TraceEvent]
    meta: dict = {}
    end_t: float = 0.0

    @model_validator(mode="after")
    def sorted_events(self) -> Trace:
        self.events.sort(key=lambda event: event.t)
        self.end_t = max(self.end_t, max((event.t for event in self.events), default=0.0))
        return self

    def of_type(self, name: str) -> list[TraceEvent]:
        return [event for event in self.events if event.type == name]

    def write_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as out:
            for event in self.events:
                out.write(event.model_dump_json() + "\n")

    @classmethod
    def read_jsonl(
        cls, path: str | Path, *, id: str | None = None, source: str = "export"
    ) -> Trace:
        events = [
            TraceEvent.model_validate_json(line)
            for line in Path(path).read_text().splitlines()
            if line
        ]
        return cls(id=id or Path(path).stem, source=source, events=events)


def _event(t: float, name: str, data: dict, ts: str | None = None) -> TraceEvent | None:
    if name in INPUTS:
        kind = "input"
    elif name in OUTPUTS:
        kind = "output"
    elif name in {"Activity", "LLMError"}:
        kind = "activity"
    else:
        return None
    data = dict(data)
    if name == "Activity" and data.get("kind") == "decision":
        kind = "decision"
        try:
            detail = json.loads(data.get("detail") or "{}")
            if isinstance(detail, dict):
                data.update(detail)
        except (ValueError, TypeError):
            pass
    return TraceEvent(t=float(t), ts=ts or data.get("ts"), kind=kind, type=name, data=data)


def from_decision_bench(trace: object) -> Trace:
    start = trace.start
    events = []
    for entry in trace.entries:
        ts = (start + timedelta(seconds=entry.t)).isoformat()
        item = _event(entry.t, entry.kind, entry.data, ts)
        if item:
            events.append(item)
    return Trace(id=trace.scenario_id, source="decision_bench", events=events, end_t=trace.end_t)


def from_session_replay(rows: list[dict], scenario_id: str) -> Trace:
    """Accept run_scenario's combined timeline, including type=IN input rows."""
    events = []
    for row in rows:
        if row.get("type") == "IN":
            if "heard" in row:
                name, data = "Utterance", {"text": row["heard"]}
            elif "person" in row:
                name, data = "PersonState", row["person"]
            else:
                continue
        else:
            name = row.get("type", "")
            data = {k: v for k, v in row.items() if k not in {"type", "t", "ts"}}
        item = _event(row["t"], name, data, row.get("ts"))
        if item:
            events.append(item)
    return Trace(id=scenario_id, source="session_replay", events=events)


def _timestamp(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def from_export(lines: list[dict], *, id: str = "export") -> Trace:
    def stamp(row: dict) -> float | None:
        recorded = _timestamp(row.get("recorded_at"))
        return recorded if recorded is not None else _timestamp(row.get("ts"))

    first = next((value for row in lines if (value := stamp(row)) is not None), 0.0)
    events = []
    for row in lines:
        when = stamp(row)
        if when is None:
            continue
        item = _event(
            when - first, row.get("event_type", ""), row.get("payload") or {}, row.get("ts")
        )
        if item:
            events.append(item)
    return Trace(id=id, source="export", events=events)


def from_agent_log(lines: list[str | dict], *, id: str = "agent-log") -> Trace:
    """Best effort: missing timestamps use line order; published speech logs omit text."""
    parsed = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line) if isinstance(line, str) else line
        except ValueError:
            continue
        if row.get("service") != "agent":
            continue
        stamp = _timestamp(row.get("ts") or row.get("timestamp") or row.get("time"))
        parsed.append((stamp, index, row))
    if not parsed:
        return Trace(id=id, source="agent_log", events=[])
    first = next((stamp for stamp, _, _ in parsed if stamp is not None), None)
    events = []
    for stamp, index, row in parsed:
        t = stamp - first if stamp is not None and first is not None else float(index)
        message = row.get("message", "")
        if message == "dropped deferred Say" or (
            "pending say" in message and ("drop" in message or "discard" in message)
        ):
            data = {"decision": "pending_say_dropped", **row}
            item = TraceEvent(t=t, ts=row.get("ts"), kind="decision", type="Activity", data=data)
        elif message.startswith("vetoed"):
            data = {"decision": "vetoed", **row}
            item = TraceEvent(t=t, ts=row.get("ts"), kind="decision", type="Activity", data=data)
        else:
            item = _event(t, row.get("event_type", ""), row, row.get("ts"))
        if item:
            events.append(item)
    return Trace(id=id, source="agent_log", events=events)


class Playback(BaseModel):
    say_t: float
    text: str
    start: float
    end: float
    interrupted: bool
    estimated: bool
    interruptible: bool
    strategy: str | None


def playbacks(trace: Trace, thresholds: Thresholds) -> list[Playback]:
    says = trace.of_type("Say")
    reports = [e for e in trace.of_type("Activity") if e.data.get("kind") == "playback"]
    # Only 'playing' is an audible start; received/requested/unlocked are setup.
    starts = [e for e in reports if e.data.get("detail") == "playing"]
    ends = [
        e
        for e in reports
        if e.data.get("phase") == "end"
        or e.data.get("detail")
        in {"ended", "interrupted", "failed", "no_audio", "barge-in", "replaced by newer Say"}
    ]
    result = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    live = bool(reports)
    for index, say in enumerate(says):
        # Pair by time, not by index: a Say owns the first unused 'playing' report before the
        # next Say, so one blocked or failed playback cannot shift every later pairing.
        next_say_t = says[index + 1].t if index + 1 < len(says) else math.inf
        start_pair = next(
            (
                (i, e)
                for i, e in enumerate(starts)
                if i not in used_starts and say.t <= e.t < next_say_t
            ),
            None,
        )
        start_event = start_pair[1] if start_pair else None
        if start_pair:
            used_starts.add(start_pair[0])
        if start_event is None and live:
            # A live page reported playback for other Says but never played this one.
            continue
        if start_event:
            end_pair = next(
                ((i, e) for i, e in enumerate(ends) if i not in used_ends and e.t >= start_event.t),
                None,
            )
            end_event = end_pair[1] if end_pair else None
            if end_pair:
                used_ends.add(end_pair[0])
            start = start_event.t
            end = end_event.t if end_event else max(trace.end_t, start)
            interrupted = bool(
                end_event
                and end_event.data.get("detail")
                in {"interrupted", "barge-in", "replaced by newer Say"}
            )
            estimated = False
        else:
            start = say.t + thresholds.estimate_tts_s
            end = start + max(
                thresholds.estimate_min_s,
                len(say.data.get("text", "").split()) / thresholds.estimate_words_per_s,
            )
            interrupted, estimated = False, True
        result.append(
            Playback(
                say_t=say.t,
                text=say.data.get("text", ""),
                start=start,
                end=end,
                interrupted=interrupted,
                estimated=estimated,
                interruptible=say.data.get("interruptible", True),
                strategy=say.data.get("strategy"),
            )
        )
    return result
