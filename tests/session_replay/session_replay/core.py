"""Extract and replay recorded bus events against the current agent."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from agent.config import AgentConfig
from agent.llm import Intent, Interpretation, local_llm
from agent.main import run_once
from agent.profile import load_profile
from agent.session import Session
from agent.strategies import load_strategies
from nc_shared.bus import FakeBus
from nc_shared.events import EVENT_TYPES, BaseEvent

INPUTS = {
    ("person", "PersonState"),
    ("speech_in", "Utterance"),
    ("debug", "DebugControl"),
    ("debug", "ResetSession"),
}
OUTPUTS = {"SessionState", "GoalChanged", "Say", "Show", "LightCommand", "Notify", "DebugControl"}


def _is_input(row: dict) -> bool:
    key = (row.get("stream"), row.get("event_type"))
    return key in INPUTS and not (
        key == ("debug", "DebugControl") and row.get("payload", {}).get("source") == "agent"
    )


def parse_ts(value: str) -> datetime:
    """Parse an ISO timestamp and require an explicit offset."""
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"timestamp needs a timezone offset: {value}")
    return result.astimezone(UTC)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read JSONL with useful line-number errors."""
    rows = []
    with Path(path).open() as handle:
        for number, raw in enumerate(handle, 1):
            if raw.strip():
                try:
                    rows.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def extract(source: str | Path, target: str | Path, *, since=None, until=None) -> int:
    """Keep agent inputs and text-only observed outputs in timestamp order."""
    kept = []
    for row in read_jsonl(source):
        observed = (
            row.get("event_type") in OUTPUTS and row.get("payload", {}).get("source") == "agent"
        )
        if not _is_input(row) and not observed:
            continue
        when = parse_ts(row["ts"])
        if (since and when < since) or (until and when > until):
            continue
        if observed:
            row = {**row, "observed": True}
        kept.append((when, row))
    kept.sort(key=lambda pair: pair[0])
    with Path(target).open("w") as handle:
        for _, row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(kept)


class RecordedLLM:
    """Interpret text from a scenario map; use fixed caregiver speech and no plan."""

    model = "recorded"

    def __init__(self, interpretations: dict) -> None:
        self.interpretations = interpretations

    def interpret(self, utterance, turns, profile):  # noqa: ARG002
        value = self.interpretations.get(utterance, {"intent": "unclear", "distress": 0})
        return Interpretation(intent=Intent(value["intent"]), distress=value["distress"])

    def compose(self, *args, **kwargs):  # noqa: ARG002
        return None

    def plan(self, *args, **kwargs):  # noqa: ARG002
        return None


class TimelineBus(FakeBus):
    """Capture publications with the simulated clock, including deferred Say."""

    def __init__(self, local_tz: ZoneInfo) -> None:
        super().__init__()
        self.now: datetime | None = None
        self.start: datetime | None = None
        self.local_tz = local_tz
        self.timeline: list[dict] = []

    def publish(self, event: BaseEvent, maxlen: int | None = None) -> str:
        if self.now is not None and event.source == "agent":
            event = event.model_copy(update={"ts": self.now.astimezone(UTC)})
            if type(event).__name__ in OUTPUTS:
                self.timeline.append(_timeline_row(self.start, self.now, event))
        return super().publish(event, maxlen=maxlen)


def _timeline_row(start: datetime, when: datetime, event: BaseEvent) -> dict:
    payload = event.model_dump(mode="json")
    row = {
        "t": round((when - start).total_seconds(), 3),
        "ts": when.isoformat(),
        "type": type(event).__name__,
    }
    for key in (
        "phase",
        "goal",
        "reason",
        "strategy",
        "text",
        "face",
        "light",
        "level",
        "state",
        "to_goal",
        "from_goal",
        "time_offset_hours",
        "force_in_bed",
    ):
        if key in payload:
            row[key] = payload[key]
    return row


def run_scenario(
    scenario: str | Path,
    *,
    expect: dict | None = None,
    llm_mode: str = "none",
    model: str | None = None,
    backend: str | None = None,
    base_url: str | None = None,
    strategies: str | None = None,
    person: str | None = None,
    tz: str = "America/New_York",
    tail_s: float = 60,
) -> list[dict]:
    """Step run_once on an exact 0.5 second clock and at each input timestamp."""
    if tail_s < 0:
        raise ValueError("tail_s must be nonnegative")
    rows = [row for row in read_jsonl(scenario) if _is_input(row) and not row.get("observed")]
    if not rows:
        raise ValueError("scenario has no agent inputs")
    inputs = sorted(((parse_ts(row["ts"]), row) for row in rows), key=lambda pair: pair[0])
    start = inputs[0][0]
    zone = ZoneInfo(tz)
    config = AgentConfig.from_env()
    # Replay always permits a session to start; local timezone still governs
    # caregiver phrases and the simulated datetime passed to the agent.
    config = replace(
        config,
        night_start=time(0),
        night_end=time(0),
        strategies_path=strategies or config.strategies_path,
        person_path=person or config.person_path,
        llm_model=model or config.llm_model,
        llm_backend=backend or config.llm_backend,
        llm_url=base_url or config.llm_url,
    )
    profile = load_profile(config.person_path)
    session = Session(
        config=config,
        strategies=load_strategies(config.strategies_path),
        id_fn=lambda: "replay-session",
    )
    bus = TimelineBus(zone)
    bus.start = start
    bus.ensure_group("person", "agent")
    bus.ensure_group("speech_in", "agent")
    bus.ensure_group("debug", "agent")
    if llm_mode == "recorded":
        llm = RecordedLLM((expect or {}).get("interpretations", {}))
    elif llm_mode == "live":
        url = base_url or (
            os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
            if config.llm_backend == "ollama"
            else config.llm_url
        )
        llm = local_llm(
            config.llm_backend,
            url=url,
            model=config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
        )
    elif llm_mode == "none":
        llm = None
    else:
        raise ValueError(f"invalid LLM mode: {llm_mode}")

    def step(when: datetime, row: dict | None = None) -> None:
        bus.now = when
        if row is not None:
            event = EVENT_TYPES[row["event_type"]].model_validate(row["payload"])
            bus.publish(event)
            item = _timeline_row(start, when, event)
            item["type"] = "IN"
            if row["event_type"] == "Utterance":
                item["heard"] = event.text
            elif row["event_type"] == "PersonState":
                item["person"] = {"state": event.state, "zone": event.zone}
            elif row["event_type"] == "DebugControl":
                item["debug"] = {
                    "force_in_bed": event.force_in_bed,
                    "time_offset_hours": event.time_offset_hours,
                }
            else:
                item["reset_session"] = True
            bus.timeline.append(item)
        run_once(
            bus,
            session,
            block_ms=None,
            now_fn=lambda: when.astimezone(zone),
            profile=profile,
            llm=llm,
        )

    end = inputs[-1][0] + timedelta(seconds=tail_s)
    tick = start
    index = 0
    while tick <= end or index < len(inputs):
        next_input = inputs[index][0] if index < len(inputs) else None
        if next_input is not None and (tick > end or next_input <= tick):
            when, row = inputs[index]
            step(when, row)
            index += 1
        elif tick <= end:
            step(tick)
            tick += timedelta(seconds=0.5)
    return bus.timeline


def check_expectations(timeline: list[dict], spec: dict) -> tuple[bool, list[str]]:
    """Check ordered output subsequences in windows following matching inputs."""

    def matches_anchor(row: dict, anchor: dict) -> bool:
        return all(
            (
                isinstance(value, dict)
                and isinstance(row.get(key), dict)
                and all(row[key].get(field) == wanted for field, wanted in value.items())
            )
            if key == "debug"
            else row.get(key) == value
            for key, value in anchor.items()
            if key != "occurrence"
        )

    reports = []
    passed = True
    for number, item in enumerate(spec.get("expect", []), 1):
        anchor = item.get("after", {})
        occurrence = anchor.get("occurrence", 1)
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
            raise ValueError("after.occurrence must be a positive integer")
        anchors = [
            i
            for i, row in enumerate(timeline)
            if row["type"] == "IN" and matches_anchor(row, anchor)
        ]
        if len(anchors) < occurrence:
            passed = False
            reports.append(f"FAIL expectation {number}: anchor {anchor!r} not found")
            continue
        start_i = anchors[occurrence - 1]
        limit = timeline[start_i]["t"] + float(item.get("within_s", float("inf")))
        excerpt = [row for row in timeline[start_i + 1 :] if row["t"] <= limit]
        cursor = 0
        missing = None
        for expected in item.get("events", []):
            match = next(
                (
                    i
                    for i in range(cursor, len(excerpt))
                    if all(excerpt[i].get(k) == v for k, v in expected.items())
                ),
                None,
            )
            if match is None:
                missing = expected
                break
            cursor = match + 1
        forbidden = next(
            (
                row
                for row in excerpt
                if row["type"] != "IN"
                and any(
                    all(row.get(k) == v for k, v in pattern.items())
                    for pattern in item.get("absent", [])
                )
            ),
            None,
        )
        ok = missing is None and forbidden is None
        passed &= ok
        reports.append(
            f"{'PASS' if ok else 'FAIL'} expectation {number}: after {anchor!r}"
            + (f" missing {missing!r}" if missing else "")
            + (f" forbidden {forbidden!r}" if forbidden else "")
        )
        reports.extend("  " + format_row(row) for row in excerpt if row["type"] != "IN")
    for number, expected in enumerate(spec.get("never", []), 1):
        violations = [
            row
            for row in timeline
            if row["type"] != "IN" and all(row.get(k) == v for k, v in expected.items())
        ]
        ok = not violations
        passed &= ok
        reports.append(f"{'PASS' if ok else 'FAIL'} never {number}: {expected!r}")
        reports.extend("  " + format_row(row) for row in violations)
    return passed, reports


def format_row(row: dict) -> str:
    """Compact human-readable timeline entry."""
    if row["type"] == "IN":
        if "heard" in row:
            detail = f"heard {json.dumps(row['heard'], ensure_ascii=False)}"
        elif "person" in row:
            detail = f"person {row['person']['state']}@{row['person']['zone']}"
        elif "debug" in row:
            debug = row["debug"]
            detail = (
                f"debug force_in_bed={debug['force_in_bed']} offset={debug['time_offset_hours']}"
            )
        else:
            detail = "reset_session"
    else:
        detail = (
            row["type"]
            + " "
            + " ".join(
                f"{key}={json.dumps(row[key], ensure_ascii=False)}"
                for key in (
                    "phase",
                    "goal",
                    "to_goal",
                    "reason",
                    "strategy",
                    "text",
                    "face",
                    "light",
                    "state",
                    "level",
                    "force_in_bed",
                    "time_offset_hours",
                )
                if key in row
            )
        )
    return f"{row['t']:7.1f}s {detail}"


def load_expect(path: str | Path) -> dict:
    """Load an expectation document."""
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("expectation file must contain a mapping")
    return value
