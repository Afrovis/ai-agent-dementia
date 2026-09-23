"""Extract and replay recorded bus events against the current agent."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import agent.main as agent_main
import yaml
from agent.config import AgentConfig
from agent.llm import Intent, Interpretation, local_llm
from agent.main import run_once
from agent.profile import load_profile
from agent.session import Session
from agent.strategies import load_strategies
from nc_shared.bus import FakeBus
from nc_shared.events import EVENT_TYPES, BaseEvent

INPUTS = {("person", "PersonState"), ("speech_in", "Utterance")}
OUTPUTS = {"SessionState", "GoalChanged", "Say", "Show", "LightCommand", "Notify"}
LLM_KINDS = {"interpret", "compose", "plan"}


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
        key = (row.get("stream"), row.get("event_type"))
        observed = (
            row.get("event_type") in OUTPUTS and row.get("payload", {}).get("source") == "agent"
        )
        activity = (
            row.get("event_type") == "Activity"
            and row.get("payload", {}).get("source") == "agent"
            and row.get("payload", {}).get("kind") in LLM_KINDS
            and row.get("payload", {}).get("phase") == "end"
            and row.get("payload", {}).get("duration_ms") is not None
        )
        if key not in INPUTS and not observed and not activity:
            continue
        when = parse_ts(row["ts"])
        if (since and when < since) or (until and when > until):
            continue
        if activity:
            payload = row["payload"]
            row = {
                "stream": row["stream"],
                "event_type": "Activity",
                "ts": row["ts"],
                "payload": {key: payload[key] for key in ("kind", "phase", "duration_ms")},
                "observed": True,
            }
        elif observed:
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


class _LatencyLLM:
    """Advance the replay clock at each agent LLM entry point."""

    def __init__(
        self,
        client,
        bus,
        durations: dict[str, list[float]],
        fallback: float,
        scenario: str | Path,
        warned: bool,
    ) -> None:
        self.client = client
        self.bus = bus
        self.durations = durations
        self.fallback = fallback
        self.model = getattr(client, "model", type(client).__name__)
        self.scenario = scenario
        self.warned = warned

    def _call(self, kind, *args, **kwargs):
        try:
            return getattr(self.client, kind)(*args, **kwargs)
        finally:
            values = self.durations[kind]
            if not values and not self.warned:
                warnings.warn(
                    f"{self.scenario}: missing recorded {kind} duration; "
                    f"using fixed:{self.fallback}",
                    stacklevel=2,
                )
                self.warned = True
            seconds = values.pop(0) if values else self.fallback
            self.bus.now += timedelta(seconds=seconds)

    def interpret(self, *args, **kwargs):
        return self._call("interpret", *args, **kwargs)

    def compose(self, *args, **kwargs):
        return self._call("compose", *args, **kwargs)

    def plan(self, *args, **kwargs):
        return self._call("plan", *args, **kwargs)


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
            if type(event).__name__ in OUTPUTS or type(event).__name__ == "Activity":
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
        "kind",
        "detail",
        "duration_ms",
        "ok",
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
    llm_latency: str = "none",
    llm_latency_fallback: float = 2.5,
) -> list[dict]:
    """Step run_once on an exact 0.5 second clock and at each input timestamp."""
    if tail_s < 0:
        raise ValueError("tail_s must be nonnegative")
    if not 0 <= llm_latency_fallback < float("inf"):
        raise ValueError("--llm-latency-fallback must be nonnegative")
    if llm_latency == "none":
        fixed_s = 0.0
    elif llm_latency == "recorded":
        fixed_s = llm_latency_fallback
    elif llm_latency.startswith("fixed:"):
        try:
            fixed_s = float(llm_latency.removeprefix("fixed:"))
        except ValueError as exc:
            raise ValueError("--llm-latency requires fixed:<nonnegative seconds>") from exc
        if not 0 <= fixed_s < float("inf"):
            raise ValueError("--llm-latency requires fixed:<nonnegative seconds>")
    else:
        raise ValueError(f"invalid --llm-latency: {llm_latency}")
    all_rows = read_jsonl(scenario)
    scenario_interpretations = {}
    for row in all_rows:
        if row.get("event_type") == "InterpretationMap" and row.get("observed"):
            scenario_interpretations.update(row.get("payload", {}).get("interpretations", {}))
    durations = {kind: [] for kind in LLM_KINDS}
    warned = False
    if llm_latency == "recorded":
        for row in all_rows:
            payload = row.get("payload", {})
            if (
                row.get("event_type") == "Activity"
                and row.get("observed")
                and payload.get("kind") in LLM_KINDS
                and payload.get("phase") == "end"
            ):
                value = payload.get("duration_ms")
                if isinstance(value, (float, int)) and 0 <= value < float("inf"):
                    durations[payload["kind"]].append(value / 1000)
        if not any(durations.values()):
            warnings.warn(
                f"{scenario}: no recorded LLM durations; using fixed:{fixed_s}",
                stacklevel=2,
            )
            warned = True
    rows = [
        row
        for row in all_rows
        if (row.get("stream"), row.get("event_type")) in INPUTS and not row.get("observed")
    ]
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
    if llm_mode == "recorded":
        llm = RecordedLLM({**scenario_interpretations, **(expect or {}).get("interpretations", {})})
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
    latency_on = llm_latency != "none" and llm is not None
    if latency_on:
        llm = _LatencyLLM(
            llm, bus, durations, fixed_s, scenario, warned or llm_latency != "recorded"
        )

    def step(when: datetime, row: dict | None = None) -> None:
        bus.now = max(when, bus.now) if bus.now is not None else when
        if row is not None:
            event = EVENT_TYPES[row["event_type"]].model_validate(row["payload"])
            bus.publish(event)
            item = _timeline_row(start, bus.now, event)
            item["type"] = "IN"
            if row["event_type"] == "Utterance":
                item["heard"] = event.text
            else:
                item["person"] = {"state": event.state, "zone": event.zone}
            bus.timeline.append(item)
        if latency_on:
            with patch.object(
                agent_main,
                "time",
                SimpleNamespace(perf_counter=lambda: (bus.now - start).total_seconds()),
            ):
                run_once(
                    bus,
                    session,
                    block_ms=None,
                    now_fn=lambda: bus.now.astimezone(zone),
                    profile=profile,
                    llm=llm,
                )
        else:
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
        current = bus.now or tick
        if next_input is not None and (tick > end or next_input <= max(tick, current)):
            when, row = inputs[index]
            step(when, row)
            index += 1
        elif tick <= end:
            when = max(tick, current)
            step(when)
            tick = when + timedelta(seconds=0.5)
    return bus.timeline


def check_expectations(timeline: list[dict], spec: dict) -> tuple[bool, list[str]]:
    """Check ordered output subsequences in windows following matching inputs."""
    reports = []
    passed = True
    for number, item in enumerate(spec.get("expect", []), 1):
        anchor = item.get("after", {})
        anchors = [
            i
            for i, row in enumerate(timeline)
            if row["type"] == "IN" and all(row.get(key) == value for key, value in anchor.items())
        ]
        if not anchors:
            passed = False
            reports.append(f"FAIL expectation {number}: anchor {anchor!r} not found")
            continue
        start_i = anchors[0]
        anchor_t = timeline[start_i]["t"]
        limit = anchor_t + float(item.get("within_s", float("inf")))
        lower = anchor_t + float(item.get("delay_s", 0))
        excerpt = [row for row in timeline[start_i + 1 :] if lower <= row["t"] <= limit]
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
        spacing = item.get("min_spacing_s")
        close_pair = None
        if spacing is not None:
            matching = [
                row for row in excerpt if row.get("type") == item.get("spacing_type", "Say")
            ]
            close_pair = next(
                (
                    (a, b)
                    for a, b in zip(matching, matching[1:])
                    if b["t"] - a["t"] < float(spacing)
                ),
                None,
            )
        ok = missing is None and forbidden is None and close_pair is None
        passed &= ok
        reports.append(
            f"{'PASS' if ok else 'FAIL'} expectation {number}: after {anchor!r}"
            + (f" missing {missing!r}" if missing else "")
            + (f" forbidden {forbidden!r}" if forbidden else "")
            + (f" spacing {close_pair!r}" if close_pair else "")
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
        detail = (
            f"heard {json.dumps(row['heard'], ensure_ascii=False)}"
            if "heard" in row
            else f"person {row['person']['state']}@{row['person']['zone']}"
        )
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
