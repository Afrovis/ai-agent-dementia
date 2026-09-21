"""Replay decision scenarios through the real agent loop with an injected clock."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path

from agent.config import AgentConfig
from agent.llm import LLMClient
from agent.main import (
    PERSON_GROUP,
    PERSON_STREAM,
    UTTERANCE_GROUP,
    UTTERANCE_STREAM,
    run_once,
)
from agent.profile import PersonProfile
from agent.rules import Phase
from agent.session import Session
from agent.strategies import FAMILIAR_VOICE_ID, StrategyDef, load_strategies
from nc_shared.bus import FakeBus
from nc_shared.events import BaseEvent, PersonState, Utterance

from decision_bench.schema import (
    DEFAULT_PROFILE_PATH,
    Scenario,
    load_default_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRATEGIES_PATH = REPO_ROOT / "config/strategies.example.yaml"
PERSON_HEARTBEAT_SECONDS = 60.0


@dataclass(frozen=True)
class TraceEntry:
    """One input, agent output, state change, or LLM fault at a scenario second."""

    t: float
    kind: str
    data: dict[str, object]


@dataclass
class Trace:
    """Ordered evidence produced by one scenario replay."""

    scenario_id: str
    start: datetime
    entries: list[TraceEntry] = field(default_factory=list)
    end_t: float = 0.0


class _TraceBus(FakeBus):
    def __init__(self, trace: Trace) -> None:
        super().__init__()
        self.trace = trace
        self.now_t = 0.0
        self.record_outputs = True

    def publish(self, event: BaseEvent, maxlen: int | None = None) -> str:
        message_id = super().publish(event, maxlen=maxlen)
        if self.record_outputs:
            self.trace.entries.append(
                TraceEntry(
                    t=self.now_t,
                    kind=type(event).__name__,
                    data=event.model_dump(mode="json"),
                )
            )
        return message_id

    def publish_input(self, event: BaseEvent, *, repeated: bool = False) -> str:
        self.record_outputs = False
        try:
            message_id = self.publish(event)
        finally:
            self.record_outputs = True
        data = event.model_dump(mode="json")
        data["input"] = True
        if repeated:
            data["repeated"] = True
        self.trace.entries.append(TraceEntry(self.now_t, type(event).__name__, data))
        return message_id


class _TracingLLM:
    """Record failures at the LLM seam while preserving fail-quiet ``None`` semantics."""

    def __init__(self, client: LLMClient, bus: _TraceBus) -> None:
        self.client = client
        self.bus = bus

    def _call(self, task: str, *args: object) -> object | None:
        try:
            result = getattr(self.client, task)(*args)
        except Exception as exc:  # noqa: BLE001 - the real clients fail quiet too
            self.bus.trace.entries.append(
                TraceEntry(self.bus.now_t, "LLMError", {"task": task, "error": type(exc).__name__})
            )
            return None
        if result is None:
            self.bus.trace.entries.append(TraceEntry(self.bus.now_t, "LLMNone", {"task": task}))
        return result

    def interpret(self, utterance: str, turns, profile):
        return self._call("interpret", utterance, turns, profile)

    def compose(
        self,
        strategy_name: str,
        caregiver_phrase_template: str,
        profile,
        time_words: str,
        scene_note: str | None,
        utterance: str | None = None,
        goal: str | None = None,
    ):
        return self._call(
            "compose",
            strategy_name,
            caregiver_phrase_template,
            profile,
            time_words,
            scene_note,
            utterance,
            goal,
        )

    def plan(self, session_state, profile):
        return self._call("plan", session_state, profile)


def _profile(scenario: Scenario, profile_path: Path | None) -> PersonProfile:
    raw = load_default_profile(profile_path or DEFAULT_PROFILE_PATH) | scenario.profile
    tuple_fields = {"night_themes", "calming_things", "things_to_avoid", "physical_notes"}
    values = {key: tuple(value) if key in tuple_fields else value for key, value in raw.items()}
    return PersonProfile(**values)


def _strategies(scenario: Scenario, path: Path | None) -> list[StrategyDef]:
    strategies = load_strategies(path or DEFAULT_STRATEGIES_PATH, env={})
    if not scenario.voice_clip:
        return strategies
    return [
        replace(item, enabled=True, clip_id="decision-bench-familiar-voice")
        if item.id == FAMILIAR_VOICE_ID
        else item
        for item in strategies
    ]


def scenario_end(scenario: Scenario, *, tail_seconds: float = 30.0) -> float:
    """Last timeline/checkpoint/deadline second plus the configured tail."""
    points = [event.t for event in scenario.timeline]
    for checkpoint in scenario.checkpoints:
        if checkpoint.window is not None:
            points.append(checkpoint.window[1])
        if checkpoint.escalate_by is not None:
            points.append(checkpoint.deadline_from + checkpoint.escalate_by)
    return max(points, default=0.0) + tail_seconds


def _state_data(session: Session) -> dict[str, object]:
    strategy = session._engine.current()  # noqa: SLF001 - benchmark state observation only
    active = strategy.id if strategy and session.phase in (Phase.ENGAGED, Phase.ESCALATED) else None
    return {"phase": session.phase.value, "goal": session.goal, "strategy": active}


def run_scenario(
    scenario: Scenario,
    *,
    llm: LLMClient,
    profile_path: Path | None = None,
    strategies_path: Path | None = None,
    tick_seconds: float = 1.0,
    tail_seconds: float = 30.0,
) -> Trace:
    """Replay ``scenario`` through ``agent.main.run_once`` and return its trace."""
    if tick_seconds <= 0:
        raise ValueError("tick_seconds must be greater than zero")
    if tail_seconds < 0:
        raise ValueError("tail_seconds must not be negative")

    start = datetime.combine(date(2026, 1, 1), scenario.start)
    trace = Trace(scenario_id=scenario.id, start=start)
    bus = _TraceBus(trace)
    bus.ensure_group(PERSON_STREAM, PERSON_GROUP)
    bus.ensure_group(UTTERANCE_STREAM, UTTERANCE_GROUP)
    profile = _profile(scenario, profile_path)
    session = Session(
        config=AgentConfig(),
        id_fn=lambda: f"bench-{scenario.id}",
        strategies=_strategies(scenario, strategies_path),
    )
    client = _TracingLLM(llm, bus)
    end_t = scenario_end(scenario, tail_seconds=tail_seconds)
    trace.end_t = end_t
    event_index = 0
    latest_person = None
    last_person_publish: float | None = None
    prior_state: dict[str, object] | None = None

    step = 0
    while True:
        t = min(step * tick_seconds, end_t)
        bus.now_t = t
        published_person_now = False
        while event_index < len(scenario.timeline) and scenario.timeline[event_index].t <= t:
            item = scenario.timeline[event_index]
            if item.person is not None:
                latest_person = item.person
                event = PersonState(source="decision_bench", **item.person.model_dump())
                bus.publish_input(event)
                last_person_publish = t
                published_person_now = True
            else:
                assert item.utterance is not None
                bus.publish_input(Utterance(source="decision_bench", **item.utterance.model_dump()))
            event_index += 1

        # Real perceive publishes on state changes and repeats its latest reading every
        # PERCEIVE_HEARTBEAT_SECONDS (60 seconds by default).
        if (
            latest_person is not None
            and not published_person_now
            and last_person_publish is not None
            and t - last_person_publish >= PERSON_HEARTBEAT_SECONDS
        ):
            bus.publish_input(
                PersonState(source="decision_bench", **latest_person.model_dump()), repeated=True
            )
            last_person_publish = t

        now = start + timedelta(seconds=t)
        run_once(
            bus,
            session,
            block_ms=0,
            now_fn=lambda current=now: current,
            profile=profile,
            llm=client,
        )
        state = _state_data(session)
        if state != prior_state:
            trace.entries.append(TraceEntry(t, "State", state))
            prior_state = state
        if t >= end_t:
            break
        step += 1

    trace.entries.sort(key=lambda entry: entry.t)
    return trace
