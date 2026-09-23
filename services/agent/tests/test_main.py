"""Tests for `agent.main`: wiring `agent.session.Session` to the bus.

Uses `FakeBus` throughout, per HANDOFF.md section 4: no Redis, camera,
mic, or Ollama. `now_fn` is always an explicit, advancing fixed clock, so
nothing here sleeps for a real duration.
"""

import json
import wave
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from nc_shared.bus import FakeBus
from nc_shared.events import (
    Activity,
    CloudCall,
    GoalChanged,
    LightCommand,
    Notify,
    PersonState,
    Say,
    SessionState,
    Show,
    SpeechStarted,
    Utterance,
)

from agent.config import AgentConfig
from agent.llm import Composition, FakeLLM, Intent, Interpretation, Plan
from agent.main import (
    PERSON_GROUP,
    PERSON_STREAM,
    UTTERANCE_GROUP,
    UTTERANCE_STREAM,
    _maybe_publish_say,
    _publish_cloud_call,
    _publish_transition,
    maybe_emit_health,
    maybe_emit_session_heartbeat,
    run_once,
)
from agent.profile import PersonProfile
from agent.rules import Phase
from agent.session import Session, Transition
from agent.strategies import DEFAULT_STRATEGIES, ESCALATE_PHONE_ID, FAMILIAR_VOICE_ID

NIGHT = datetime(2026, 1, 1, 23, 0)


def make_bus() -> FakeBus:
    bus = FakeBus()
    bus.ensure_group(PERSON_STREAM, PERSON_GROUP)
    bus.ensure_group(UTTERANCE_STREAM, UTTERANCE_GROUP)
    bus.ensure_group("session", "test")
    bus.ensure_group("notify", "test")
    bus.ensure_group("show", "test")
    bus.ensure_group("say", "test")
    bus.ensure_group("light", "test")
    bus.ensure_group("cloud", "test")
    bus.ensure_group("activity", "test")
    return bus


def test_cloud_call_audit_uses_the_live_session_and_exact_payload():
    bus = make_bus()
    session = Session(config=AgentConfig())
    session.session_id = "session-1"
    payload = {"task": "classify", "input": {"utterance": "Where am I?"}}

    _publish_cloud_call(bus, session, "interpret", "claude-opus-5", payload)

    events = [event for _id, event in bus.read("cloud", "test", "c1", count=10)]
    assert events == [
        CloudCall(
            source="agent",
            session_id="session-1",
            task="interpret",
            model="claude-opus-5",
            payload=payload,
            ts=events[0].ts,
        )
    ]


def make_clock(start: datetime):
    """A `now_fn` that returns `start` until advanced by the caller."""
    box = {"now": start}

    def now_fn():
        return box["now"]

    def advance(seconds: float):
        box["now"] = box["now"] + timedelta(seconds=seconds)

    return now_fn, advance


def test_run_once_publishes_session_state_and_show_on_a_phase_change():
    bus = make_bus()
    config = AgentConfig()
    session = Session(config=config)
    now_fn, _advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    published = run_once(bus, session, now_fn=now_fn)

    assert len(published) == 1
    assert published[0].phase == "OBSERVING"

    session_events = bus.read("session", "test", "c1", count=10)
    assert len(session_events) == 1
    assert isinstance(session_events[0][1], SessionState)
    assert session_events[0][1].phase == "OBSERVING"

    show_events = bus.read("show", "test", "c1", count=10)
    assert len(show_events) == 1
    assert isinstance(show_events[0][1], Show)


def test_run_once_acks_person_state_messages():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, _advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="in_bed", confidence=0.9, zone="bed"))
    run_once(bus, session, now_fn=now_fn)

    assert bus.pending(PERSON_STREAM, PERSON_GROUP) == []


def test_utterance_moves_observing_to_engaged_through_run_once():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    assert session.phase.value == "OBSERVING"

    advance(2)
    bus.publish(Utterance(source="listen", text="hello", confidence=0.9, duration_s=1.0))
    published = run_once(bus, session, now_fn=now_fn)

    assert any(event.phase == "ENGAGED" for event in published)
    assert bus.pending(UTTERANCE_STREAM, UTTERANCE_GROUP) == []


def test_speech_started_is_acked_without_advancing_the_session():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, _advance = make_clock(NIGHT)
    bus.publish(SpeechStarted(source="listen", session_id="session-1"))

    assert run_once(bus, session, now_fn=now_fn) == []
    assert session.phase.value == "IDLE"
    assert bus.pending(UTTERANCE_STREAM, UTTERANCE_GROUP) == []


def test_escalation_publishes_a_notify_with_repeat_until_ack_and_source():
    bus = make_bus()
    session = Session(config=AgentConfig(floor_limit_seconds=0.0))
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)

    advance(1)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    published = run_once(bus, session, now_fn=now_fn)

    assert any(event.phase == "ESCALATED" for event in published)

    notify_events = bus.read("notify", "test", "c1", count=10)
    assert len(notify_events) == 1
    notify = notify_events[0][1]
    assert isinstance(notify, Notify)
    assert notify.repeat_until_ack is True
    assert notify.source == "agent"
    assert notify.level == "critical"


def test_escalation_from_idle_publishes_a_notify_too():
    # A fall straight from `in_bed` to `on_floor`, no prior session: rule 5
    # must still escalate and `agent.main` must still wire up a real
    # `Notify`, not just a `Transition`.
    bus = make_bus()
    session = Session(config=AgentConfig(floor_limit_seconds=0.0))
    now_fn, _advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    published = run_once(bus, session, now_fn=now_fn)

    assert any(event.phase == "ESCALATED" for event in published)

    notify_events = bus.read("notify", "test", "c1", count=10)
    assert len(notify_events) == 1
    notify = notify_events[0][1]
    assert isinstance(notify, Notify)
    assert notify.repeat_until_ack is True
    assert notify.source == "agent"
    assert notify.level == "critical"


def test_run_once_ticks_the_observing_timeout_with_no_new_messages():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=20.0))
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    assert session.phase.value == "OBSERVING"

    advance(25)
    published = run_once(bus, session, now_fn=now_fn)
    assert any(event.phase == "ENGAGED" for event in published)


def test_maybe_emit_session_heartbeat_respects_interval():
    bus = make_bus()
    session = Session(config=AgentConfig())

    last = maybe_emit_session_heartbeat(bus, session, None, NIGHT, interval=60.0)
    assert last == NIGHT

    soon = NIGHT + timedelta(seconds=10)
    unchanged = maybe_emit_session_heartbeat(bus, session, last, soon, interval=60.0)
    assert unchanged == last

    later = NIGHT + timedelta(seconds=61)
    updated = maybe_emit_session_heartbeat(bus, session, last, later, interval=60.0)
    assert updated == later

    events = bus.read("session", "test", "c1", count=10)
    assert len(events) == 2


def test_goal_changed_reaches_the_bus_with_correct_fields():
    # `zone_confirm_readings=1` here: this test is about the `GoalChanged`
    # wiring through `agent.main`, not the zone hysteresis itself, which
    # `tests/test_session.py` covers directly.
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0, zone_confirm_readings=1))
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)
    assert session.phase.value == "ENGAGED"

    bus.publish(
        PersonState(source="perceive", state="walking", confidence=0.9, zone="bathroom_path")
    )
    advance(1)
    run_once(bus, session, now_fn=now_fn)
    assert session.goal == "restroom"

    goal_events = bus.read("session", "test", "c1", count=10)
    goal_changed = [e for _id, e in goal_events if isinstance(e, GoalChanged)]
    assert len(goal_changed) == 1
    assert goal_changed[0].from_goal == "return_to_bed"
    assert goal_changed[0].to_goal == "restroom"
    assert goal_changed[0].session_id == session.session_id

    light_events = bus.read("light", "test", "c1", count=10)
    commands = [event for _id, event in light_events if isinstance(event, LightCommand)]
    assert [(event.state, event.reason) for event in commands] == [("on", "restroom_goal_started")]


def test_return_from_restroom_turns_light_off_and_selects_guided_return():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0, zone_confirm_readings=1))
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)

    bus.publish(PersonState(source="perceive", state="walking", confidence=0.9, zone="door"))
    advance(1)
    run_once(bus, session, now_fn=now_fn)
    bus.publish(PersonState(source="perceive", state="walking", confidence=0.9, zone="bed"))
    advance(9)
    run_once(bus, session, now_fn=now_fn)

    commands = [event for _id, event in bus.read("light", "test", "c1", count=10)]
    assert [event.state for event in commands] == ["on", "off"]
    shows = [event for _id, event in bus.read("show", "test", "c1", count=20)]
    assert shows[-1].headline == "Let's head back to bed"


def test_disabled_path_light_strategy_does_not_emit_hardware_command():
    bus = make_bus()
    strategies = [
        dc_replace(strategy, enabled=False) if strategy.id == "path_light" else strategy
        for strategy in DEFAULT_STRATEGIES
    ]
    session = Session(
        config=AgentConfig(observe_seconds=1.0, zone_confirm_readings=1),
        strategies=strategies,
    )
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)
    bus.publish(PersonState(source="perceive", state="walking", confidence=0.9, zone="door"))
    advance(1)
    run_once(bus, session, now_fn=now_fn)

    assert session.goal == "restroom"
    assert bus.read("light", "test", "c1", count=10) == []


def test_escalation_during_restroom_trip_keeps_path_light_on():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0, zone_confirm_readings=1))
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)
    bus.publish(PersonState(source="perceive", state="walking", confidence=0.9, zone="door"))
    advance(1)
    run_once(bus, session, now_fn=now_fn)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    advance(1)
    run_once(bus, session, now_fn=now_fn)

    commands = [event for _id, event in bus.read("light", "test", "c1", count=10)]
    assert [event.state for event in commands] == ["on"]


def test_session_state_goal_reflects_the_live_goal():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0, zone_confirm_readings=1))
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)

    bus.publish(
        PersonState(source="perceive", state="walking", confidence=0.9, zone="bathroom_path")
    )
    advance(1)
    published = run_once(bus, session, now_fn=now_fn)

    assert any(event.goal == "restroom" for event in published)
    assert session.goal == "restroom"


def test_maybe_emit_health_respects_interval():
    bus = FakeBus()
    bus.ensure_group("health", "test")

    last = maybe_emit_health(bus, None, 0.0)
    assert last == 0.0

    unchanged = maybe_emit_health(bus, last, 10.0)
    assert unchanged == last

    updated = maybe_emit_health(bus, last, 31.0)
    assert updated == 31.0

    events = bus.read("health", "test", "c1", count=10)
    assert len(events) == 2


# --- issue #14: strategy-driven Show/Say through run_once -----------------


def small_strategies(*, dwell=100.0, cooldown=50.0, n=3):
    """Same trick as `tests/test_strategies.py`'s `strategies_by_order`."""
    base = DEFAULT_STRATEGIES[0]
    return [
        dc_replace(
            base,
            id=f"s{i}",
            order=i,
            enabled=True,
            dwell_seconds=dwell,
            cooldown_seconds=cooldown,
            say_template=f"Say for s{i}." if i > 1 else None,
        )
        for i in range(1, n + 1)
    ]


def test_entering_engaged_publishes_the_first_strategys_show():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0), strategies=small_strategies())
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)
    assert session.phase.value == "ENGAGED"

    show_events = [e for _id, e in bus.read("show", "test", "c1", count=10)]
    assert isinstance(show_events[-1], Show)
    # s1 has no `say_template` (like `ambient_orient`): no Say published.
    say_events = bus.read("say", "test", "c1", count=10)
    assert say_events == []


def test_a_strategy_with_a_say_template_publishes_a_say():
    bus = make_bus()
    session = Session(
        config=AgentConfig(observe_seconds=1.0), strategies=small_strategies(dwell=10.0)
    )
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)  # -> ENGAGED, s1 (no Say)
    assert session.phase.value == "ENGAGED"

    advance(11)  # s1's dwell elapses -> advance to s2, which has a Say
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)

    say_events = [e for _id, e in bus.read("say", "test", "c1", count=10)]
    assert len(say_events) == 1
    assert isinstance(say_events[0], Say)
    assert say_events[0].text == "Say for s2."
    assert say_events[0].strategy == "s2"


def test_absent_reading_suppresses_ordinary_strategy_say_but_not_show():
    bus = make_bus()
    session = Session(
        config=AgentConfig(
            observe_seconds=1.0,
            absent_limit_seconds=600.0,
        ),
        strategies=[
            small_strategies(dwell=1.0, n=2)[0],
            dc_replace(small_strategies(dwell=1.0, n=2)[1], face="speaking"),
        ],
    )
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)  # -> ENGAGED, silent s1

    advance(2)
    bus.publish(PersonState(source="perceive", state="absent", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)  # -> s2, whose Say is suppressed

    assert session.last_person_state == "absent"
    assert bus.read("say", "test", "c1", count=10) == []
    shows = [event for _id, event in bus.read("show", "test", "c1", count=20)]
    assert shows[-1].headline == session.strategies[1].headline_template
    assert shows[-1].face == "awake"


def test_terminal_say_still_publishes_when_the_latest_state_is_absent():
    bus = make_bus()
    session = Session(config=AgentConfig(absent_limit_seconds=0.0))
    now_fn, _advance = make_clock(NIGHT)

    # Rule 5 escalates on this same reading. Publication therefore sees
    # `last_person_state == "absent"`, exercising the terminal exception
    # rather than relying on a later occupancy update.
    bus.publish(PersonState(source="perceive", state="absent", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)

    says = [event for _id, event in bus.read("say", "test", "c1", count=10)]
    assert says[-1].strategy == "escalate_phone"
    assert says[-1].text == "Someone is coming to help."


def test_escalated_speech_is_interpreted_and_reassured_after_gap():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(9)
    llm = FakeLLM(
        interpretations=[Interpretation(intent=Intent.UNCLEAR, distress=2)],
        compositions=[Composition(text="Help is on the way.")],
    )
    bus.publish(Utterance(source="listen", text="Help me", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn, llm=llm)

    assert session.phase == Phase.ESCALATED
    assert session.goal == "wait_for_caregiver"
    reassurances = [event for _, event in bus.read("say", "test", "c1")]
    assert reassurances[-1].strategy == "reassure_waiting"
    assert reassurances[-1].text == "Help is on the way."
    assert [name for name, _ in llm.calls] == ["interpret", "compose"]
    assert len(bus.read("notify", "test", "c1")) == 1


def test_severe_pain_escalates_with_pain_reply_and_skips_planner_speech():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0))
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.PAIN, distress=2)])
    bus.publish(
        Utterance(source="listen", text="My hip really hurts", confidence=0.9, duration_s=1)
    )
    run_once(bus, session, now_fn=now_fn, llm=llm)

    assert session.phase == Phase.ESCALATED
    assert session.reassurance_count == 0
    notify = [event for _, event in bus.read("notify", "test", "c1")]
    assert [(event.level, event.title, event.body) for event in notify] == [
        ("attention", "Pain reported", "They said they are in pain; please check in.")
    ]
    says = [event for _, event in bus.read("say", "test", "c1")]
    assert [(event.strategy, event.text) for event in says] == [
        ("acknowledge_pain", "I'm sorry it hurts; I'm letting someone know now.")
    ]
    decisions = [
        json.loads(event.detail)
        for _, event in bus.read("activity", "test", "c1", count=100)
        if event.kind == "decision"
    ]
    assert any(
        d.get("decision") == "said" and d.get("trigger") == "pain_reported" and d["reply"]
        for d in decisions
    )


def test_mild_pain_is_comforted_once_then_silent_without_planner_speech():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0))
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(
        interpretations=[Interpretation(intent=Intent.PAIN, distress=1)] * 3,
        plans=[Plan(next_strategy="soft_greeting", confidence=1.0)] * 3,
    )
    for utterance in ("My hip aches", "It still aches", "Does my hip still hurt?"):
        bus.publish(Utterance(source="listen", text=utterance, confidence=0.9, duration_s=1))
        run_once(bus, session, now_fn=now_fn, llm=llm)
        advance(9)
    says = [event for _, event in bus.read("say", "test", "c1")]
    assert [(event.strategy, event.text) for event in says] == [
        ("comfort_pain", "I'm sorry it hurts; I'm here with you."),
        ("comfort_pain", "I'm sorry it hurts; I'm here with you."),
    ]
    decisions = [
        json.loads(event.detail)
        for _, event in bus.read("activity", "test", "c1", count=100)
        if event.kind == "decision"
    ]
    assert any(
        d.get("decision") == "no_reply" and d["reason"] == "pain_acknowledged" for d in decisions
    )
    assert session.phase == Phase.ENGAGED


def test_restroom_progress_once_question_and_new_trip():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0, zone_confirm_readings=1))
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)
    bus.publish(PersonState(source="perceive", state="walking", confidence=0.9, zone="door"))
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.FINE, distress=0),
            Interpretation(intent=Intent.FINE, distress=0),
            Interpretation(intent=Intent.CONFUSED_TIME, distress=0),
            Interpretation(intent=Intent.FINE, distress=0),
        ]
    )
    for utterance in ("Nearly there", "Almost there now", "Where's the switch again?"):
        advance(17)
        bus.publish(Utterance(source="listen", text=utterance, confidence=0.9, duration_s=1))
        run_once(bus, session, now_fn=now_fn, llm=llm)
    assert [event.strategy for _, event in bus.read("say", "test", "c1")] == [
        "path_light",
        "acknowledge_progress",
        "path_light",
    ]
    assert [event.state for _, event in bus.read("light", "test", "c1")] == ["on"]
    decisions = [
        json.loads(event.detail)
        for _, event in bus.read("activity", "test", "c1", count=100)
        if event.kind == "decision"
    ]
    assert any(
        d.get("decision") == "no_reply" and d["reason"] == "progress_acknowledged"
        for d in decisions
    )

    back = session.propose_goal("return_to_bed", "test", now_fn())
    assert back is not None
    _publish_transition(bus, back, session, now_fn())
    again = session.propose_goal("restroom", "test", now_fn())
    assert again is not None
    _publish_transition(bus, again, session, now_fn())
    advance(17)
    bus.publish(
        Utterance(source="listen", text="I'm fine, going along", confidence=0.9, duration_s=1)
    )
    run_once(bus, session, now_fn=now_fn, llm=llm)
    assert [event.strategy for _, event in bus.read("say", "test", "c2")][
        -1
    ] == "acknowledge_progress"


def test_escalated_reassurance_cap_questions_distress_and_varied_composition():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.UNCLEAR, distress=distress) for distress in (1, 1, 1, 1, 3)
        ],
        compositions=[Composition(text="Help is on the way.") for _ in range(4)],
    )
    for utterance in (
        "I am here",
        "I am still here",
        "I am waiting",
        "Is anyone there?",
        "Help me",
    ):
        advance(9)
        bus.publish(Utterance(source="listen", text=utterance, confidence=0.9, duration_s=1))
        run_once(bus, session, now_fn=now_fn, llm=llm)

    reassurances = [
        event
        for _, event in bus.read("say", "test", "c1", count=20)
        if event.strategy == "reassure_waiting"
    ]
    assert len(reassurances) == 4
    assert session.reassurance_count == 4
    assert reassurances[0].text == "Help is on the way."
    assert reassurances[1].text != reassurances[0].text
    assert len({event.text.lower() for event in reassurances}) == 4
    decisions = [
        json.loads(event.detail)
        for _, event in bus.read("activity", "test", "c1", count=100)
        if event.kind == "decision"
    ]
    assert [decision for decision in decisions if decision["decision"] == "no_reply"] == [
        {
            "decision": "no_reply",
            "reason": "reassured_enough",
            "text": "I am waiting",
            "phase": "ESCALATED",
            "goal": "wait_for_caregiver",
            "reassurances": 2,
        }
    ]


def test_escalated_no_llm_fallback_obeys_cap_and_resets_on_new_escalation():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    for utterance in ("I am here", "Still here", "Waiting", "When will help come"):
        advance(9)
        bus.publish(Utterance(source="listen", text=utterance, confidence=0.9, duration_s=1))
        run_once(bus, session, now_fn=now_fn)
    reassurances = [
        event
        for _, event in bus.read("say", "test", "c1", count=20)
        if event.strategy == "reassure_waiting"
    ]
    assert len(reassurances) == 3
    assert len({event.text for event in reassurances}) == 3
    assert session.reassurance_count == 3

    session._apply(Phase.COOLDOWN, reason="test", now=now_fn())
    assert session.reassurance_count == 0
    session._apply(Phase.ESCALATED, reason="test", now=now_fn())
    assert session.reassurance_count == 0
    advance(9)
    bus.publish(Utterance(source="listen", text="I am here again", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn)
    assert session.reassurance_count == 1


def test_escalated_unavailable_interpretation_obeys_cap():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(interpretations=[None, None, None, None])
    for utterance in ("I am here", "Still here", "Waiting", "When will my daughter arrive"):
        advance(9)
        bus.publish(Utterance(source="listen", text=utterance, confidence=0.9, duration_s=1))
        run_once(bus, session, now_fn=now_fn, llm=llm)
    assert session.reassurance_count == 3
    assert [name for name, _ in llm.calls] == [
        "interpret",
        "compose",
        "interpret",
        "compose",
        "interpret",
        "interpret",
        "compose",
    ]


def test_escalated_restroom_need_keeps_alert_and_guides():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(9)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.NEED_RESTROOM, distress=0)])
    bus.publish(Utterance(source="listen", text="I need the toilet", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn, llm=llm)

    assert session.phase == Phase.ESCALATED
    assert session.goal == "wait_for_caregiver"
    assert [event.state for _, event in bus.read("light", "test", "c1")][-1] == "on"
    assert [event.strategy for _, event in bus.read("say", "test", "c1")][-1] == "path_light"
    assert len(bus.read("notify", "test", "c1")) == 1


def test_time_question_answers_without_advancing_ladder():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, _advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.CONFUSED_TIME, distress=0)])
    bus.publish(Utterance(source="listen", text="What time is it?", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn, llm=llm)

    assert session.phase == Phase.ENGAGED
    assert session._engine.current_id == "ambient_orient"
    assert session.strategy_index == 0
    says = [event for _, event in bus.read("say", "test", "c1")]
    assert len(says) == 1
    assert says[0].strategy == "orient_time_place"
    assert "11 o'clock at night" in says[0].text


def test_cooldown_up_answers_time_question_without_changing_session():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    session.phase = Phase.COOLDOWN
    session.session_id = "cooldown-session"
    session.strategy_index = 4
    session._cooldown_since = NIGHT
    bus.publish(PersonState(source="perceive", state="sitting_up", confidence=0.9, zone="bed"))
    run_once(bus, session, now_fn=now_fn)

    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.CONFUSED_TIME, distress=0)])
    bus.publish(Utterance(source="listen", text="What time is it?", confidence=0.9, duration_s=1))
    advance(1)
    assert run_once(bus, session, now_fn=now_fn, llm=llm) == []

    says = [event for _, event in bus.read("say", "test", "c1")]
    assert [event.strategy for event in says] == ["orient_time_place"]
    assert session.phase == Phase.COOLDOWN
    assert session.strategy_index == 4
    assert session._cooldown_since == NIGHT
    assert session.recent_utterances == ("What time is it?",)
    assert [name for name, _ in llm.calls] == ["interpret", "compose"]
    assert llm.calls[0][1]["last_turns"] == []
    assert bus.read("notify", "test", "c1") == []


def test_cooldown_in_bed_ignores_speech():
    bus = make_bus()
    session = Session(config=AgentConfig())
    session.phase = Phase.COOLDOWN
    session.session_id = "cooldown-session"
    bus.publish(PersonState(source="perceive", state="in_bed", confidence=0.9, zone="bed"))
    run_once(bus, session, now_fn=lambda: NIGHT)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.CONFUSED_TIME, distress=0)])
    bus.publish(Utterance(source="listen", text="What time is it?", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=lambda: NIGHT, llm=llm)

    assert bus.read("say", "test", "c1") == []
    assert session.recent_utterances == ()
    assert llm.calls == []


@pytest.mark.parametrize(
    ("intent", "distress", "text", "strategy"),
    [
        (Intent.NEED_RESTROOM, 0, "I need the toilet", "path_light"),
        (Intent.LOOKING_FOR_PERSON, 0, "Where is my daughter?", "reassure_waiting"),
        (Intent.UNCLEAR, 2, "Help me", "reassure_waiting"),
        (Intent.FINE, 0, "I'm fine", None),
        (Intent.UNCLEAR, 0, "Hmm", None),
    ],
)
def test_cooldown_direct_intents_only_reply(intent, distress, text, strategy):
    bus = make_bus()
    session = Session(config=AgentConfig())
    session.phase = Phase.COOLDOWN
    session.session_id = "cooldown-session"
    session._cooldown_since = NIGHT
    bus.publish(PersonState(source="perceive", state="sitting_up", confidence=0.9, zone="bed"))
    run_once(bus, session, now_fn=lambda: NIGHT)
    llm = FakeLLM(interpretations=[Interpretation(intent=intent, distress=distress)])
    bus.publish(Utterance(source="listen", text=text, confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=lambda: NIGHT + timedelta(seconds=1), llm=llm)

    assert [event.strategy for _, event in bus.read("say", "test", "c1")] == (
        [strategy] if strategy else []
    )
    assert session.phase == Phase.COOLDOWN
    assert session.goal == "return_to_bed"
    assert session._cooldown_since == NIGHT
    assert "plan" not in [name for name, _ in llm.calls]
    assert bus.read("notify", "test", "c1") == []
    assert not any(isinstance(event, GoalChanged) for _, event in bus.read("session", "test", "c1"))


def test_cooldown_up_without_llm_gives_one_reassuring_reply():
    bus = make_bus()
    session = Session(config=AgentConfig())
    session.phase = Phase.COOLDOWN
    session.session_id = "cooldown-session"
    bus.publish(PersonState(source="perceive", state="sitting_up", confidence=0.9, zone="bed"))
    run_once(bus, session, now_fn=lambda: NIGHT)
    bus.publish(Utterance(source="listen", text="Hello", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=lambda: NIGHT + timedelta(seconds=1))

    assert [event.strategy for _, event in bus.read("say", "test", "c1")] == ["reassure_waiting"]
    assert session.phase == Phase.COOLDOWN


def test_recent_utterance_counts_as_presence_after_camera_loses_person():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now = NIGHT
    session.on_person_state("standing", "other", now)
    session.on_utterance(now)
    session.on_person_state("absent", "other", now)
    session.on_utterance(now + timedelta(seconds=1))
    transition = session.propose_goal("restroom", "test", now + timedelta(seconds=1))
    assert transition is not None
    _publish_transition(bus, transition, session, now + timedelta(seconds=1))

    assert session.person_present(now + timedelta(seconds=30))
    assert not session.person_present(now + timedelta(seconds=32))
    assert [event.strategy for _, event in bus.read("say", "test", "c1")] == ["path_light"]


def test_speaking_face_waits_for_deferred_say():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    bus.read("show", "test", "baseline")
    advance(1)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.NEED_RESTROOM, distress=0)])
    bus.publish(Utterance(source="listen", text="I need the toilet", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn, llm=llm)
    assert [event.face for _, event in bus.read("show", "test", "before")][-1] == "awake"
    assert [event.strategy for _, event in bus.read("say", "test", "before")][
        -1
    ] == "escalate_phone"

    advance(8)
    run_once(bus, session, now_fn=now_fn)
    assert [event.face for _, event in bus.read("show", "test", "after")][-1] == "speaking"
    assert [event.strategy for _, event in bus.read("say", "test", "after")][-1] == "path_light"


def test_rejected_llm_composition_uses_the_rendered_caregiver_template(caplog):
    bus = make_bus()
    profile = PersonProfile(name="Jean", caregiver_name="Tom")
    strategy = next(item for item in DEFAULT_STRATEGIES if item.id == "validate_and_redirect")
    transition = Transition(
        phase=Phase.ENGAGED,
        session_id="session-1",
        goal="return_to_bed",
        strategy_index=3,
        reason="strategy_advanced",
        strategy=strategy,
    )
    unsafe = "Jean, I hear you, but Tom is here and we can rest now."
    llm = FakeLLM(compositions=[Composition(text=unsafe)])
    session = Session(config=AgentConfig())

    _maybe_publish_say(bus, transition, session, NIGHT, profile, llm)

    says = [event for _id, event in bus.read("say", "test", "c1", count=10)]
    assert says[-1].text == "It's alright, Jean, let's rest now and talk more in the morning."
    assert unsafe not in caplog.text
    activities = [event for _, event in bus.read("activity", "test", "c1")]
    assert all(isinstance(event, Activity) for event in activities)
    activities = [event for event in activities if event.kind != "decision"]
    assert [(event.kind, event.phase, event.ok) for event in activities] == [
        ("compose", "start", True),
        ("compose", "end", False),
    ]
    assert "rejected" in activities[-1].detail


def test_familiar_voice_publishes_clip_say_and_family_show(tmp_path, monkeypatch):
    clip_id = "family-message"
    with wave.open(str(Path(tmp_path) / f"{clip_id}.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 800)
    monkeypatch.setenv("VOICE_CLIP_DIR", str(tmp_path))
    familiar = dc_replace(
        next(s for s in DEFAULT_STRATEGIES if s.id == FAMILIAR_VOICE_ID),
        enabled=True,
        clip_id=clip_id,
    )
    session = Session(config=AgentConfig(observe_seconds=1.0), strategies=[familiar])
    bus = make_bus()
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)

    say = [event for _id, event in bus.read("say", "test", "c1", count=10)][-1]
    show = [event for _id, event in bus.read("show", "test", "c1", count=10)][-1]
    assert say.strategy == FAMILIAR_VOICE_ID
    assert say.clip_id == clip_id
    assert say.interruptible is True
    assert show.photo_id == "demo_family"


def test_a_say_rejected_by_rule_3_is_not_published_and_falls_back_to_silence():
    bus = make_bus()
    strategies = small_strategies(dwell=1.0)
    # Two consecutive sentences: rule 3 rejects this outright.
    strategies[1] = dc_replace(strategies[1], say_template="It is night. Let's rest.")
    session = Session(config=AgentConfig(observe_seconds=1.0), strategies=strategies)
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)  # -> ENGAGED, s1

    advance(2)  # s1's 1s dwell elapses -> advance to s2, the bad template
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)

    assert bus.read("say", "test", "c1", count=10) == []
    # The `Show` still publishes -- silence, not a crash or a missing update.
    show_events = bus.read("show", "test", "c1", count=10)
    assert len(show_events) > 0


def test_the_minimum_say_gap_is_enforced_across_strategy_advances():
    bus = make_bus()
    strategies = small_strategies(dwell=1.0)
    session = Session(
        config=AgentConfig(observe_seconds=1.0, say_min_gap_seconds=8.0), strategies=strategies
    )
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)  # -> ENGAGED, s1 (no say)

    advance(2)  # -> s2 (has a say), published
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    first_says = [e for _id, e in bus.read("say", "test", "c1", count=10)]
    assert len(first_says) == 1

    advance(2)  # -> s3 (has a say), only 2s after the last one: rejected
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    second_says = [e for _id, e in bus.read("say", "test", "c1", count=10)]
    assert second_says == []


def _session_with_pending_say(bus, *, gap=8.0):
    terminal = next(item for item in DEFAULT_STRATEGIES if item.id == ESCALATE_PHONE_ID)
    session = Session(
        config=AgentConfig(say_min_gap_seconds=gap),
        strategies=[*small_strategies(dwell=100.0, n=4), terminal],
    )
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=lambda: NIGHT)
    bus.publish(Utterance(source="listen", text="Hello", confidence=0.9, duration_s=1.0))
    run_once(bus, session, now_fn=lambda: NIGHT)
    for offset, strategy in ((1, "s2"), (2, "s3")):
        now = NIGHT + timedelta(seconds=offset)
        transition = session.propose_strategy(strategy, now)
        assert transition is not None
        _publish_transition(bus, transition, session, now)
    return session


def test_deferred_say_publishes_on_quiet_tick_without_duplicate_show(caplog):
    caplog.set_level("INFO", logger="agent")
    bus = make_bus()
    session = _session_with_pending_say(bus)
    assert [event.strategy for _, event in bus.read("say", "test", "first")] == ["s2"]
    assert bus.read("show", "test", "first")

    run_once(bus, session, now_fn=lambda: NIGHT + timedelta(seconds=9))

    assert [event.strategy for _, event in bus.read("say", "test", "second")] == ["s3"]
    assert len(bus.read("show", "test", "second")) == 0
    assert "deferred Say" in caplog.text
    assert "published deferred Say" in caplog.text


def test_newer_say_supersedes_pending_say():
    bus = make_bus()
    session = _session_with_pending_say(bus)
    now = NIGHT + timedelta(seconds=3)
    transition = session.propose_strategy("s4", now)
    assert transition is not None
    _publish_transition(bus, transition, session, now)

    run_once(bus, session, now_fn=lambda: NIGHT + timedelta(seconds=9))

    assert [event.strategy for _, event in bus.read("say", "test", "c1")] == ["s2", "s4"]


def test_immediate_terminal_say_cancels_pending_say():
    bus = make_bus()
    session = _session_with_pending_say(bus)
    now = NIGHT + timedelta(seconds=3)
    transition = session.on_person_state("on_floor", "other", now)
    assert transition is not None
    _publish_transition(bus, transition, session, now)
    run_once(bus, session, now_fn=lambda: NIGHT + timedelta(seconds=10))

    assert session._pending_say is None
    assert [event.strategy for _, event in bus.read("say", "test", "c1")] == [
        "s2",
        "escalate_phone",
    ]


def test_pending_say_drops_after_goal_change_or_session_end(caplog):
    caplog.set_level("INFO", logger="agent")
    for ending in ("goal", "session"):
        bus = make_bus()
        session = _session_with_pending_say(bus)
        now = NIGHT + timedelta(seconds=3)
        if ending == "goal":
            session.propose_goal("restroom", "test", now)
        else:
            session.on_person_state("in_bed", "bed", now)
            session.on_person_state("in_bed", "bed", now + timedelta(seconds=20))
        run_once(
            bus,
            session,
            now_fn=lambda: NIGHT + timedelta(seconds=24 if ending == "session" else 12),
        )
        assert [event.strategy for _, event in bus.read("say", "test", "c1")] == ["s2"]
    assert "dropped deferred Say" in caplog.text


def test_pending_say_expires_after_max_age():
    bus = make_bus()
    session = _session_with_pending_say(bus, gap=60.0)
    run_once(bus, session, now_fn=lambda: NIGHT + timedelta(seconds=33))
    assert session._pending_say is None
    assert [event.strategy for _, event in bus.read("say", "test", "c1")] == ["s2"]


def _decisions(bus, include_said: bool = False):
    decisions = [
        (event, json.loads(event.detail))
        for _, event in bus.read("activity", "test", "decisions")
        if isinstance(event, Activity) and event.kind == "decision"
    ]
    return [d for d in decisions if include_said or d[1]["decision"] != "said"]


def test_said_decision_names_the_trigger_for_direct_and_deferred_says():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.CONFUSED_TIME, distress=0)])
    bus.publish(Utterance(source="listen", text="What time is it?", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn, llm=llm)

    said = [d for _, d in _decisions(bus, include_said=True) if d["decision"] == "said"]
    assert said and said[-1]["strategy"] == "orient_time_place"
    assert said[-1]["trigger"] == "utterance_reply"
    assert said[-1]["reply"] is True
    assert said[-1]["deferred_s"] == 0


def test_said_decision_marks_a_ladder_step_as_not_a_reply():
    bus = make_bus()
    session = Session(config=AgentConfig(observe_seconds=1.0), strategies=list(DEFAULT_STRATEGIES))
    now_fn, advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    for _ in range(120):
        run_once(bus, session, now_fn=now_fn)
        advance(1)
    said = [d for _, d in _decisions(bus, include_said=True) if d["decision"] == "said"]
    assert said, "the ladder should have spoken"
    assert all(d["reply"] is False for d in said)


def test_pending_say_drop_decisions_include_age_and_reason():
    for reason in ("max_age", "strategy_changed"):
        bus = make_bus()
        session = _session_with_pending_say(bus, gap=60.0)
        pending = session._pending_say
        assert pending is not None
        if reason == "strategy_changed":
            session.strategy_index += 1
            now = NIGHT + timedelta(seconds=9)
        else:
            now = NIGHT + timedelta(seconds=33)
        run_once(bus, session, now_fn=lambda: now)

        decisions = _decisions(bus)
        assert len(decisions) == 1
        event, detail = decisions[0]
        assert (event.service, event.phase, event.ok, event.duration_ms) == (
            "agent",
            "end",
            False,
            None,
        )
        assert detail == {
            "decision": "pending_say_dropped",
            "reason": reason,
            "text": pending.event.text,
            "strategy": pending.event.strategy,
            "direct": False,
            "age_s": (now - pending.queued_at).total_seconds(),
            "phase": session.phase.value,
            "goal": session.goal,
        }


def test_pending_say_veto_records_veto_before_drop():
    bus = make_bus()
    session = _session_with_pending_say(bus)
    run_once(
        bus,
        session,
        now_fn=lambda: NIGHT + timedelta(seconds=9),
        profile=PersonProfile(things_to_avoid=("Do not mention s3",)),
    )

    decisions = _decisions(bus)
    assert [detail["decision"] for _, detail in decisions] == ["vetoed", "pending_say_dropped"]
    assert decisions[0][1]["rule"] == "no_avoided_term"
    assert decisions[0][1]["event_type"] == "Say"
    assert decisions[1][1]["reason"] == "vetoed"
    assert session._pending_say is None


def test_decision_publish_failure_does_not_interrupt_pending_drop(monkeypatch):
    bus = make_bus()
    session = _session_with_pending_say(bus, gap=60.0)
    publish = bus.publish

    def fail_decision(event, **kwargs):
        if isinstance(event, Activity) and event.kind == "decision":
            raise OSError("activity stream unavailable")
        return publish(event, **kwargs)

    monkeypatch.setattr(bus, "publish", fail_decision)
    run_once(bus, session, now_fn=lambda: NIGHT + timedelta(seconds=33))
    assert session._pending_say is None


def test_interpretation_publishes_decision_record():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now_fn, _advance = make_clock(NIGHT)
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.CONFUSED_TIME, distress=1)])
    bus.publish(Utterance(source="listen", text="What time is it?", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn, llm=llm)

    interpreted = [d for _, d in _decisions(bus) if d["decision"] == "interpreted"]
    assert len(interpreted) == 1
    assert interpreted[0]["intent"] == "confused_time"
    assert interpreted[0]["distress"] == 1
    assert interpreted[0]["text"] == "What time is it?"


def test_escalation_show_reflects_escalate_phone_strategy():
    bus = make_bus()
    session = Session(config=AgentConfig(floor_limit_seconds=0.0))
    now_fn, _advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)

    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)

    show_events = [e for _id, e in bus.read("show", "test", "c1", count=10)]
    assert show_events[-1].headline == "Someone is coming to help"
    say_events = [e for _id, e in bus.read("say", "test", "c1", count=10)]
    assert any(s.text == "Someone is coming to help." for s in say_events)


def test_the_escalation_say_survives_a_recent_ordinary_say():
    """Review finding: the minimum-gap rule applied to `escalate_phone`
    too, so a fall seconds after a strategy spoke published the critical
    `Notify` but silenced the one sentence telling the person on the floor
    that help was coming."""
    bus = make_bus()
    session = Session(
        config=AgentConfig(observe_seconds=1.0, say_min_gap_seconds=8.0, floor_limit_seconds=0.0),
        strategies=list(DEFAULT_STRATEGIES),
    )
    now_fn, advance = make_clock(NIGHT)

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    advance(2)
    run_once(bus, session, now_fn=now_fn)  # -> ENGAGED, ambient_orient (silent)

    # Walk to the first rung that actually speaks.
    for _ in range(40):
        advance(31)
        bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
        run_once(bus, session, now_fn=now_fn)
        if [e for _id, e in bus.read("say", "test", "c1", count=10)]:
            break
    else:
        raise AssertionError("no strategy ever spoke")

    # One second later the person is on the floor: rule 5 escalates.
    advance(1)
    bus.publish(PersonState(source="perceive", state="on_floor", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=now_fn)
    assert session.phase.value == "ESCALATED"

    says = [e for _id, e in bus.read("say", "test", "c2", count=50)]
    assert says, "the escalation Say was dropped by the minimum-gap rule"
    assert says[-1].strategy == "escalate_phone"
    assert says[-1].text == "Someone is coming to help."
    assert says[-1].interruptible is False
