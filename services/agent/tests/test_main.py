"""Tests for `agent.main`: wiring `agent.session.Session` to the bus.

Uses `FakeBus` throughout, per HANDOFF.md section 4: no Redis, camera,
mic, or Ollama. `now_fn` is always an explicit, advancing fixed clock, so
nothing here sleeps for a real duration.
"""

import wave
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta
from pathlib import Path

from nc_shared.bus import FakeBus
from nc_shared.events import (
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
from agent.main import (
    PERSON_GROUP,
    PERSON_STREAM,
    UTTERANCE_GROUP,
    UTTERANCE_STREAM,
    _publish_cloud_call,
    maybe_emit_health,
    maybe_emit_session_heartbeat,
    run_once,
)
from agent.session import Session
from agent.strategies import DEFAULT_STRATEGIES, FAMILIAR_VOICE_ID

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
