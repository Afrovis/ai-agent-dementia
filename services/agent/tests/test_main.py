"""Tests for `agent.main`: wiring `agent.session.Session` to the bus.

Uses `FakeBus` throughout, per HANDOFF.md section 4: no Redis, camera,
mic, or Ollama. `now_fn` is always an explicit, advancing fixed clock, so
nothing here sleeps for a real duration.
"""

from dataclasses import replace as dc_replace
from datetime import datetime, timedelta

from nc_shared.bus import FakeBus
from nc_shared.events import GoalChanged, Notify, PersonState, Say, SessionState, Show, Utterance

from agent.config import AgentConfig
from agent.main import (
    PERSON_GROUP,
    PERSON_STREAM,
    UTTERANCE_GROUP,
    UTTERANCE_STREAM,
    maybe_emit_health,
    maybe_emit_session_heartbeat,
    run_once,
)
from agent.session import Session
from agent.strategies import DEFAULT_STRATEGIES

NIGHT = datetime(2026, 1, 1, 23, 0)


def make_bus() -> FakeBus:
    bus = FakeBus()
    bus.ensure_group(PERSON_STREAM, PERSON_GROUP)
    bus.ensure_group(UTTERANCE_STREAM, UTTERANCE_GROUP)
    bus.ensure_group("session", "test")
    bus.ensure_group("notify", "test")
    bus.ensure_group("show", "test")
    bus.ensure_group("say", "test")
    return bus


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
