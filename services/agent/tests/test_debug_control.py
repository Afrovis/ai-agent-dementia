"""Live debug overrides use bus events and an injected clock."""

from datetime import UTC, datetime, timedelta

from nc_shared.bus import FakeBus
from nc_shared.events import (
    BedZoneStatus,
    DebugControl,
    LightCommand,
    PersonState,
    ResetSession,
    Say,
    Show,
    Utterance,
)

from agent.config import AgentConfig
from agent.llm import FakeLLM, Intent, Interpretation
from agent.main import (
    DEBUG_GROUP,
    DEBUG_STREAM,
    PERSON_GROUP,
    PERSON_STREAM,
    _publish_transition,
    run_once,
)
from agent.rules import Phase
from agent.session import Session, Transition
from agent.strategies import DEFAULT_STRATEGIES


def make_clock(start: datetime):
    box = {"now": start}

    def now_fn():
        return box["now"]

    def advance(seconds: float):
        box["now"] += timedelta(seconds=seconds)

    return now_fn, advance


def setup(start: datetime = datetime(2026, 9, 22, 14, tzinfo=UTC)):
    bus = FakeBus()
    bus.ensure_group(DEBUG_STREAM, DEBUG_GROUP)
    bus.ensure_group(PERSON_STREAM, PERSON_GROUP)
    for stream in ("session", "say", "show", "light"):
        bus.ensure_group(stream, "test")
    session = Session(config=AgentConfig())
    now_fn, advance = make_clock(start)
    return bus, session, now_fn, advance


def request(bus, now, *, offset=0.0, forced=False, source="embodiment"):
    bus.publish(
        DebugControl(source=source, ts=now, time_offset_hours=offset, force_in_bed=forced),
        maxlen=100,
    )


def person(bus, state, zone="other", scene_note=None):
    bus.publish(
        PersonState(
            source="perceive", state=state, zone=zone, confidence=0.9, scene_note=scene_note
        )
    )


def test_offset_starts_only_the_night_session_and_leaves_timers_unshifted():
    bus, session, now_fn, advance = setup()
    person(bus, "sitting_up", "bed")
    assert run_once(bus, session, now_fn=now_fn) == []

    request(bus, now_fn(), offset=13)
    person(bus, "sitting_up", "bed")
    states = run_once(bus, session, now_fn=now_fn)
    assert [state.phase for state in states] == ["OBSERVING"]
    assert session.wall_clock(now_fn()).hour == 3

    advance(19)
    assert run_once(bus, session, now_fn=now_fn) == []
    advance(1)
    assert [state.phase for state in run_once(bus, session, now_fn=now_fn)] == ["ENGAGED"]
    assert bus.pending(DEBUG_STREAM, DEBUG_GROUP) == []


def test_rendered_time_words_use_shifted_hour():
    bus, session, now_fn, _ = setup()
    request(bus, now_fn(), offset=13)
    run_once(bus, session, now_fn=now_fn)
    session.phase = Phase.ENGAGED
    session.session_id = "test-session"
    session._last_person_state = "standing"
    strategy = next(item for item in DEFAULT_STRATEGIES if item.id == "orient_time_place")
    transition = Transition(
        phase=Phase.ENGAGED,
        session_id=session.session_id,
        goal=session.goal,
        strategy_index=0,
        reason="test",
        strategy=strategy,
    )
    _publish_transition(bus, transition, session, now_fn())
    says = [event for _, event in bus.read("say", "test", "test") if isinstance(event, Say)]
    shows = [event for _, event in bus.read("show", "test", "test") if isinstance(event, Show)]
    assert says and "the middle of the night" in says[-1].text
    assert shows and "3 o'clock at night" in shows[-1].body


def test_force_and_release_use_real_person_state_and_publish_transitions():
    bus, session, now_fn, _ = setup(datetime(2026, 9, 22, 23, tzinfo=UTC))
    person(bus, "standing")
    assert [state.phase for state in run_once(bus, session, now_fn=now_fn)] == ["OBSERVING"]

    request(bus, now_fn(), forced=True)
    assert [state.phase for state in run_once(bus, session, now_fn=now_fn)] == ["IDLE"]
    person(bus, "standing", scene_note="doorway")
    assert run_once(bus, session, now_fn=now_fn) == []
    assert session.last_person_state == "in_bed"
    assert session.last_scene_note == "doorway"

    request(bus, now_fn(), forced=False)
    assert [state.phase for state in run_once(bus, session, now_fn=now_fn)] == ["OBSERVING"]
    assert session.last_person_state == "standing"


def test_force_before_person_keeps_standing_from_starting_session():
    bus, session, now_fn, _ = setup(datetime(2026, 9, 22, 23, tzinfo=UTC))
    request(bus, now_fn(), forced=True)
    person(bus, "standing")
    assert run_once(bus, session, now_fn=now_fn) == []
    assert session.phase == Phase.IDLE


def test_forced_in_bed_silences_cooldown_reply():
    bus, session, now_fn, _ = setup(datetime(2026, 9, 22, 23, tzinfo=UTC))
    session.phase = Phase.COOLDOWN
    session.session_id = "cooldown-session"
    person(bus, "sitting_up", "bed")
    run_once(bus, session, now_fn=now_fn)
    request(bus, now_fn(), forced=True)
    run_once(bus, session, now_fn=now_fn)
    llm = FakeLLM(interpretations=[Interpretation(intent=Intent.CONFUSED_TIME, distress=0)])
    bus.publish(Utterance(source="listen", text="What time is it?", confidence=0.9, duration_s=1))
    run_once(bus, session, now_fn=now_fn, llm=llm)

    assert session.phase == Phase.COOLDOWN
    assert session.last_person_state == "in_bed"
    assert bus.read("say", "test", "c1") == []
    assert llm.calls == []


def test_stale_requests_and_agent_echoes_are_acked_and_ignored():
    bus, session, now_fn, _ = setup()
    request(bus, now_fn() - timedelta(seconds=61), offset=13)
    request(bus, now_fn(), offset=12, source="agent")
    bus.publish(BedZoneStatus(source="perceive", has_bed=False), maxlen=100)
    assert run_once(bus, session, now_fn=now_fn) == []
    assert session.debug_overrides.time_offset_hours == 0
    assert bus.pending(DEBUG_STREAM, DEBUG_GROUP) == []
    assert len(bus.read(DEBUG_STREAM, "observer", "test")) == 3


def test_applied_state_is_echoed_from_agent():
    bus, session, now_fn, _ = setup()
    request(bus, now_fn(), offset=13, forced=True)
    run_once(bus, session, now_fn=now_fn)
    events = [event for _, event in bus.read(DEBUG_STREAM, "observer", "test")]
    echoes = [
        event for event in events if isinstance(event, DebugControl) and event.source == "agent"
    ]
    assert len(echoes) == 1
    assert echoes[0].time_offset_hours == 13
    assert echoes[0].force_in_bed is True
    run_once(bus, session, now_fn=now_fn)
    assert len([event for _, event in bus.read(DEBUG_STREAM, "observer", "test")]) == 0


def test_reset_clears_escalated_session_and_starts_new_one_on_next_reading():
    bus, session, now_fn, _ = setup(datetime(2026, 9, 22, 23, tzinfo=UTC))
    ids = iter(("first", "second"))
    session.id_fn = lambda: next(ids)
    person(bus, "sitting_up", "bed")
    run_once(bus, session, now_fn=now_fn)
    assert session.session_id == "first"
    session.phase = Phase.ESCALATED
    session.goal = "wait_for_caregiver"
    session._last_real_person = ("in_bed", "bed")
    session._pending_zone = "door"
    bus.publish(ResetSession(source="embodiment", ts=now_fn()), maxlen=100)
    states = run_once(bus, session, now_fn=now_fn)
    assert [state.phase for state in states] == ["IDLE"]
    assert session.session_id is None
    assert session.goal == "return_to_bed"
    assert session._pending_zone is None
    assert any(
        isinstance(event, LightCommand) and event.state == "off"
        for _, event in bus.read("light", "test", "test")
    )
    person(bus, "sitting_up", "bed")
    assert [state.phase for state in run_once(bus, session, now_fn=now_fn)] == ["OBSERVING"]
    assert session.session_id == "second"


def test_reset_from_cooldown_reapplies_real_person_and_preserves_overrides():
    bus, session, now_fn, _ = setup(datetime(2026, 9, 22, 23, tzinfo=UTC))
    session.phase = Phase.COOLDOWN
    session._cooldown_since = now_fn()
    session._last_real_person = ("sitting_up", "other")
    session.debug_overrides.time_offset_hours = 2
    bus.publish(ResetSession(source="embodiment", ts=now_fn()), maxlen=100)
    states = run_once(bus, session, now_fn=now_fn)
    assert [state.phase for state in states] == ["IDLE", "OBSERVING"]
    assert session.debug_overrides.time_offset_hours == 2
    assert session._cooldown_since is None
    assert session.last_person_state == "sitting_up"


def test_stale_reset_is_ignored():
    bus, session, now_fn, _ = setup()
    session.phase = Phase.COOLDOWN
    bus.publish(ResetSession(source="embodiment", ts=now_fn() - timedelta(seconds=61)), maxlen=100)
    assert run_once(bus, session, now_fn=now_fn) == []
    assert session.phase == Phase.COOLDOWN
    assert bus.pending(DEBUG_STREAM, DEBUG_GROUP) == []


def test_reset_keeps_force_in_bed_and_suppresses_real_sitting_up():
    bus, session, now_fn, _ = setup(datetime(2026, 9, 22, 23, tzinfo=UTC))
    session.phase = Phase.COOLDOWN
    session._last_real_person = ("sitting_up", "other")
    session.debug_overrides.force_in_bed = True
    bus.publish(ResetSession(source="embodiment", ts=now_fn()), maxlen=100)
    assert [state.phase for state in run_once(bus, session, now_fn=now_fn)] == ["IDLE"]
    assert session.debug_overrides.force_in_bed is True
    assert session._last_real_person == ("sitting_up", "other")
    assert session.last_person_state == "in_bed"
