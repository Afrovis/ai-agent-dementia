"""End-to-end wiring checks for issue #15's injectable LLM boundary."""

from dataclasses import replace
from datetime import datetime, timedelta

from nc_shared.bus import FakeBus
from nc_shared.events import Notify, PersonState, Say, Utterance

from agent.config import AgentConfig
from agent.llm import Composition, FakeLLM, Intent, Interpretation, Plan
from agent.main import PERSON_GROUP, PERSON_STREAM, UTTERANCE_GROUP, UTTERANCE_STREAM, run_once
from agent.profile import PersonProfile
from agent.session import Session
from agent.strategies import DEFAULT_STRATEGIES

NIGHT = datetime(2026, 1, 1, 23, 0)


def make_bus() -> FakeBus:
    bus = FakeBus()
    for stream, group in (
        (PERSON_STREAM, PERSON_GROUP),
        (UTTERANCE_STREAM, UTTERANCE_GROUP),
        ("session", "test"),
        ("notify", "test"),
        ("say", "test"),
    ):
        bus.ensure_group(stream, group)
    return bus


def test_interpretation_goal_and_planner_are_advisory_and_rule_checked():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now = NIGHT
    llm = FakeLLM(
        interpretations=[Interpretation(intent=Intent.NEED_RESTROOM, distress=0)],
        # This follows the interpretation's legal root -> restroom move,
        # so restroom -> drink_water must be rejected by validate_goal.
        plans=[Plan(goal_change="drink_water", confidence=0.9)],
    )
    profile = PersonProfile(name="Jean", night_themes=("looking for work",))

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=lambda: now, profile=profile, llm=llm)
    bus.publish(
        Utterance(source="listen", text="I need the toilet", confidence=0.9, duration_s=1.0)
    )
    run_once(
        bus,
        session,
        now_fn=lambda: now + timedelta(seconds=1),
        profile=profile,
        llm=llm,
    )

    assert session.goal == "restroom"
    assert [call[0] for call in llm.calls] == ["interpret", "plan"]
    assert llm.calls[0][1]["last_turns"] == []
    assert llm.calls[0][1]["profile"]["night_themes"] == ["looking for work"]
    assert llm.calls[1][1]["profile"] == profile.prompt_data()
    planner_state = llm.calls[1][1]["session_state"]
    assert planner_state["current_strategy"] == "path_light"
    assert planner_state["strategy_order"][0] == "ambient_orient"
    assert "drink_water" in planner_state["allowed_goals"]


def test_planner_may_select_only_the_exact_next_available_strategy():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now = NIGHT
    llm = FakeLLM(
        plans=[
            Plan(next_strategy="soft_greeting", confidence=0.9),
            # Skipping straight to rung five is not an allowed proposal.
            Plan(next_strategy="guided_return", confidence=0.9),
        ]
    )

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=lambda: now, llm=llm)
    bus.publish(Utterance(source="listen", text="hello", confidence=0.9, duration_s=1.0))
    run_once(bus, session, now_fn=lambda: now + timedelta(seconds=1), llm=llm)
    assert session.strategy_index == 1

    bus.publish(Utterance(source="listen", text="please", confidence=0.9, duration_s=1.0))
    run_once(bus, session, now_fn=lambda: now + timedelta(seconds=2), llm=llm)
    assert session.strategy_index == 1


def test_second_distressed_interpretation_escalates_through_session_rules():
    bus = make_bus()
    session = Session(config=AgentConfig())
    now = NIGHT
    llm = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.UNCLEAR, distress=2),
            Interpretation(intent=Intent.UNCLEAR, distress=3),
        ],
        plans=[None, Plan(goal_change="return_to_bed", confidence=0.9)],
    )

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=lambda: now, llm=llm)
    for offset, text in ((1, "Help me"), (2, "I am scared")):
        bus.publish(Utterance(source="listen", text=text, confidence=0.9, duration_s=1.0))
        run_once(
            bus,
            session,
            now_fn=lambda offset=offset: now + timedelta(seconds=offset),
            llm=llm,
        )

    assert session.phase.value == "ESCALATED"
    assert session.goal == "wait_for_caregiver"
    assert [name for name, _ in llm.calls].count("plan") == 1
    notifications = bus.read("notify", "test", "c", count=10)
    assert any(
        isinstance(event, Notify) and event.level == "attention" for _, event in notifications
    )


def test_validate_and_redirect_uses_validated_llm_composition():
    bus = make_bus()
    # Zero dwell lets ticks reach strategy four without waiting in real time.
    strategies = [
        replace(item, dwell_seconds=0.0, cooldown_seconds=0.0) for item in DEFAULT_STRATEGIES
    ]
    session = Session(config=AgentConfig(), strategies=strategies)
    llm = FakeLLM(compositions=[Composition(text="I hear you, and we can rest together now.")])
    profile = PersonProfile(name="Jean", things_to_avoid=("mentioning hospital",))
    now = NIGHT

    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="other"))
    run_once(bus, session, now_fn=lambda: now, profile=profile, llm=llm)
    bus.publish(
        Utterance(source="listen", text="Where is my mother", confidence=0.9, duration_s=1.0)
    )
    run_once(
        bus,
        session,
        now_fn=lambda: now + timedelta(seconds=1),
        profile=profile,
        llm=llm,
    )
    # Preserve the deterministic eight-second silence gap between each
    # ordinary strategy's Say and the composed fourth-rung Say.
    for offset in (10, 20):
        run_once(
            bus,
            session,
            now_fn=lambda offset=offset: now + timedelta(seconds=offset),
            profile=profile,
            llm=llm,
        )

    say_events = bus.read("say", "test", "c", count=10)
    assert any(
        isinstance(event, Say)
        and event.strategy == "validate_and_redirect"
        and event.text == "I hear you, and we can rest together now."
        for _, event in say_events
    )
    compose_calls = [payload for name, payload in llm.calls if name == "compose"]
    assert compose_calls[-1]["latest_utterance"] == "Where is my mother"
    assert compose_calls[-1]["goal"] == session.goal
    assert "{" not in compose_calls[-1]["caregiver_phrase_template"]
    assert compose_calls[-1]["profile"]["things_to_avoid"] == ["mentioning hospital"]


def test_idle_utterance_is_not_sent_to_the_llm():
    bus = make_bus()
    session = Session(config=AgentConfig())
    llm = FakeLLM(
        interpretations=[Interpretation(intent=Intent.UNCLEAR, distress=3)],
        plans=[Plan(goal_change="comfort", confidence=0.9)],
    )

    bus.publish(
        Utterance(source="listen", text="background speech", confidence=0.9, duration_s=1.0)
    )
    run_once(bus, session, now_fn=lambda: NIGHT, llm=llm)

    assert session.phase.value == "IDLE"
    assert session.recent_utterances == ()
    assert llm.calls == []
