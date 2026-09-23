"""Regression cases for the deterministic veto and its publication boundary."""

import json
import logging
from datetime import datetime, timedelta

import pytest
from nc_shared.events import Say, SessionState, Show

from agent.config import AgentConfig
from agent.main import _maybe_publish_say, _publish_transition
from agent.profile import PersonProfile
from agent.rules import Phase
from agent.session import Session, Transition
from agent.strategies import DEFAULT_STRATEGIES
from agent.veto import Proposal, VetoContext, avoid_terms, check
from tests.test_main import make_bus

NIGHT = datetime(2026, 1, 1, 23, 0)
AVOID = ("Do not mention the hospital", "Avoid loud or urgent language")


def context(
    *,
    utterances=(),
    person_state="standing",
    phase="ENGAGED",
    goal="return_to_bed",
    settled=False,
    night=True,
):
    return VetoContext(phase, goal, person_state, utterances, settled, night, AVOID)


@pytest.mark.parametrize(
    ("proposal", "facts", "rule", "clause"),
    [
        pytest.param(
            Proposal("strategy", "guided_return"),
            context(utterances=("I need to pee.",)),
            "no_redirect_from_toilet_need",
            "TOIL-01",
            id="restroom-03::first-response",
        ),
        pytest.param(
            Proposal("strategy", "guided_return"),
            context(utterances=("I need the toilet.", "Where's the loo? This isn't right.")),
            "no_redirect_from_toilet_need",
            "TOIL-01",
            id="restroom-04::wrong-way",
        ),
        pytest.param(
            Proposal("strategy", "guided_return"),
            context(
                utterances=("Oh no. I've wet myself. I'm so sorry.", "I don't know what to do.")
            ),
            "no_redirect_from_toilet_need",
            "TOIL-01",
            id="restroom-05::needs-help",
        ),
        pytest.param(
            Proposal("strategy", "guided_return"),
            context(
                utterances=("Oh no. I've wet myself. I'm so sorry.",), person_state="sitting_up"
            ),
            "no_redirect_from_toilet_need",
            "TOIL-01",
            id="restroom-05::accident",
        ),
        pytest.param(
            Proposal("strategy", "guided_return"),
            context(utterances=("I need the loo again.",), person_state="walking", goal="restroom"),
            "no_redirect_from_toilet_need",
            "TOIL-01",
            id="restroom-06::second-trip",
        ),
        pytest.param(
            Proposal("strategy", "guided_return"),
            context(utterances=("I'm so cold.", "I can't get warm."), person_state="sitting_up"),
            "no_redirect_from_stated_need",
            "NICE-01",
            id="distress-pain-05::still-cold",
        ),
        pytest.param(
            Proposal("strategy", "guided_return"),
            context(utterances=("Help!", "Help! Help!", "Help me!"), person_state="sitting_up"),
            "no_redirect_from_stated_need",
            "NICE-01",
            id="distress-pain-06::repeated-calls",
        ),
        pytest.param(
            Proposal("strategy", "orient_time_place"),
            context(utterances=("I need to pee.",), person_state="in_bed", settled=True),
            "no_orienting_a_settling_person",
            "NICE-05",
            id="restroom-03::back-in-bed",
        ),
        pytest.param(
            Proposal("strategy", "orient_time_place"),
            context(person_state="in_bed", settled=True),
            "no_orienting_a_settling_person",
            "NICE-05",
            id="restroom-04::back-in-bed",
        ),
        pytest.param(
            Proposal("strategy", "orient_time_place"),
            context(utterances=("Good morning!",), person_state="walking", night=False),
            "no_night_orientation_by_day",
            "AA-03",
            id="false-alarm-07::morning",
        ),
        pytest.param(
            Proposal(
                "say",
                "validate_and_redirect",
                text="It's alright, let's rest now and talk more in the morning.",
            ),
            context(person_state="in_bed", settled=True),
            "silence_when_settled",
            "NICE-05",
            id="disorientation-05::settled",
        ),
        pytest.param(
            Proposal("say", "orient_time_place", text="Hello Jean, it's night-time."),
            context(person_state="in_bed", settled=True),
            "silence_when_settled",
            "NICE-05",
            id="disorientation-06::settles",
        ),
        pytest.param(
            Proposal(
                "say",
                "validate_and_redirect",
                text="It's alright, let's rest now and talk more in the morning.",
            ),
            context(person_state="in_bed", settled=True),
            "silence_when_settled",
            "NICE-05",
            id="distress-pain-04::settles",
        ),
        pytest.param(
            Proposal(
                "say",
                "validate_and_redirect",
                text="Do you remember that the children are grown up now.",
            ),
            context(),
            "no_memory_question",
            "VAL-01",
            id="disorientation-05::breakfast",
        ),
        pytest.param(
            Proposal(
                "say",
                "escalate_phone",
                text="Someone is coming to take you to the hospital.",
                terminal=True,
            ),
            context(person_state="sitting_up"),
            "no_avoided_term",
            "NICE-04",
            id="distress-pain-03::chest-pain",
        ),
        pytest.param(
            Proposal(
                "say", "validate_and_redirect", text="You are safe at home, not in the hospital."
            ),
            context(person_state="sitting_up"),
            "no_avoided_term",
            "NICE-04",
            id="distress-pain-04::nightmare",
        ),
    ],
)
def test_regression_denials(proposal, facts, rule, clause):
    verdict = check(proposal, facts)
    assert verdict.allowed is False
    assert verdict.rule == rule
    assert verdict.clause == clause


@pytest.mark.parametrize(
    ("proposal", "facts"),
    [
        (Proposal("strategy", "guided_return"), context(utterances=("Where is my husband?",))),
        (Proposal("strategy", "path_light"), context(utterances=("I need the toilet.",))),
        (Proposal("strategy", "orient_time_place"), context()),
        (
            Proposal("say", "escalate_phone", text="Someone is coming to help.", terminal=True),
            context(person_state="in_bed", settled=True),
        ),
        (Proposal("say", "soft_greeting", text="Hello Jean."), context()),
        (Proposal("notify", "critical"), context()),
        (Proposal("light", "on"), context()),
    ],
)
def test_allowed_actions(proposal, facts):
    assert check(proposal, facts).allowed is True


def test_avoid_terms_only_extracts_literal_prohibitions():
    assert avoid_terms(AVOID) == ("hospital",)


def transition_for(strategy_id):
    strategy = next(item for item in DEFAULT_STRATEGIES if item.id == strategy_id)
    return Transition(Phase.ENGAGED, "session-1", "return_to_bed", 0, "test", strategy=strategy)


def test_vetoed_strategy_keeps_bookkeeping_but_suppresses_show_and_say(caplog):
    bus = make_bus()
    session = Session(config=AgentConfig())
    session.phase = Phase.ENGAGED
    session.on_person_state("standing", "other", NIGHT)
    session.record_utterance("I need the toilet.")
    caplog.set_level(logging.WARNING, logger="agent")

    _publish_transition(bus, transition_for("guided_return"), session, NIGHT)

    assert (
        len(
            [
                event
                for _, event in bus.read("session", "test", "c1")
                if isinstance(event, SessionState)
            ]
        )
        == 1
    )
    assert bus.read("show", "test", "c1") == []
    assert bus.read("say", "test", "c1") == []
    warnings = [
        json.loads(record.message) for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert warnings == [
        {
            "service": "agent",
            "message": "vetoed strategy",
            "event_type": "Show",
            "rule": "no_redirect_from_toilet_need",
            "clause": "TOIL-01",
            "action": "strategy:guided_return",
            "reason": "guided_return while a stated toilet need is unmet",
        }
    ]


def test_vetoed_say_does_not_advance_speech_clock():
    bus = make_bus()
    session = Session(config=AgentConfig())
    session.on_person_state("in_bed", "bed", NIGHT)
    transition = transition_for("soft_greeting")

    _maybe_publish_say(bus, transition, session, NIGHT + timedelta(seconds=1), PersonProfile())

    assert bus.read("say", "test", "c1") == []
    assert session.seconds_since_last_say(NIGHT + timedelta(seconds=2)) is None


def test_allowed_strategy_publishes_show_and_say():
    bus = make_bus()
    session = Session(config=AgentConfig())
    session.phase = Phase.ENGAGED
    session.on_person_state("standing", "other", NIGHT)

    _publish_transition(bus, transition_for("soft_greeting"), session, NIGHT)

    assert (
        len([event for _, event in bus.read("show", "test", "c1") if isinstance(event, Show)]) == 1
    )
    assert len([event for _, event in bus.read("say", "test", "c1") if isinstance(event, Say)]) == 1
