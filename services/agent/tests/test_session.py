"""Tests for `agent.session.Session`: the phase state machine.

All deterministic and time-injected: every test builds explicit `datetime`s
and advances them by hand, never sleeping and never touching a real clock,
per HANDOFF.md section 4.
"""

from datetime import datetime, timedelta

from agent.config import AgentConfig
from agent.rules import Phase
from agent.session import Session

NIGHT = datetime(2026, 1, 1, 23, 0)  # inside the default 21:00-07:00 window
DAY = datetime(2026, 1, 1, 12, 0)  # outside it


def make_session(**config_kwargs) -> Session:
    config = AgentConfig(**config_kwargs)
    ids = iter(f"sess-{i}" for i in range(1, 100))
    return Session(config=config, id_fn=lambda: next(ids))


def test_no_session_starts_outside_the_night_window():
    session = make_session()
    assert session.on_person_state("standing", DAY) is None
    assert session.phase == Phase.IDLE


def test_sitting_up_starts_observing_inside_the_window():
    session = make_session()
    transition = session.on_person_state("sitting_up", NIGHT)
    assert transition is not None
    assert transition.phase == Phase.OBSERVING
    assert transition.session_id is not None
    assert session.phase == Phase.OBSERVING
    assert session.session_id == transition.session_id


def test_observing_times_out_to_engaged_after_20s():
    session = make_session(observe_seconds=20.0)
    session.on_person_state("standing", NIGHT)
    assert session.phase == Phase.OBSERVING

    almost = NIGHT + timedelta(seconds=19)
    assert session.on_person_state("standing", almost) is None
    assert session.phase == Phase.OBSERVING

    later = NIGHT + timedelta(seconds=21)
    transition = session.on_person_state("standing", later)
    assert transition is not None
    assert transition.phase == Phase.ENGAGED


def test_observing_times_out_via_tick_with_no_new_person_state():
    session = make_session(observe_seconds=20.0)
    session.on_person_state("standing", NIGHT)
    transition = session.tick(NIGHT + timedelta(seconds=25))
    assert transition is not None
    assert transition.phase == Phase.ENGAGED


def test_observing_returns_to_idle_on_in_bed():
    session = make_session()
    session.on_person_state("sitting_up", NIGHT)
    assert session.phase == Phase.OBSERVING

    transition = session.on_person_state("in_bed", NIGHT + timedelta(seconds=5))
    assert transition is not None
    assert transition.phase == Phase.IDLE
    assert transition.session_id is None
    assert session.phase == Phase.IDLE
    assert session.session_id is None


def test_utterance_short_circuits_observing_to_engaged():
    session = make_session(observe_seconds=20.0)
    session.on_person_state("sitting_up", NIGHT)
    assert session.phase == Phase.OBSERVING

    transition = session.on_utterance(NIGHT + timedelta(seconds=2))
    assert transition is not None
    assert transition.phase == Phase.ENGAGED


def test_utterance_has_no_effect_in_idle():
    session = make_session()
    assert session.on_utterance(NIGHT) is None
    assert session.phase == Phase.IDLE


def test_rule5_on_floor_fires_immediately_by_default():
    session = make_session(floor_limit_seconds=0.0)
    session.on_person_state("sitting_up", NIGHT)
    assert session.phase == Phase.OBSERVING

    transition = session.on_person_state("on_floor", NIGHT + timedelta(seconds=1))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None
    assert transition.notify.level == "critical"


def test_rule5_on_floor_respects_a_configured_grace_period():
    # observe_seconds set well past the window under test, so OBSERVING's
    # own 20s timeout cannot fire first and mask rule 5's timing.
    session = make_session(floor_limit_seconds=10.0, observe_seconds=999.0)
    session.on_person_state("sitting_up", NIGHT)

    # The floor timer starts on the first `on_floor` classification, not on
    # session start -- 5s after that is still within a 10s grace period.
    too_soon = session.on_person_state("on_floor", NIGHT + timedelta(seconds=5))
    assert too_soon is None
    assert session.phase == Phase.OBSERVING

    transition = session.on_person_state("on_floor", NIGHT + timedelta(seconds=16))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED


def test_rule5_absent_does_not_fire_before_its_limit():
    session = make_session(absent_limit_seconds=600.0, observe_seconds=999.0)
    session.on_person_state("sitting_up", NIGHT)

    still_within_limit = session.on_person_state("absent", NIGHT + timedelta(seconds=300))
    assert still_within_limit is None
    assert session.phase == Phase.OBSERVING


def test_rule5_absent_fires_after_its_limit():
    session = make_session(absent_limit_seconds=600.0, observe_seconds=999.0)
    session.on_person_state("sitting_up", NIGHT)

    # The absent timer starts on the first `absent` classification.
    session.on_person_state("absent", NIGHT + timedelta(seconds=1))
    transition = session.on_person_state("absent", NIGHT + timedelta(seconds=602))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None
    assert transition.notify.level == "critical"


def test_rule5_absent_timer_resets_if_the_person_reappears():
    session = make_session(absent_limit_seconds=10.0)
    session.on_person_state("sitting_up", NIGHT)

    session.on_person_state("absent", NIGHT + timedelta(seconds=5))
    session.on_person_state("standing", NIGHT + timedelta(seconds=8))  # reappears
    transition = session.on_person_state("absent", NIGHT + timedelta(seconds=15))
    # Only 7s of unbroken absence since the reappearance -- must not fire yet.
    assert transition is None


def test_rule5_fires_from_idle_with_no_prior_session():
    # A fall straight from `in_bed` to `on_floor` -- no intervening
    # `sitting_up`/`standing` reading, so no session has started yet.
    # HANDOFF.md rule 5 has no "only if a session is already live"
    # qualifier and must still escalate.
    session = make_session(floor_limit_seconds=0.0)
    assert session.phase == Phase.IDLE

    transition = session.on_person_state("on_floor", NIGHT)
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.session_id is not None
    assert transition.notify is not None
    assert transition.notify.level == "critical"
    assert session.phase == Phase.ESCALATED


def test_rule5_fires_from_idle_outside_the_night_window():
    # Rule 5 is not gated by the night window: a person on the floor at
    # noon still needs help, and `perceive` watches around the clock.
    session = make_session(floor_limit_seconds=0.0)
    transition = session.on_person_state("on_floor", DAY)
    assert transition is not None
    assert transition.phase == Phase.ESCALATED


def test_rule5_absent_fires_from_idle_after_its_limit():
    session = make_session(absent_limit_seconds=10.0)
    session.on_person_state("absent", NIGHT)
    transition = session.on_person_state("absent", NIGHT + timedelta(seconds=11))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None
    assert transition.notify.level == "critical"


def test_no_new_nudging_session_during_cooldown_but_rule5_still_escalates():
    session = make_session(observe_seconds=1.0, in_bed_stable_seconds=1.0, cooldown_seconds=300.0)
    session.on_person_state("standing", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    assert session.phase == Phase.ENGAGED

    session.on_person_state("in_bed", NIGHT + timedelta(seconds=3))
    transition = session.on_person_state("in_bed", NIGHT + timedelta(seconds=5))
    assert transition is not None
    assert transition.phase == Phase.COOLDOWN
    cooldown_session_id = transition.session_id

    # Getting up again mid-cooldown must not start a new *nudging* session.
    transition = session.on_person_state("standing", NIGHT + timedelta(seconds=10))
    assert transition is None
    assert session.phase == Phase.COOLDOWN

    # Rule 5 must still fire during cooldown: a fall does not wait its turn.
    transition = session.on_person_state("on_floor", NIGHT + timedelta(seconds=11))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.session_id == cooldown_session_id
    assert transition.notify is not None
    assert transition.notify.level == "critical"


def test_full_cycle_idle_observing_engaged_cooldown_idle():
    session = make_session(
        observe_seconds=20.0, in_bed_stable_seconds=120.0, cooldown_seconds=300.0
    )
    assert session.phase == Phase.IDLE

    t = NIGHT
    transition = session.on_person_state("standing", t)
    assert transition.phase == Phase.OBSERVING
    session_id = transition.session_id

    t += timedelta(seconds=25)
    transition = session.tick(t)
    assert transition.phase == Phase.ENGAGED
    assert transition.session_id == session_id

    t += timedelta(seconds=5)
    session.on_person_state("in_bed", t)
    t += timedelta(seconds=121)
    transition = session.on_person_state("in_bed", t)
    assert transition.phase == Phase.COOLDOWN
    assert transition.session_id == session_id

    t += timedelta(seconds=301)
    transition = session.tick(t)
    assert transition.phase == Phase.IDLE
    assert transition.session_id is None
    assert session.phase == Phase.IDLE
    assert session.goal == "return_to_bed"
    assert session.strategy_index == 0


def test_goal_and_strategy_index_stay_fixed_seams_for_later_issues():
    session = make_session()
    session.on_person_state("standing", NIGHT)
    assert session.goal == "return_to_bed"
    assert session.strategy_index == 0
