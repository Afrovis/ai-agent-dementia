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
    assert session.on_person_state("standing", "other", DAY) is None
    assert session.phase == Phase.IDLE


def test_sitting_up_starts_observing_inside_the_window():
    session = make_session()
    transition = session.on_person_state("sitting_up", "other", NIGHT)
    assert transition is not None
    assert transition.phase == Phase.OBSERVING
    assert transition.session_id is not None
    assert session.phase == Phase.OBSERVING
    assert session.session_id == transition.session_id


def test_observing_times_out_to_engaged_after_20s():
    session = make_session(observe_seconds=20.0)
    session.on_person_state("standing", "other", NIGHT)
    assert session.phase == Phase.OBSERVING

    almost = NIGHT + timedelta(seconds=19)
    assert session.on_person_state("standing", "other", almost) is None
    assert session.phase == Phase.OBSERVING

    later = NIGHT + timedelta(seconds=21)
    transition = session.on_person_state("standing", "other", later)
    assert transition is not None
    assert transition.phase == Phase.ENGAGED


def test_observing_times_out_via_tick_with_no_new_person_state():
    session = make_session(observe_seconds=20.0)
    session.on_person_state("standing", "other", NIGHT)
    transition = session.tick(NIGHT + timedelta(seconds=25))
    assert transition is not None
    assert transition.phase == Phase.ENGAGED


def test_observing_returns_to_idle_on_in_bed():
    session = make_session()
    session.on_person_state("sitting_up", "other", NIGHT)
    assert session.phase == Phase.OBSERVING

    transition = session.on_person_state("in_bed", "other", NIGHT + timedelta(seconds=5))
    assert transition is not None
    assert transition.phase == Phase.IDLE
    assert transition.session_id is None
    assert session.phase == Phase.IDLE
    assert session.session_id is None


def test_utterance_short_circuits_observing_to_engaged():
    session = make_session(observe_seconds=20.0)
    session.on_person_state("sitting_up", "other", NIGHT)
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
    session.on_person_state("sitting_up", "other", NIGHT)
    assert session.phase == Phase.OBSERVING

    transition = session.on_person_state("on_floor", "other", NIGHT + timedelta(seconds=1))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None
    assert transition.notify.level == "critical"


def test_rule5_on_floor_respects_a_configured_grace_period():
    # observe_seconds set well past the window under test, so OBSERVING's
    # own 20s timeout cannot fire first and mask rule 5's timing.
    session = make_session(floor_limit_seconds=10.0, observe_seconds=999.0)
    session.on_person_state("sitting_up", "other", NIGHT)

    # The floor timer starts on the first `on_floor` classification, not on
    # session start -- 5s after that is still within a 10s grace period.
    too_soon = session.on_person_state("on_floor", "other", NIGHT + timedelta(seconds=5))
    assert too_soon is None
    assert session.phase == Phase.OBSERVING

    transition = session.on_person_state("on_floor", "other", NIGHT + timedelta(seconds=16))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED


def test_rule5_absent_does_not_fire_before_its_limit():
    session = make_session(absent_limit_seconds=600.0, observe_seconds=999.0)
    session.on_person_state("sitting_up", "other", NIGHT)

    still_within_limit = session.on_person_state("absent", "other", NIGHT + timedelta(seconds=300))
    assert still_within_limit is None
    assert session.phase == Phase.OBSERVING


def test_rule5_absent_fires_after_its_limit():
    session = make_session(absent_limit_seconds=600.0, observe_seconds=999.0)
    session.on_person_state("sitting_up", "other", NIGHT)

    # The absent timer starts on the first `absent` classification.
    session.on_person_state("absent", "other", NIGHT + timedelta(seconds=1))
    transition = session.on_person_state("absent", "other", NIGHT + timedelta(seconds=602))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None
    assert transition.notify.level == "critical"


def test_rule5_absent_timer_resets_if_the_person_reappears():
    session = make_session(absent_limit_seconds=10.0)
    session.on_person_state("sitting_up", "other", NIGHT)

    session.on_person_state("absent", "other", NIGHT + timedelta(seconds=5))
    session.on_person_state("standing", "other", NIGHT + timedelta(seconds=8))  # reappears
    transition = session.on_person_state("absent", "other", NIGHT + timedelta(seconds=15))
    # Only 7s of unbroken absence since the reappearance -- must not fire yet.
    assert transition is None


def test_rule5_fires_from_idle_with_no_prior_session():
    # A fall straight from `in_bed` to `on_floor` -- no intervening
    # `sitting_up`/`standing` reading, so no session has started yet.
    # HANDOFF.md rule 5 has no "only if a session is already live"
    # qualifier and must still escalate.
    session = make_session(floor_limit_seconds=0.0)
    assert session.phase == Phase.IDLE

    transition = session.on_person_state("on_floor", "other", NIGHT)
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
    transition = session.on_person_state("on_floor", "other", DAY)
    assert transition is not None
    assert transition.phase == Phase.ESCALATED


def test_rule5_absent_fires_from_idle_after_its_limit():
    session = make_session(absent_limit_seconds=10.0)
    session.on_person_state("absent", "other", NIGHT)
    transition = session.on_person_state("absent", "other", NIGHT + timedelta(seconds=11))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None
    assert transition.notify.level == "critical"


def test_no_new_nudging_session_during_cooldown_but_rule5_still_escalates():
    session = make_session(observe_seconds=1.0, in_bed_stable_seconds=1.0, cooldown_seconds=300.0)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    assert session.phase == Phase.ENGAGED

    session.on_person_state("in_bed", "other", NIGHT + timedelta(seconds=3))
    transition = session.on_person_state("in_bed", "other", NIGHT + timedelta(seconds=5))
    assert transition is not None
    assert transition.phase == Phase.COOLDOWN
    cooldown_session_id = transition.session_id

    # Getting up again mid-cooldown must not start a new *nudging* session.
    transition = session.on_person_state("standing", "other", NIGHT + timedelta(seconds=10))
    assert transition is None
    assert session.phase == Phase.COOLDOWN

    # Rule 5 must still fire during cooldown: a fall does not wait its turn.
    transition = session.on_person_state("on_floor", "other", NIGHT + timedelta(seconds=11))
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
    transition = session.on_person_state("standing", "other", t)
    assert transition.phase == Phase.OBSERVING
    session_id = transition.session_id

    t += timedelta(seconds=25)
    transition = session.tick(t)
    assert transition.phase == Phase.ENGAGED
    assert transition.session_id == session_id

    t += timedelta(seconds=5)
    session.on_person_state("in_bed", "other", t)
    t += timedelta(seconds=121)
    transition = session.on_person_state("in_bed", "other", t)
    assert transition.phase == Phase.COOLDOWN
    assert transition.session_id == session_id

    t += timedelta(seconds=301)
    transition = session.tick(t)
    assert transition.phase == Phase.IDLE
    assert transition.session_id is None
    assert session.phase == Phase.IDLE
    assert session.goal == "return_to_bed"
    assert session.strategy_index == 0


def test_strategy_index_stays_a_fixed_seam_for_a_later_issue():
    session = make_session()
    session.on_person_state("standing", "other", NIGHT)
    assert session.strategy_index == 0


def enter_engaged(session: Session, t: datetime) -> None:
    """Helper: drive `session` from `IDLE` into `ENGAGED` via the ordinary
    `standing` + observe-timeout path, used by the goal tests below."""
    session.on_person_state("standing", "other", t)
    session.tick(t + timedelta(seconds=25))
    assert session.phase == Phase.ENGAGED


def feed_zone(session: Session, state: str, zone: str, t: datetime):
    """Helper: send `zone` on `state` readings, one per second, for
    `session.config.zone_confirm_readings` consecutive `PersonState`s --
    the number of agreeing readings `_confirmed_zone` requires before a
    zone may drive a goal change. Returns `(last_transition, next_t)`."""
    transition = None
    for _ in range(session.config.zone_confirm_readings):
        transition = session.on_person_state(state, zone, t)
        t += timedelta(seconds=1)
    return transition, t


def test_bathroom_path_zone_switches_goal_to_restroom():
    session = make_session()
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=30)

    transition, t = feed_zone(session, "walking", "bathroom_path", t)
    assert transition is not None
    assert transition.phase == Phase.ENGAGED
    assert transition.goal == "restroom"
    assert transition.goal_change is not None
    assert transition.goal_change.from_goal == "return_to_bed"
    assert transition.goal_change.to_goal == "restroom"
    assert session.goal == "restroom"


def test_a_single_bathroom_path_reading_does_not_switch_the_goal():
    # Below `config.zone_confirm_readings`' default of 3: the goal must
    # not move yet.
    session = make_session()
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=30)

    transition = session.on_person_state("walking", "bathroom_path", t)
    assert transition is None
    assert session.goal == "return_to_bed"


def test_flapping_zone_produces_at_most_one_goal_changed():
    # A zone reading that never settles -- alternating bathroom_path/bed
    # every single reading, as could plausibly happen right at the
    # boundary between the two zones -- must never accumulate the
    # required run of agreeing readings, so the goal must never move and
    # no `GoalChanged` may be reported.
    session = make_session()
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=30)

    goal_changes = 0
    for i in range(20):
        zone = "bathroom_path" if i % 2 == 0 else "bed"
        transition = session.on_person_state("walking", zone, t)
        t += timedelta(seconds=1)
        if transition is not None and transition.goal_change is not None:
            goal_changes += 1

    assert goal_changes <= 1
    assert session.goal == "return_to_bed"


def test_restroom_returns_to_parent_goal_when_person_comes_back():
    session = make_session()
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=30)
    _transition, t = feed_zone(session, "walking", "bathroom_path", t)
    assert session.goal == "restroom"

    t += timedelta(seconds=60)
    transition, t = feed_zone(session, "walking", "bed", t)
    assert transition is not None
    assert transition.goal == "return_to_bed"
    assert transition.goal_change is not None
    assert transition.goal_change.from_goal == "restroom"
    assert transition.goal_change.to_goal == "return_to_bed"
    assert session.goal == "return_to_bed"


def test_restroom_returns_to_parent_goal_on_in_bed_state_too():
    # `state == "in_bed"` needs no repeated readings to trigger the
    # return: `perceive` already applies its own hysteresis to `state`
    # before publishing it, unlike the bare, unconfirmed `zone` field.
    session = make_session()
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=30)
    _transition, t = feed_zone(session, "walking", "bathroom_path", t)

    t += timedelta(seconds=60)
    transition = session.on_person_state("in_bed", "other", t)
    assert transition is not None
    assert transition.goal_change is not None
    assert transition.goal_change.to_goal == "return_to_bed"


def test_restroom_times_out_back_to_the_parent_goal():
    session = make_session(restroom_timeout_seconds=120.0)
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=30)
    _transition, t = feed_zone(session, "walking", "bathroom_path", t)
    assert session.goal == "restroom"

    still_within = session.tick(t + timedelta(seconds=60))
    assert still_within is None
    assert session.goal == "restroom"

    transition = session.tick(t + timedelta(seconds=121))
    assert transition is not None
    assert transition.goal_change is not None
    assert transition.goal_change.reason == "restroom_timeout"
    assert transition.goal_change.to_goal == "return_to_bed"
    assert session.goal == "return_to_bed"


def test_goal_resets_to_root_on_idle_with_no_spurious_goal_changed():
    session = make_session()
    # This session never leaves the root goal, so returning to IDLE must
    # not report a goal change: from_goal would equal to_goal, which is
    # exactly the "spurious GoalChanged" this must avoid.
    session.on_person_state("sitting_up", "other", NIGHT)
    transition = session.on_person_state("in_bed", "other", NIGHT + timedelta(seconds=5))
    assert transition is not None
    assert transition.phase == Phase.IDLE
    assert transition.goal_change is None
    assert session.goal == "return_to_bed"


def test_goal_reset_on_idle_reports_a_real_goal_change_when_one_happened():
    # Escalate (sets goal to `wait_for_caregiver`, per HANDOFF.md rule 5),
    # then let the person settle back `in_bed` long enough for the
    # ordinary ENGAGED/ESCALATED -> COOLDOWN -> IDLE path (issue #12) to
    # run its course. The final IDLE transition must report the goal
    # actually resetting away from `wait_for_caregiver`, unlike the
    # never-left-root case above.
    session = make_session(floor_limit_seconds=0.0, in_bed_stable_seconds=1.0, cooldown_seconds=1.0)
    t = NIGHT
    session.on_person_state("on_floor", "other", t)
    assert session.phase == Phase.ESCALATED
    assert session.goal == "wait_for_caregiver"

    t += timedelta(seconds=1)
    session.on_person_state("in_bed", "other", t)
    t += timedelta(seconds=2)
    cooldown_transition = session.on_person_state("in_bed", "other", t)
    assert cooldown_transition.phase == Phase.COOLDOWN

    t += timedelta(seconds=2)
    idle_transition = session.tick(t)
    assert idle_transition is not None
    assert idle_transition.phase == Phase.IDLE
    assert idle_transition.goal_change is not None
    assert idle_transition.goal_change.from_goal == "wait_for_caregiver"
    assert idle_transition.goal_change.to_goal == "return_to_bed"
    assert session.goal == "return_to_bed"


def test_wait_for_caregiver_set_on_escalation():
    session = make_session(floor_limit_seconds=0.0)
    transition = session.on_person_state("on_floor", "other", NIGHT)
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.goal == "wait_for_caregiver"
    assert transition.goal_change is not None
    assert transition.goal_change.from_goal == "return_to_bed"
    assert transition.goal_change.to_goal == "wait_for_caregiver"
    assert session.goal == "wait_for_caregiver"


def test_wait_for_caregiver_set_on_escalation_from_a_non_root_goal():
    session = make_session()
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=1)
    _transition, t = feed_zone(session, "walking", "bathroom_path", t)
    assert session.goal == "restroom"

    t += timedelta(seconds=1)
    session.on_person_state("on_floor", "other", t)
    assert session.phase == Phase.ESCALATED
    assert session.goal == "wait_for_caregiver"


def test_propose_goal_accepts_a_legal_change():
    session = make_session()
    enter_engaged(session, NIGHT)

    transition = session.propose_goal("drink_water", "llm_plan", NIGHT + timedelta(seconds=1))
    assert transition is not None
    assert transition.goal == "drink_water"
    assert transition.goal_change is not None
    assert transition.goal_change.from_goal == "return_to_bed"
    assert transition.goal_change.to_goal == "drink_water"
    assert transition.goal_change.reason == "llm_plan"
    assert session.goal == "drink_water"


def test_propose_goal_rejects_an_illegal_change():
    session = make_session()
    enter_engaged(session, NIGHT)
    session.propose_goal("restroom", "llm_plan", NIGHT + timedelta(seconds=1))
    assert session.goal == "restroom"

    # restroom -> drink_water is not a legal direct switch (must return to
    # the parent goal first): rejected quietly, not raised.
    transition = session.propose_goal("drink_water", "llm_plan", NIGHT + timedelta(seconds=2))
    assert transition is None
    assert session.goal == "restroom"
