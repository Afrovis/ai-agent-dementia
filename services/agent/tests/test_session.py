"""Tests for `agent.session.Session`: the phase state machine.

All deterministic and time-injected: every test builds explicit `datetime`s
and advances them by hand, never sleeping and never touching a real clock,
per HANDOFF.md section 4.
"""

from dataclasses import replace as dc_replace
from datetime import datetime, timedelta

from agent.config import AgentConfig
from agent.rules import Phase
from agent.session import Session
from agent.strategies import DEFAULT_STRATEGIES, ESCALATE_PHONE_ID

NIGHT = datetime(2026, 1, 1, 23, 0)  # inside the default 21:00-07:00 window
DAY = datetime(2026, 1, 1, 12, 0)  # outside it


def make_session(*, strategies=None, **config_kwargs) -> Session:
    config = AgentConfig(**config_kwargs)
    ids = iter(f"sess-{i}" for i in range(1, 100))
    if strategies is not None:
        return Session(config=config, id_fn=lambda: next(ids), strategies=strategies)
    return Session(config=config, id_fn=lambda: next(ids))


def test_restroom_interpretation_during_floor_escalation_window_keeps_goal():
    session = make_session(observe_seconds=1, floor_limit_seconds=10)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    assert session.phase == Phase.ENGAGED
    session.on_person_state("on_floor", "other", NIGHT + timedelta(seconds=3))
    assert session.phase == Phase.ENGAGED
    assert session.on_interpretation("need_restroom", 0, NIGHT + timedelta(seconds=4)) is None
    assert session.goal == "return_to_bed"


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


def test_walking_starts_observing_inside_the_window():
    # perceive can confirm a bed exit straight into `walking`, skipping a
    # confirmed `standing` frame; that must still start a session.
    session = make_session()
    transition = session.on_person_state("walking", "other", NIGHT)
    assert transition is not None
    assert transition.phase == Phase.OBSERVING
    assert transition.reason == "person_walking"


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


def test_speaking_after_lying_down_clears_settled_even_in_idle():
    session = make_session()
    session.on_person_state("in_bed", "bed", NIGHT)
    assert session.settled is True

    session.on_utterance(NIGHT + timedelta(seconds=1))
    assert session.settled is False


def test_lying_down_after_speaking_is_settled_across_phase_reset():
    session = make_session()
    session.on_person_state("standing", "other", NIGHT)
    session.on_utterance(NIGHT + timedelta(seconds=1))
    session.on_person_state("in_bed", "bed", NIGHT + timedelta(seconds=2))

    assert session.phase == Phase.ENGAGED
    assert session.settled is True
    session._reset_timers()
    assert session.settled is True


def test_getting_up_clears_settled():
    session = make_session()
    session.on_person_state("in_bed", "bed", NIGHT)
    assert session.settled is True

    session.on_person_state("standing", "other", NIGHT + timedelta(seconds=1))
    assert session.settled is False


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
    session = make_session(
        observe_seconds=1.0,
        in_bed_stable_seconds=1.0,
        cooldown_seconds=300.0,
        floor_limit_seconds=0.0,
    )
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


def test_strategy_index_is_zero_before_any_strategy_is_selected():
    # Issue #14 fills in the strategy engine; before `ENGAGED` is ever
    # reached (here: still `OBSERVING`), no strategy has been selected yet
    # and the index stays at its initial 0.
    session = make_session()
    session.on_person_state("standing", "other", NIGHT)
    assert session.phase == Phase.OBSERVING
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
    assert transition.strategy is not None
    assert transition.strategy.id == "path_light"
    assert session.goal == "restroom"


def test_door_zone_switches_goal_to_restroom():
    session = make_session()
    t = NIGHT
    enter_engaged(session, t)
    t += timedelta(seconds=30)

    transition, _t = feed_zone(session, "walking", "door", t)
    assert transition is not None
    assert transition.goal == "restroom"
    assert transition.goal_change is not None
    assert transition.strategy is not None
    assert transition.strategy.id == "path_light"


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
    assert transition.strategy is not None
    assert transition.strategy.id == "guided_return"
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

    # Issue #14: the strategy engine's own dwell timer is independent of
    # the restroom timeout, and by 60s later it may well have advanced the
    # current strategy on its own (a `tick` with no accompanying
    # `PersonState` cannot observe "progress" toward the bed either way).
    # That is expected, unrelated behaviour; what this test actually
    # checks is that the *goal* has not timed out yet.
    still_within = session.tick(t + timedelta(seconds=60))
    if still_within is not None:
        assert still_within.goal_change is None
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
    session = make_session(floor_limit_seconds=0.0)
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


def test_propose_goal_cannot_undo_wait_for_caregiver_after_escalation():
    session = make_session(floor_limit_seconds=0.0)
    transition = session.on_person_state("on_floor", "other", NIGHT)
    assert transition is not None
    assert session.phase == Phase.ESCALATED

    proposed = session.propose_goal("return_to_bed", "llm_plan", NIGHT + timedelta(seconds=1))

    assert proposed is None
    assert session.goal == "wait_for_caregiver"


# --- issue #14: the strategy engine wired into Session -------------------


def small_strategies(*, dwell=100.0, cooldown=50.0, n=3):
    """A small, deterministic strategy catalogue -- same trick as
    `tests/test_strategies.py`'s `strategies_by_order`, independent of
    `DEFAULT_STRATEGIES`'s real dwell/cooldown values."""
    base = DEFAULT_STRATEGIES[0]
    return [
        dc_replace(
            base, id=f"s{i}", order=i, enabled=True, dwell_seconds=dwell, cooldown_seconds=cooldown
        )
        for i in range(1, n + 1)
    ]


def test_entering_engaged_selects_the_first_strategy():
    session = make_session(strategies=small_strategies(), observe_seconds=20.0)
    session.on_person_state("standing", "other", NIGHT)
    transition = session.tick(NIGHT + timedelta(seconds=21))
    assert transition is not None
    assert transition.phase == Phase.ENGAGED
    assert transition.strategy is not None
    assert transition.strategy.id == "s1"
    assert session.strategy_index == 0


def test_dwell_elapsed_with_no_progress_advances_the_strategy():
    session = make_session(strategies=small_strategies(dwell=30.0), observe_seconds=1.0)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    assert session.phase == Phase.ENGAGED

    # No progress (state stays "standing", zone stays "other") for longer
    # than the 30s dwell: the strategy must advance with no phase change.
    transition = session.on_person_state("standing", "other", NIGHT + timedelta(seconds=35))
    assert transition is not None
    assert transition.phase == Phase.ENGAGED
    assert transition.strategy is not None
    assert transition.strategy.id == "s2"
    assert session.strategy_index == 1


def test_progress_toward_bed_does_not_advance_the_strategy():
    session = make_session(strategies=small_strategies(dwell=30.0), observe_seconds=1.0)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    assert session.phase == Phase.ENGAGED

    # "Progress": the person is seen back at the bed zone before the dwell
    # would otherwise have elapsed. The strategy must not advance.
    transition = session.on_person_state("standing", "bed", NIGHT + timedelta(seconds=35))
    assert transition is None
    assert session.strategy_index == 0


def test_exhausting_every_strategy_escalates_and_publishes_a_notify():
    session = make_session(strategies=small_strategies(dwell=10.0, cooldown=1000.0, n=2))
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=21))
    assert session.phase == Phase.ENGAGED
    assert session.strategy_index == 0

    # s1's dwell elapses with no progress -> advance to s2.
    session.on_person_state("standing", "other", NIGHT + timedelta(seconds=32))
    assert session.phase == Phase.ENGAGED
    assert session.strategy_index == 1

    # s2's dwell elapses too, and nothing else is available -> escalate.
    transition = session.on_person_state("standing", "other", NIGHT + timedelta(seconds=43))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None
    assert transition.notify.level == "attention"
    assert session.phase == Phase.ESCALATED


def test_in_bed_holds_the_ladder_until_stability_moves_to_cooldown():
    """A short final strategy dwell must not outrun the longer settling
    window and manufacture a strategies-exhausted caregiver alert while
    the person remains in bed."""
    session = make_session(
        strategies=small_strategies(dwell=10.0, cooldown=1000.0, n=1),
        observe_seconds=1.0,
        in_bed_stable_seconds=120.0,
    )
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    assert session.phase == Phase.ENGAGED

    session.on_person_state("in_bed", "bed", NIGHT + timedelta(seconds=5))
    # Far beyond the only strategy's dwell, a time-only tick still holds
    # because no newer non-in-bed reading has cleared the stability marker.
    assert session.tick(NIGHT + timedelta(seconds=124)) is None
    assert session.phase == Phase.ENGAGED

    transition = session.on_person_state("in_bed", "bed", NIGHT + timedelta(seconds=125))
    assert transition is not None
    assert transition.phase == Phase.COOLDOWN
    assert transition.reason == "in_bed_stable"
    assert transition.notify is None


def test_strategies_exhausted_deescalates_after_away_then_return_to_bed():
    session = make_session(
        strategies=small_strategies(dwell=1.0, cooldown=1000.0, n=1),
        observe_seconds=1.0,
        in_bed_stable_seconds=120.0,
    )
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    escalation = session.on_person_state("standing", "other", NIGHT + timedelta(seconds=4))
    assert escalation is not None
    assert escalation.reason == "strategies_exhausted"
    assert session.phase == Phase.ESCALATED

    # The away observation must happen while already escalated; only then
    # does sitting on the bed prove the terminal presentation can wind down.
    assert session.on_person_state("walking", "other", NIGHT + timedelta(seconds=5)) is None
    returned = session.on_person_state("sitting_up", "bed", NIGHT + timedelta(seconds=6))
    assert returned is not None
    assert returned.phase == Phase.COOLDOWN
    assert returned.reason == "returned_to_bed_after_escalation"
    assert returned.notify is None


def test_rule5_escalation_does_not_use_the_quick_return_to_bed_path():
    session = make_session(floor_limit_seconds=0.0, in_bed_stable_seconds=120.0)
    session.on_person_state("on_floor", "other", NIGHT)
    assert session.phase == Phase.ESCALATED
    session.on_person_state("walking", "other", NIGHT + timedelta(seconds=1))

    assert session.on_person_state("sitting_up", "bed", NIGHT + timedelta(seconds=2)) is None
    assert session.phase == Phase.ESCALATED


def test_distress_escalation_does_not_use_the_quick_return_to_bed_path():
    session = make_session(observe_seconds=1.0, in_bed_stable_seconds=120.0)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    session.on_interpretation("unclear", 2, NIGHT + timedelta(seconds=3))
    escalation = session.on_interpretation("unclear", 3, NIGHT + timedelta(seconds=4))
    assert escalation is not None
    assert escalation.reason == "distress_detected_twice"

    session.on_person_state("walking", "other", NIGHT + timedelta(seconds=5))
    assert session.on_person_state("sitting_up", "bed", NIGHT + timedelta(seconds=6)) is None
    assert session.phase == Phase.ESCALATED


def test_escalate_phone_is_selected_on_escalation_and_stays_selected():
    # Uses the real `DEFAULT_STRATEGIES` catalogue (the default when no
    # `strategies=` override is given): `escalate_phone` is only in the
    # real catalogue, not in the small synthetic one used above.
    session = make_session(floor_limit_seconds=0.0)
    session.on_person_state("standing", "other", NIGHT)
    transition = session.on_person_state("on_floor", "other", NIGHT + timedelta(seconds=1))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.strategy is not None
    assert transition.strategy.id == ESCALATE_PHONE_ID

    # Time passing, and more "on_floor" readings, must not move it off
    # `escalate_phone`: `_run_strategy_engine` is a no-op outside `ENGAGED`
    # (see its docstring), so nothing advances it on a timer, and an
    # unchanged "on_floor" reading produces no further `Transition` at all
    # (`_track_in_bed_stability`/`_update_goal_from_zone` both no-op here
    # too).
    assert session.on_person_state("on_floor", "other", NIGHT + timedelta(seconds=500)) is None
    assert session.phase == Phase.ESCALATED


def test_a_disabled_strategy_is_skipped_and_the_next_one_is_selected():
    strategies = small_strategies()
    strategies[0] = dc_replace(strategies[0], enabled=False)
    session = make_session(strategies=strategies, observe_seconds=1.0)
    session.on_person_state("standing", "other", NIGHT)
    transition = session.tick(NIGHT + timedelta(seconds=2))
    assert transition is not None
    assert transition.strategy.id == "s2"


def test_no_strategy_available_at_all_escalates_immediately_instead_of_engaging():
    strategies = [dc_replace(s, enabled=False) for s in small_strategies()]
    session = make_session(strategies=strategies, observe_seconds=1.0)
    session.on_person_state("standing", "other", NIGHT)
    transition = session.tick(NIGHT + timedelta(seconds=2))
    assert transition is not None
    assert transition.phase == Phase.ESCALATED
    assert transition.notify is not None


def test_session_state_strategy_index_tracks_the_live_value():
    session = make_session(strategies=small_strategies(dwell=10.0), observe_seconds=1.0)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT + timedelta(seconds=2))
    assert session.strategy_index == 0

    transition = session.on_person_state("standing", "other", NIGHT + timedelta(seconds=13))
    assert transition is not None
    assert transition.strategy_index == 1
    assert session.strategy_index == 1


# --- Regression tests for the strategy-engine review findings ------------


def test_lingering_at_the_bedside_does_not_pin_the_ladder():
    """Review finding: `zone == "bed"` counts as progress on every
    reading, so someone standing at the bedside without getting in used to
    hold `ambient_orient` -- which is silent -- for the whole night, with
    no advance and no escalation (rule 5 sees neither `on_floor` nor
    `absent`)."""
    session = make_session(
        observe_seconds=1.0,
        in_bed_stable_seconds=99999.0,
        floor_limit_seconds=99999.0,
        absent_limit_seconds=99999.0,
    )
    session.on_person_state("standing", "other", NIGHT)
    session.on_person_state("standing", "other", NIGHT + timedelta(seconds=2))
    assert session.phase == Phase.ENGAGED

    seen = []
    for i in range(3, 1200):
        now = NIGHT + timedelta(seconds=i)
        transition = session.on_person_state("standing", "bed", now)
        if transition is not None and transition.strategy is not None:
            seen.append(transition.strategy.id)
        if session.phase == Phase.ESCALATED:
            break

    assert seen, "the ladder never advanced while the person lingered at the bed"
    assert session.phase == Phase.ESCALATED


def test_the_real_catalogue_escalates_rather_than_speaking_the_escalation_line():
    """Review finding: with the real catalogue the ladder used to reach
    `escalate_phone` while still `ENGAGED`, telling the person help was
    coming with no `Notify` sent and no caregiver told, then freezing there
    on its infinite dwell."""
    session = make_session(
        observe_seconds=1.0,
        in_bed_stable_seconds=99999.0,
        floor_limit_seconds=99999.0,
        absent_limit_seconds=99999.0,
    )
    session.on_person_state("standing", "other", NIGHT)
    session.on_person_state("standing", "other", NIGHT + timedelta(seconds=2))

    escalation = None
    for i in range(3, 1200):
        transition = session.tick(NIGHT + timedelta(seconds=i))
        if transition is None:
            continue
        # While ENGAGED the terminal strategy must never be selected.
        if transition.phase == Phase.ENGAGED and transition.strategy is not None:
            assert transition.strategy.id != ESCALATE_PHONE_ID
        if transition.phase == Phase.ESCALATED:
            escalation = transition
            break

    assert escalation is not None, "the ladder never escalated"
    assert escalation.reason == "strategies_exhausted"
    assert escalation.notify is not None, "escalated without telling the caregiver"
    assert escalation.strategy is not None
    assert escalation.strategy.id == ESCALATE_PHONE_ID
    assert session.goal == "wait_for_caregiver"


def test_compliance_grace_holds_ladder_until_expiry():
    session = make_session(observe_seconds=0, compliance_grace_seconds=120)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT)
    assert session.phase == Phase.ENGAGED
    session.on_interpretation("wants_bed", 0, NIGHT + timedelta(seconds=1))
    assert session.tick(NIGHT + timedelta(seconds=90)) is None
    assert session.strategy_index == 0
    advanced = session.tick(NIGHT + timedelta(seconds=121))
    assert advanced is not None and advanced.strategy.id == "soft_greeting"


def test_new_need_breaks_compliance_grace():
    session = make_session(observe_seconds=0)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT)
    session.on_interpretation("wants_bed", 0, NIGHT + timedelta(seconds=1))
    session.on_interpretation("pain", 0, NIGHT + timedelta(seconds=2))
    assert not session.compliance_hold(NIGHT + timedelta(seconds=3))


def test_self_echo_matches_short_fragment_and_overlap_only_within_twenty_seconds():
    session = make_session()
    session.record_say(NIGHT, "guided_return", "Let's go back to bed now, Jean.")
    assert session.is_self_echo("Go.", NIGHT + timedelta(seconds=2))
    assert session.is_self_echo("Go back to bed now", NIGHT + timedelta(seconds=2))
    assert not session.is_self_echo("I need the restroom", NIGHT + timedelta(seconds=2))
    assert not session.is_self_echo("Go.", NIGHT + timedelta(seconds=21))


def test_ladder_skips_a_strategy_already_spoken_as_a_direct_reply():
    session = make_session(observe_seconds=0)
    session.on_person_state("standing", "other", NIGHT)
    session.tick(NIGHT)
    soft = session.tick(NIGHT + timedelta(seconds=30))
    assert soft is not None and soft.strategy.id == "soft_greeting"
    session.record_say(NIGHT + timedelta(seconds=31), "orient_time_place", "You are home.")
    next_rung = session.tick(NIGHT + timedelta(seconds=50))
    assert next_rung is not None and next_rung.strategy.id == "validate_and_redirect"


def test_persistent_distress_followup_is_once_after_a_minute():
    session = make_session(floor_limit_seconds=0)
    first = session.on_person_state("on_floor", "other", NIGHT)
    assert first.notify.level == "critical"
    assert session.on_interpretation("unclear", 2, NIGHT + timedelta(seconds=10)) is None
    assert session.on_interpretation("fine", 0, NIGHT + timedelta(seconds=20)) is None
    assert session.on_interpretation("unclear", 2, NIGHT + timedelta(seconds=30)) is None
    followup = session.on_interpretation("unclear", 2, NIGHT + timedelta(seconds=61))
    assert followup.notify.level == "critical"
    assert followup.notify.title == "Still in distress"
    assert "3 times since the first alert" in followup.notify.body
    assert session.on_interpretation("unclear", 3, NIGHT + timedelta(seconds=90)) is None


def test_short_floor_reading_does_not_escalate():
    session = make_session()
    assert session.on_person_state("on_floor", "other", NIGHT) is None
    assert session.tick(NIGHT + timedelta(seconds=5)) is None
    assert session.on_person_state("sitting_up", "other", NIGHT + timedelta(seconds=5)) is not None
    assert session.phase == Phase.OBSERVING
    assert session._brief_floor_notice.level == "info"
    assert session._brief_floor_notice.repeat_until_ack is False
    assert session._brief_floor_notice.body == (
        "The camera saw them on the floor for 5 s; they are up again."
    )


def test_floor_limit_is_checked_before_a_late_clear_reading():
    session = make_session()
    session.on_person_state("on_floor", "other", NIGHT)
    transition = session.on_person_state("sitting_up", "other", NIGHT + timedelta(seconds=11))
    assert transition.phase == Phase.ESCALATED
    assert transition.notify.title == "Possible fall"
    assert session._brief_floor_notice is None
