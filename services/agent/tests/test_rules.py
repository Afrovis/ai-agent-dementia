"""Tests for `agent.rules`: every accept/reject case in `ALLOWED_TRANSITIONS`,
including its exact boundaries, the strategy-disabled seam, (issue #13)
every accept/reject case in `agent.goals.ALLOWED_GOAL_CHANGES`, and (issue
#14) `validate_strategy` and `validate_say`."""

import pytest

from agent.goals import ALLOWED_GOAL_CHANGES, GOALS
from agent.profile import PersonProfile
from agent.rules import (
    ALLOWED_TRANSITIONS,
    Phase,
    validate,
    validate_composition,
    validate_goal,
    validate_say,
    validate_strategy,
)

ALL_PHASES = list(Phase)
ALL_GOALS = sorted(GOALS)


def test_every_allowed_transition_is_accepted():
    for current, targets in ALLOWED_TRANSITIONS.items():
        for target in targets:
            result = validate(current, target)
            assert result.accepted is True
            assert result.reason is None


@pytest.mark.parametrize("current", ALL_PHASES)
@pytest.mark.parametrize("proposed", ALL_PHASES)
def test_every_pair_matches_the_table_exactly(current, proposed):
    result = validate(current, proposed)
    expected = proposed in ALLOWED_TRANSITIONS.get(current, frozenset())
    assert result.accepted is expected
    if not expected:
        assert result.reason is not None


def test_escalated_to_cooldown_is_allowed():
    assert validate(Phase.ESCALATED, Phase.COOLDOWN).accepted is True


def test_escalated_to_engaged_is_rejected():
    result = validate(Phase.ESCALATED, Phase.ENGAGED)
    assert result.accepted is False
    assert "ESCALATED -> ENGAGED" in result.reason


def test_cooldown_exits_to_idle_or_escalated_via_rule5():
    assert validate(Phase.COOLDOWN, Phase.IDLE).accepted is True
    assert validate(Phase.COOLDOWN, Phase.ESCALATED).accepted is True
    for target in (Phase.OBSERVING, Phase.ENGAGED, Phase.COOLDOWN):
        assert validate(Phase.COOLDOWN, target).accepted is False


def test_idle_starts_observing_or_escalates_via_rule5():
    assert validate(Phase.IDLE, Phase.OBSERVING).accepted is True
    assert validate(Phase.IDLE, Phase.ESCALATED).accepted is True
    for target in (Phase.ENGAGED, Phase.COOLDOWN, Phase.IDLE):
        assert validate(Phase.IDLE, target).accepted is False


def test_no_transition_to_self_is_allowed():
    for phase in ALL_PHASES:
        assert validate(phase, phase).accepted is False


def test_strategy_disabled_rejects_even_an_otherwise_legal_transition():
    result = validate(
        Phase.IDLE, Phase.OBSERVING, strategy="ambient_orient", strategy_disabled=True
    )
    assert result.accepted is False
    assert "ambient_orient" in result.reason


def test_strategy_not_disabled_has_no_effect():
    result = validate(Phase.IDLE, Phase.OBSERVING, strategy="ambient_orient")
    assert result.accepted is True


def test_every_allowed_goal_change_is_accepted():
    for current, targets in ALLOWED_GOAL_CHANGES.items():
        for target in targets:
            result = validate_goal(current, target)
            assert result.accepted is True
            assert result.reason is None


@pytest.mark.parametrize("current", ALL_GOALS)
@pytest.mark.parametrize("proposed", ALL_GOALS)
def test_every_goal_pair_matches_the_table_exactly(current, proposed):
    result = validate_goal(current, proposed)
    expected = proposed in ALLOWED_GOAL_CHANGES.get(current, frozenset())
    assert result.accepted is expected
    if not expected:
        assert result.reason is not None


def test_no_goal_change_to_self_is_allowed():
    for goal in ALL_GOALS:
        assert validate_goal(goal, goal).accepted is False


def test_root_goal_reaches_every_sub_goal():
    assert validate_goal("return_to_bed", "restroom").accepted is True
    assert validate_goal("return_to_bed", "drink_water").accepted is True
    assert validate_goal("return_to_bed", "comfort").accepted is True
    assert validate_goal("return_to_bed", "wait_for_caregiver").accepted is True


def test_sub_goals_return_to_the_parent_goal():
    assert validate_goal("restroom", "return_to_bed").accepted is True
    assert validate_goal("drink_water", "return_to_bed").accepted is True
    assert validate_goal("comfort", "return_to_bed").accepted is True


def test_sub_goals_may_be_interrupted_by_escalation():
    assert validate_goal("restroom", "wait_for_caregiver").accepted is True
    assert validate_goal("drink_water", "wait_for_caregiver").accepted is True
    assert validate_goal("comfort", "wait_for_caregiver").accepted is True


def test_no_direct_switch_between_sibling_sub_goals():
    assert validate_goal("restroom", "drink_water").accepted is False
    assert validate_goal("drink_water", "comfort").accepted is False
    assert validate_goal("comfort", "restroom").accepted is False


def test_wait_for_caregiver_only_returns_to_root():
    assert validate_goal("wait_for_caregiver", "return_to_bed").accepted is True
    assert validate_goal("wait_for_caregiver", "restroom").accepted is False
    assert validate_goal("wait_for_caregiver", "drink_water").accepted is False
    assert validate_goal("wait_for_caregiver", "comfort").accepted is False


def test_unknown_goal_is_rejected():
    result = validate_goal("return_to_bed", "made_up_goal")
    assert result.accepted is False
    result = validate_goal("made_up_goal", "return_to_bed")
    assert result.accepted is False


# --- validate_strategy (issue #14) ---------------------------------------


def test_validate_strategy_accepts_when_not_disabled():
    result = validate_strategy("ambient_orient", disabled=False)
    assert result.accepted is True
    assert result.reason is None


def test_validate_strategy_rejects_when_disabled():
    result = validate_strategy("ambient_orient", disabled=True)
    assert result.accepted is False
    assert "ambient_orient" in result.reason


# --- validate_composition (decision-bench deterministic proxy) -----------


@pytest.mark.parametrize(
    "text",
    [
        "Jean, I know you are thinking about the kids, but Tom is here and we can rest now.",
        "Jean, I hear you are waiting, but let's return to bed so Tom can check on you.",
        "Jean, it is 3 o'clock, and let's rest now while Tom helps you settle in.",
    ],
)
def test_validate_composition_rejects_contrast_or_invented_caregiver_claims(text):
    result = validate_composition(text, PersonProfile(caregiver_name="Tom"))
    assert result.accepted is False
    # Rejection reasons are safe to log and therefore must never echo the
    # private model-composed candidate.
    assert text not in result.reason


@pytest.mark.parametrize(
    "text",
    [
        "Jean, it is 12 o'clock, so let's rest while Tom stays nearby for you.",
        "Jean, I hear you need comfort; Tom is near, and everything is settled for a rest.",
        "Butterflies can be calming while Tom is nearby.",
    ],
)
def test_validate_composition_allows_nearby_without_strengthening_it(text):
    result = validate_composition(text, PersonProfile(caregiver_name="Tom"))
    assert result.accepted is True
    assert result.reason is None


def test_validate_composition_skips_caregiver_claim_check_for_an_empty_name():
    result = validate_composition("Someone is here with you.", PersonProfile(caregiver_name=""))
    assert result.accepted is True


# --- validate_say (issue #14, HANDOFF.md rule 3) -------------------------


def test_validate_say_accepts_one_calm_sentence():
    result = validate_say("Let's rest now.", seconds_since_last_say=None, min_gap_seconds=8.0)
    assert result.accepted is True


def test_validate_say_rejects_empty_text():
    result = validate_say("   ", seconds_since_last_say=None, min_gap_seconds=8.0)
    assert result.accepted is False


def test_validate_say_rejects_more_than_one_sentence():
    result = validate_say(
        "It is night. Let's rest.", seconds_since_last_say=None, min_gap_seconds=8.0
    )
    assert result.accepted is False
    assert "one sentence" in result.reason


def test_validate_say_accepts_a_single_trailing_period():
    result = validate_say("It is night.", seconds_since_last_say=None, min_gap_seconds=8.0)
    assert result.accepted is True


@pytest.mark.parametrize(
    "text",
    [
        "No, that's not right.",
        "You can't do that.",
        "You're wrong about that.",
    ],
)
def test_validate_say_rejects_every_forbidden_phrase(text):
    result = validate_say(text, seconds_since_last_say=None, min_gap_seconds=8.0)
    assert result.accepted is False


def test_validate_say_does_not_false_positive_on_a_word_containing_no():
    # "known" contains "no" as a substring but not as a whole word.
    result = validate_say(
        "It is well known that nights are calm.", seconds_since_last_say=None, min_gap_seconds=8.0
    )
    assert result.accepted is True


def test_validate_say_rejects_a_question_mark():
    result = validate_say(
        "Do you remember where you live?", seconds_since_last_say=None, min_gap_seconds=8.0
    )
    assert result.accepted is False


def test_validate_say_rejects_an_interrogative_opener_even_without_a_mark():
    result = validate_say(
        "Remember your address for me now", seconds_since_last_say=None, min_gap_seconds=8.0
    )
    assert result.accepted is False


def test_validate_say_enforces_the_minimum_gap():
    result = validate_say("Let's rest now.", seconds_since_last_say=3.0, min_gap_seconds=8.0)
    assert result.accepted is False
    assert "8" in result.reason


def test_validate_say_accepts_when_gap_is_satisfied():
    result = validate_say("Let's rest now.", seconds_since_last_say=8.0, min_gap_seconds=8.0)
    assert result.accepted is True


def test_validate_say_no_gap_check_before_any_say():
    result = validate_say("Let's rest now.", seconds_since_last_say=None, min_gap_seconds=8.0)
    assert result.accepted is True
