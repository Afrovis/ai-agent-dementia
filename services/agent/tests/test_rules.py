"""Tests for `agent.rules`: every accept/reject case in `ALLOWED_TRANSITIONS`,
including its exact boundaries, plus the strategy-disabled seam."""

import pytest

from agent.rules import ALLOWED_TRANSITIONS, Phase, validate

ALL_PHASES = list(Phase)


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
