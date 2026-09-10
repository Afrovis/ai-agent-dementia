"""Tests for `perception_bench.scoring`. No `perceive` import needed here:
these functions take and return plain data."""

from perception_bench.scoring import (
    STATE_NAMES,
    LatencyResult,
    score_predictions,
    target_states_meet_accuracy,
)


def test_score_predictions_perfect_agreement():
    pairs = [("standing", "standing")] * 5 + [("on_floor", "on_floor")] * 3
    result = score_predictions(pairs)

    assert result.overall_accuracy == 1.0
    assert result.per_state["standing"].recall == 1.0
    assert result.per_state["standing"].precision == 1.0
    assert result.per_state["on_floor"].recall == 1.0


def test_score_predictions_confusion_and_recall():
    pairs = [
        ("standing", "standing"),
        ("standing", "sitting_up"),  # missed
        ("standing", "standing"),
        ("sitting_up", "standing"),  # false positive for standing
    ]
    result = score_predictions(pairs)

    assert result.per_state["standing"].recall == 2 / 3
    assert result.per_state["standing"].true_positives == 2
    assert result.per_state["standing"].false_positives == 1
    assert result.confusion["standing"]["sitting_up"] == 1
    assert result.confusion["sitting_up"]["standing"] == 1


def test_score_predictions_zero_support_state_has_no_recall():
    result = score_predictions([("standing", "standing")])
    assert result.per_state["on_floor"].support == 0
    assert result.per_state["on_floor"].recall is None
    assert result.per_state["on_floor"].precision is None


def test_every_state_name_appears_in_confusion_matrix_even_at_zero():
    result = score_predictions([])
    for state in STATE_NAMES:
        assert result.confusion[state] == dict.fromkeys(STATE_NAMES, 0)
    assert result.overall_accuracy is None


def test_target_states_meet_accuracy_reports_none_for_no_support():
    result = score_predictions([("sitting_up", "sitting_up")])
    verdicts = target_states_meet_accuracy(result)
    assert verdicts["standing"] is None
    assert verdicts["on_floor"] is None


def test_target_states_meet_accuracy_pass_and_fail():
    pairs = [("standing", "standing")] * 19 + [("standing", "sitting_up")]  # 95% recall exactly
    result = score_predictions(pairs)
    verdicts = target_states_meet_accuracy(result, states=("standing",), target=0.95)
    assert verdicts["standing"] is True

    pairs_fail = [("standing", "standing")] * 9 + [("standing", "sitting_up")]  # 90%
    result_fail = score_predictions(pairs_fail)
    verdicts_fail = target_states_meet_accuracy(result_fail, states=("standing",), target=0.95)
    assert verdicts_fail["standing"] is False


def test_latency_result_within_target():
    assert LatencyResult(samples=[("a", 0.5), ("b", 1.9)]).within_target(2.0) is True
    assert LatencyResult(samples=[("a", 0.5), ("b", 2.1)]).within_target(2.0) is False
    assert LatencyResult(samples=[]).within_target(2.0) is None


def test_latency_result_max_and_mean():
    latency = LatencyResult(samples=[("a", 1.0), ("b", 3.0)])
    assert latency.max_latency_s == 3.0
    assert latency.mean_latency_s == 2.0
    assert LatencyResult(samples=[]).max_latency_s is None
