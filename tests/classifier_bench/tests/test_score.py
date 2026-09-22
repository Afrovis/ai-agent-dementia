from __future__ import annotations

import pytest

from classifier_bench.score import BASELINE_SEED, format_report, score_records


def test_score_hand_computed_metrics_and_breakdowns() -> None:
    records = [
        {
            "set": "agree",
            "truth": "acceptable",
            "label": "acceptable",
            "confidence": 0.8,
            "action_kind": "goal",
        },
        {
            "set": "agree",
            "truth": "acceptable",
            "label": "forbidden",
            "confidence": 0.6,
            "action_kind": "goal",
        },
        {
            "set": "agree",
            "truth": "must_not",
            "label": "forbidden",
            "confidence": 0.9,
            "action_kind": "say",
        },
        {
            "set": "agree",
            "truth": "must_not",
            "label": "irrelevant",
            "confidence": 0.7,
            "action_kind": "say",
        },
        {
            "set": "disputed",
            "run1": "acceptable",
            "run2": "absent",
            "label": "acceptable",
        },
        {
            "set": "disputed",
            "run1": "must_not",
            "run2": "acceptable",
            "label": "irrelevant",
        },
    ]

    report = score_records(records)
    agree = report["agree"]
    assert agree["accuracy"] == pytest.approx(2 / 3)
    assert agree["forced_accuracy"] == pytest.approx(0.5)
    assert agree["abstention_rate"] == pytest.approx(0.25)
    assert agree["inversions"]["acceptable_as_forbidden"] == 1
    assert agree["inversions"]["acceptable_as_forbidden_rate"] == pytest.approx(0.5)
    assert agree["inversions"]["forbidden_as_acceptable"] == 0
    # p(acceptable) = .8, .4, .1 against outcomes 1, 1, 0.
    assert agree["brier_score"] == pytest.approx((0.04 + 0.36 + 0.01) / 3)
    assert agree["ece_5_bin"] == pytest.approx((0.1 + 0.6 + 0.2) / 3)
    assert agree["baselines"]["always_acceptable"] == pytest.approx(0.5)
    assert agree["baselines"]["always_forbidden"] == pytest.approx(0.5)
    assert agree["baselines"]["uniform_random"] == pytest.approx(0.25)
    assert agree["baselines"]["uniform_random_seed"] == BASELINE_SEED
    assert agree["by_action_kind"]["goal"]["accuracy"] == pytest.approx(0.5)
    assert agree["by_truth"]["must_not"]["forced_accuracy"] == pytest.approx(0.5)
    assert report["disputed"]["matches_run1_rate"] == pytest.approx(0.5)
    assert report["disputed"]["matches_run2_rate"] == 0
    assert report["disputed"]["matches_neither_rate"] == pytest.approx(0.5)
    assert report["disputed"]["abstention_rate"] == pytest.approx(0.5)
    text = format_report(report)
    assert "27.0%" in text
    assert "74.0%" in text
