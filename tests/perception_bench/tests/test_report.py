"""Tests for `perception_bench.report`, in particular the exit-code gate:
non-zero only when a *measured, non-advisory* target was missed, zero
when a tier or state simply was not measured, and zero when only tier 1
(advisory, synthetic smoke-test) targets were missed.

`test_exit_code_is_nonzero_for_a_deliberately_failing_tier2_case` is
exactly that: a hand-built `AccuracyResult` well below the 95% target on
tier 2, to prove the gate actually fires on a *gating* tier rather than
assuming it does. Kept as a permanent regression test rather than
removed, per this issue's verification ask.
"""

from perception_bench.daylight import Tier2Result
from perception_bench.infrared import Tier3Result
from perception_bench.report import build_report, report_to_dict
from perception_bench.scoring import LatencyResult, score_predictions
from perception_bench.synthetic import ClipRun

EMPTY_TIER2 = Tier2Result(
    accuracy=None,
    skipped_reason="no data",
    degraded=False,
    clip_count=0,
    pose_backend_name="mediapipe",
)
EMPTY_TIER3 = Tier3Result(clip_count=0, missing_reason="no manifest")


def _fake_run(pairs, latency_samples=(), undetected=()):
    return ClipRun(
        clip_name="fake",
        pairs=pairs,
        latency=LatencyResult(samples=list(latency_samples)),
        undetected_transitions=list(undetected),
    )


def test_exit_code_is_zero_when_targets_pass():
    pairs = [("standing", "standing")] * 20 + [("on_floor", "on_floor")] * 20
    run = _fake_run(pairs, latency_samples=[("-> on_floor", 0.5)])
    accuracy = score_predictions(pairs)
    latency = run.latency

    report = build_report([run], accuracy, latency, EMPTY_TIER2, EMPTY_TIER3)
    assert report.exit_code == 0


def test_exit_code_is_nonzero_for_a_deliberately_failing_tier2_case():
    # Deliberately bad: standing correctly recognised only half the time
    # on tier 2 (daylight footage), well under the >95% target -- proves
    # the gate fires on a real miss on a tier that actually gates.
    bad_pairs = [("standing", "standing")] * 5 + [("standing", "sitting_up")] * 5
    good_run = _fake_run([("standing", "standing")] * 20 + [("on_floor", "on_floor")] * 20)
    tier1_accuracy = score_predictions(good_run.pairs)

    tier2_accuracy = score_predictions(bad_pairs)
    tier2 = Tier2Result(
        accuracy=tier2_accuracy,
        skipped_reason=None,
        degraded=False,
        clip_count=1,
        pose_backend_name="mediapipe",
    )

    report = build_report([good_run], tier1_accuracy, good_run.latency, tier2, EMPTY_TIER3)
    assert report.exit_code != 0
    standing_verdicts = [v for v in report.verdicts if "tier2" in v.label and "standing" in v.label]
    assert any(v.passed is False for v in standing_verdicts)


def test_tier1_failure_is_advisory_and_does_not_gate_the_exit_code():
    # The known real failure this issue surfaced: tier 1 (synthetic
    # clips) misses standing recall and latency, but tier 1 is a smoke
    # test against generated-in-process clips, not the real recorded
    # footage the targets in PLAN.md section 12 are defined against.
    pairs = [("standing", "standing")] * 5 + [("standing", "sitting_up")] * 5
    run = _fake_run(pairs, latency_samples=[("-> on_floor", 5.0)])
    accuracy = score_predictions(pairs)

    report = build_report([run], accuracy, run.latency, EMPTY_TIER2, EMPTY_TIER3)
    tier1_verdicts = [v for v in report.verdicts if v.label.startswith("tier1")]
    assert any(v.passed is False for v in tier1_verdicts)
    assert all(v.advisory for v in tier1_verdicts)
    assert report.exit_code == 0


def test_exit_code_is_zero_when_a_tier_is_simply_not_measured():
    pairs = [("standing", "standing")] * 20 + [("on_floor", "on_floor")] * 20
    run = _fake_run(pairs, latency_samples=[("-> on_floor", 0.5)])
    accuracy = score_predictions(pairs)
    latency = run.latency

    report = build_report([run], accuracy, latency, EMPTY_TIER2, EMPTY_TIER3)
    tier2_verdicts = [v for v in report.verdicts if v.label.startswith("tier2")]
    tier3_verdicts = [v for v in report.verdicts if v.label.startswith("tier3")]
    assert all(v.passed is None for v in tier2_verdicts)
    assert all(v.passed is None for v in tier3_verdicts)
    assert report.exit_code == 0


def test_exit_code_is_zero_when_no_latency_transitions_were_measured():
    pairs = [("standing", "standing")] * 20 + [("on_floor", "on_floor")] * 20
    run = _fake_run(pairs, latency_samples=[])  # no transitions scripted
    accuracy = score_predictions(pairs)

    report = build_report([run], accuracy, run.latency, EMPTY_TIER2, EMPTY_TIER3)
    latency_verdicts = [v for v in report.verdicts if "latency" in v.label]
    assert all(v.passed is None for v in latency_verdicts)
    assert report.exit_code == 0


def test_report_to_dict_is_json_serialisable():
    import json

    pairs = [("standing", "standing")] * 20 + [("on_floor", "on_floor")] * 20
    run = _fake_run(pairs, latency_samples=[("-> on_floor", 0.5)])
    accuracy = score_predictions(pairs)

    report = build_report([run], accuracy, run.latency, EMPTY_TIER2, EMPTY_TIER3)
    encoded = json.dumps(report_to_dict(report))
    decoded = json.loads(encoded)
    assert decoded["exit_code"] == 0
    assert decoded["tier1"]["per_state"]["standing"]["recall"] == 1.0
