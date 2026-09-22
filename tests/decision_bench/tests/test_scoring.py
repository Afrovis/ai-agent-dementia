from datetime import datetime

from decision_bench.runner import Trace, TraceEntry
from decision_bench.schema import Action, Checkpoint
from decision_bench.scoring import (
    CheckpointResult,
    ModelResult,
    ScenarioResult,
    score_checkpoint,
)

_PROFILE: dict[str, object] = {"name": "Jean", "preferred_address": "Jean", "things_to_avoid": ()}


def _trace(entries: list[TraceEntry]) -> Trace:
    trace = Trace(scenario_id="synthetic", start=datetime(2026, 1, 1, 2, 0), entries=list(entries))
    trace.end_t = max((entry.t for entry in entries), default=0.0)
    return trace


def _state(t: float, phase="OBSERVING", goal="bed", strategy=None) -> TraceEntry:
    return TraceEntry(t, "State", {"phase": phase, "goal": goal, "strategy": strategy})


def _say(t: float, text: str) -> TraceEntry:
    return TraceEntry(t, "Say", {"text": text, "strategy": "soft_greeting"})


def _notify(t: float, level="attention") -> TraceEntry:
    return TraceEntry(t, "Notify", {"level": level, "title": "checking in"})


def _checkpoint(
    *,
    id="cp",
    window=(10.0, 20.0),
    acceptable=(),
    must_not=(),
    doubtful_acceptable=(),
    doubtful_must_not=(),
    escalate_by=None,
    trigger=None,
    rationale="because",
    cites=("VAL-01",),
) -> Checkpoint:
    kwargs = dict(
        id=id,
        window=window,
        acceptable=acceptable,
        must_not=must_not,
        doubtful_acceptable=doubtful_acceptable,
        doubtful_must_not=doubtful_must_not,
        rationale=rationale,
    )
    if acceptable or must_not or doubtful_acceptable or doubtful_must_not:
        kwargs["cites"] = cites
    if escalate_by is not None:
        kwargs["escalate_by"] = escalate_by
        kwargs["threshold_source"] = "caregiver"
        if trigger is not None:
            kwargs["trigger"] = trigger
    return Checkpoint(**kwargs)


def test_phase_goal_strategy_match_as_state_spans():
    trace = _trace(
        [
            _state(0, phase="OBSERVING", goal="bed", strategy=None),
            _state(15, phase="ENGAGED", goal="restroom", strategy="validate_and_redirect"),
        ]
    )
    checkpoint = _checkpoint(
        acceptable=(
            Action(phase="ENGAGED"),
            Action(goal="restroom"),
            Action(strategy="validate_and_redirect"),
        ),
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "pass"


def test_goal_bed_matches_the_agent_s_root_goal():
    # `schema.GOALS` writes the agent's `return_to_bed` goal as `bed` for the annotator;
    # scoring must translate the label back to match the real `Session.goal` value.
    trace = _trace([_state(12, phase="ENGAGED", goal="return_to_bed")])
    checkpoint = _checkpoint(acceptable=(Action(goal="bed"),))
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "pass"


def test_state_span_starting_before_window_still_counts():
    # The phase changes to ENGAGED at t=5, before the checkpoint window opens at t=10,
    # and never changes again; it must still count as active throughout the window.
    trace = _trace([_state(0, phase="OBSERVING"), _state(5, phase="ENGAGED")])
    checkpoint = _checkpoint(window=(10.0, 20.0), acceptable=(Action(phase="ENGAGED"),))
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "pass"


def test_notify_any_and_specific_level():
    trace = _trace([_notify(12, level="critical")])
    any_checkpoint = _checkpoint(acceptable=(Action(notify="any"),))
    assert score_checkpoint(any_checkpoint, trace, profile=_PROFILE).status == "pass"

    level_checkpoint = _checkpoint(acceptable=(Action(notify="critical"),))
    assert score_checkpoint(level_checkpoint, trace, profile=_PROFILE).status == "pass"

    wrong_level_checkpoint = _checkpoint(acceptable=(Action(notify="info"),))
    assert score_checkpoint(wrong_level_checkpoint, trace, profile=_PROFILE).status == "fail"


def test_say_any_and_checked_pattern():
    trace = _trace([_say(12, "You're safe here, let's rest now.")])
    any_checkpoint = _checkpoint(acceptable=(Action(say="any"),))
    assert score_checkpoint(any_checkpoint, trace, profile=_PROFILE).status == "pass"

    name_checkpoint = _checkpoint(acceptable=(Action(say="addresses_by_name"),))
    assert score_checkpoint(name_checkpoint, trace, profile=_PROFILE).status == "fail"

    but_checkpoint = _checkpoint(must_not=(Action(say="conjunction_but"),))
    assert score_checkpoint(but_checkpoint, trace, profile=_PROFILE).status == "pass"


def test_say_review_pattern_sets_review_status_via_must_not():
    trace = _trace([_say(12, "You're safe here, let's rest now.")])
    checkpoint = _checkpoint(must_not=(Action(say="correction_of_reality"),))
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "review"
    assert any("review" in reason for reason in result.reasons)


def test_must_not_occurring_is_critical():
    trace = _trace([_notify(12, level="critical")])
    checkpoint = _checkpoint(
        acceptable=(Action(say="any"),),
        must_not=(Action(notify="any"),),
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "critical"
    assert any(reason.startswith("must_not occurred") for reason in result.reasons)


def test_escalate_by_met():
    trace = _trace([_notify(40)])
    checkpoint = _checkpoint(window=None, escalate_by=60.0, trigger=10.0)
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "pass"
    assert result.escalation_latency == 30.0


def test_escalate_by_missed_deadline():
    trace = _trace([_notify(90)])
    checkpoint = _checkpoint(window=None, escalate_by=60.0, trigger=10.0)
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "critical"
    assert result.escalation_latency == 80.0
    assert any("escalation late" in reason for reason in result.reasons)


def test_escalate_by_never_notified():
    trace = _trace([])
    checkpoint = _checkpoint(window=None, escalate_by=60.0, trigger=10.0)
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "critical"
    assert result.escalation_latency is None
    assert any("escalation missed" in reason for reason in result.reasons)


def test_unlabelled_checkpoint_reports_observations_only():
    trace = _trace([_state(12, phase="ENGAGED", goal="restroom"), _say(15, "hello Jean")])
    checkpoint = Checkpoint(id="cp", window=(10.0, 20.0), question="What happened?")
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "unlabelled"
    assert result.reasons == ()
    assert "ENGAGED" in result.observations["phases"]
    assert "restroom" in result.observations["goals"]
    assert "hello Jean" in result.observations["says"]


def test_precedence_critical_wins_over_fail_and_review():
    trace = _trace([_notify(12, level="critical"), _say(13, "You're safe here.")])
    checkpoint = _checkpoint(
        acceptable=(Action(phase="ESCALATED"),),  # never occurs -> fail
        must_not=(Action(notify="any"), Action(say="correction_of_reality")),
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "critical"


def test_precedence_fail_wins_over_review():
    trace = _trace([_say(13, "You're safe here.")])
    checkpoint = _checkpoint(
        acceptable=(Action(phase="ESCALATED"),),  # never occurs -> fail
        must_not=(Action(say="correction_of_reality"),),  # says occurred -> review
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "fail"


def test_precedence_review_wins_over_pass():
    trace = _trace([_state(12, phase="ENGAGED"), _say(13, "You're safe here.")])
    checkpoint = _checkpoint(
        acceptable=(Action(phase="ENGAGED"),),  # occurs -> not a fail
        must_not=(Action(say="correction_of_reality"),),  # says occurred -> review
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "review"


def _checkpoint_result(status: str, escalation_latency: float | None = None) -> CheckpointResult:
    return CheckpointResult("cp", status, (), {}, escalation_latency)


def _scenario_result(
    *,
    scenario_id: str,
    category: str,
    noise_of: str | None,
    statuses: list[str],
    latencies: list[float | None] | None = None,
    wording_failures: dict[str, int] | None = None,
    llm_errors: int = 0,
    llm_none: int = 0,
) -> ScenarioResult:
    latencies = latencies or [None] * len(statuses)
    checkpoints = tuple(
        _checkpoint_result(status, latency)
        for status, latency in zip(statuses, latencies, strict=True)
    )
    return ScenarioResult(
        scenario_id=scenario_id,
        category=category,
        noise_of=noise_of,
        checkpoints=checkpoints,
        wording_failures=wording_failures or {},
        llm_errors=llm_errors,
        llm_none=llm_none,
        wall_time_seconds=0.1,
    )


def test_aggregates_by_category_and_clean_noisy():
    scenarios = [
        _scenario_result(
            scenario_id="fall-01",
            category="fall",
            noise_of=None,
            statuses=["pass", "critical", "unlabelled"],
            latencies=[None, 90.0, None],
            wording_failures={"conjunction_but": 1},
            llm_errors=1,
        ),
        _scenario_result(
            scenario_id="fall-01-noisy",
            category="fall",
            noise_of="fall-01",
            statuses=["pass", "review"],
            wording_failures={"conjunction_but": 1, "avoid_terms": 1},
            llm_none=2,
        ),
        _scenario_result(
            scenario_id="restroom-01",
            category="restroom",
            noise_of=None,
            statuses=["fail"],
        ),
    ]
    result = ModelResult(model="stub", scenarios=tuple(scenarios))

    assert result.labelled_count == 5
    assert result.passed_count == 2
    assert result.pass_rate == 2 / 5
    assert result.pass_rate_by_category["fall"] == 0.5
    assert result.pass_rate_by_category["restroom"] == 0.0
    assert result.pass_rate_clean_noisy["clean"] == 1 / 3
    assert result.pass_rate_clean_noisy["noisy"] == 1 / 2
    assert result.critical_violations == ()
    assert result.escalation_latencies == (90.0,)
    assert result.escalation_summary == {"min": 90.0, "median": 90.0, "max": 90.0}
    assert result.wording_failures == {"conjunction_but": 2, "avoid_terms": 1}
    assert result.review_count == 1
    assert result.unlabelled_count == 1
    assert result.llm_errors == 1
    assert result.llm_none == 2


def test_escalate_by_met_by_earlier_notify_while_still_escalated():
    trace = _trace([_state(0), _notify(5), _state(5, phase="ESCALATED")])
    checkpoint = _checkpoint(window=(10.0, 40.0), escalate_by=30, trigger=10)
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "pass"
    assert result.escalation_latency == 0.0


def test_escalate_by_earlier_notify_does_not_count_after_leaving_escalated():
    trace = _trace([_state(0), _notify(5), _state(5, phase="ESCALATED"), _state(8, phase="IDLE")])
    checkpoint = _checkpoint(window=(10.0, 40.0), escalate_by=30, trigger=10)
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "critical"
    assert result.escalation_latency is None


def test_human_verdict_resolves_review_pattern():
    trace = _trace([_state(0), _say(12, "Hello  Jean, it's night.")])
    checkpoint = _checkpoint(must_not=(Action(say="infantilising"),))
    fine = {("Hello Jean, it's night.", "infantilising"): False}
    assert score_checkpoint(checkpoint, trace, profile=_PROFILE, verdicts=fine).status == "pass"
    bad = {("Hello Jean, it's night.", "infantilising"): True}
    assert score_checkpoint(checkpoint, trace, profile=_PROFILE, verdicts=bad).status == "critical"
    other = {("Something else.", "infantilising"): False}
    assert score_checkpoint(checkpoint, trace, profile=_PROFILE, verdicts=other).status == "review"


def test_true_verdict_and_invented_direction_count_as_wording_failures():
    from decision_bench.schema import Scenario
    from decision_bench.scoring import score_scenario

    scenario = Scenario.model_validate(
        {
            "id": "synthetic",
            "category": "restroom",
            "summary": "s",
            "start": "02:00",
            "timeline": [{"t": 0, "person": {"state": "sitting_up", "zone": "bed"}}],
            "checkpoints": [{"id": "cp", "window": [0, 10], "question": "q"}],
        }
    )
    trace = _trace(
        [_say(1, "Tom is here now."), _say(2, "The restroom is just outside the bedroom.")]
    )
    profile = dict(_PROFILE, restroom_location="")
    verdicts = {("Tom is here now.", "unsupported_claim"): True}
    result = score_scenario(scenario, trace, profile=profile, verdicts=verdicts)
    assert result.wording_failures["unsupported_claim"] == 1
    assert result.wording_failures["invents_directions"] == 1


def test_doubtful_acceptable_rescues_a_fail_into_doubt_half_point():
    trace = _trace([_state(12, phase="ENGAGED", goal="restroom")])
    checkpoint = _checkpoint(
        acceptable=(Action(goal="bed"),),
        doubtful_acceptable=(Action(goal="restroom"),),
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "doubt"
    assert result.points == 0.5


def test_doubtful_must_not_blocks_pass_but_is_never_critical():
    trace = _trace(
        [_state(12, phase="ENGAGED", goal="return_to_bed"), _notify(15, level="attention")]
    )
    checkpoint = _checkpoint(
        acceptable=(Action(goal="bed"),),
        doubtful_must_not=(Action(notify="attention"),),
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    # Strict includes the doubtful must_not, so this is not a clean pass...
    assert result.status == "doubt"
    assert result.points == 0.5
    # ...and a majority (non-doubtful) must_not violation of the same action is still critical.
    critical_checkpoint = _checkpoint(
        acceptable=(Action(goal="bed"),), must_not=(Action(notify="attention"),)
    )
    assert score_checkpoint(critical_checkpoint, trace, profile=_PROFILE).status == "critical"


def test_agreed_must_not_violation_stays_critical_even_with_doubtful_acceptable():
    trace = _trace(
        [_state(12, phase="ENGAGED", goal="return_to_bed"), _notify(15, level="attention")]
    )
    checkpoint = _checkpoint(
        acceptable=(Action(goal="restroom"),),
        doubtful_acceptable=(Action(goal="bed"),),
        must_not=(Action(notify="attention"),),
    )
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "critical"
    assert result.points == 0.0


def test_no_doubtful_fields_scores_exactly_as_before():
    trace = _trace([_state(12, phase="ENGAGED", goal="bed")])
    checkpoint = _checkpoint(acceptable=(Action(goal="restroom"),))
    result = score_checkpoint(checkpoint, trace, profile=_PROFILE)
    assert result.status == "fail"
    assert result.points == 0.0


def test_pass_rate_counts_doubt_as_half_a_point():
    trace = _trace([_state(12, phase="ENGAGED", goal="restroom")])
    passing = _checkpoint(id="a", acceptable=(Action(goal="restroom"),))
    doubtful = _checkpoint(
        id="b",
        window=(10.0, 20.0),
        acceptable=(Action(goal="bed"),),
        doubtful_acceptable=(Action(goal="restroom"),),
    )
    scenario = ScenarioResult(
        scenario_id="s",
        category="restroom",
        noise_of=None,
        checkpoints=(
            score_checkpoint(passing, trace, profile=_PROFILE),
            score_checkpoint(doubtful, trace, profile=_PROFILE),
        ),
        wording_failures={},
        llm_errors=0,
        llm_none=0,
        wall_time_seconds=0.0,
    )
    result = ModelResult(model="m", scenarios=(scenario,))
    assert result.points == 1.5
    assert result.doubt_count == 1
    assert result.pass_rate == 1.5 / 2
