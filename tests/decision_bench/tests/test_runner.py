from decision_bench.runner import run_scenario, scenario_end
from decision_bench.schema import TimelineEvent, UtteranceInput, load_scenarios
from decision_bench.stub_llm import StubLLM


def _scenario(scenario_id):
    return next(item for item in load_scenarios() if item.id == scenario_id)


def test_fall_reaches_escalated_and_notifies():
    trace = run_scenario(_scenario("fall-01"), llm=StubLLM())
    assert any(item.kind == "State" and item.data["phase"] == "ESCALATED" for item in trace.entries)
    assert any(item.kind == "Notify" for item in trace.entries)


def test_restroom_utterance_sets_restroom_goal():
    trace = run_scenario(_scenario("restroom-01"), llm=StubLLM())
    assert any(item.kind == "State" and item.data["goal"] == "restroom" for item in trace.entries)


def test_quick_false_alarm_never_notifies():
    trace = run_scenario(_scenario("false-alarm-01"), llm=StubLLM())
    assert not any(item.kind == "Notify" for item in trace.entries)


def test_trace_is_monotonic_and_reaches_computed_end():
    scenario = _scenario("fall-01")
    trace = run_scenario(scenario, llm=StubLLM())
    assert [item.t for item in trace.entries] == sorted(item.t for item in trace.entries)
    assert trace.end_t == scenario_end(scenario)
    assert trace.entries[-1].t <= trace.end_t


def test_fixed_latency_blocks_next_utterance_and_times_compose():
    scenario = _scenario("conversation-01")
    scenario = scenario.model_copy(
        update={
            "timeline": (
                *(
                    event.model_copy(
                        update={
                            "utterance": UtteranceInput(
                                text="I have to pick up the kids", duration_s=1.3
                            )
                        }
                    )
                    if event.utterance is not None
                    else event
                    for event in scenario.timeline
                ),
                TimelineEvent(t=56, utterance=UtteranceInput(text="Hello again", duration_s=1)),
            )
        }
    )
    trace = run_scenario(scenario, llm=StubLLM(), llm_latency="fixed:2.5")
    compose_end = next(
        e
        for e in trace.entries
        if e.kind == "Activity" and e.data["kind"] == "compose" and e.data["phase"] == "end"
    )
    late_input = next(
        e for e in trace.entries if e.kind == "Utterance" and e.data["text"] == "Hello again"
    )
    say = next(
        e
        for e in trace.entries
        if e.kind == "Say" and e.data["strategy"] == "validate_and_redirect"
    )
    assert compose_end.t == say.t == 58
    # Planning after the reply consumes another fixed 2.5 s before the input arrives.
    assert late_input.t == 60.5
    assert compose_end.data["duration_ms"] == 2500
    assert say.data["ts"] == "2026-01-01T02:20:58Z"


def test_recorded_latency_requires_capture():
    import pytest

    with pytest.raises(ValueError, match="no captures"):
        run_scenario(_scenario("conversation-01"), llm=StubLLM(), llm_latency="recorded")
