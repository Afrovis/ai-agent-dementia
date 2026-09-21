from decision_bench.runner import run_scenario, scenario_end
from decision_bench.schema import load_scenarios
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
