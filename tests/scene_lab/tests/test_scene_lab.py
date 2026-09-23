from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from scene_lab.bugs import (
    RunDir,
    fingerprint,
    harness_error,
    merge,
    render_merge_md,
    results_to_entries,
)
from scene_lab.invariants import InvariantResult, check_trace, is_question_or_request
from scene_lab.thresholds import load
from scene_lab.trace import (
    Trace,
    TraceEvent,
    from_agent_log,
    from_decision_bench,
    from_export,
    from_session_replay,
    playbacks,
)


def ev(t, type, **data):
    kind = (
        "input"
        if type in {"PersonState", "Utterance", "SpeechStarted"}
        else "activity"
        if type == "Activity"
        else "output"
    )
    return TraceEvent(t=t, kind=kind, type=type, data=data)


def scene(*events, end=120, meta=None):
    return Trace(id="hand-built", source="live", events=list(events), end_t=end, meta=meta or {})


def score(trace, rule):
    return [r for r in check_trace(trace, load()) if r.id == rule]


def fail(trace, rule, severity, reason):
    hits = [r for r in score(trace, rule) if not r.passed]
    assert hits and hits[0].severity == severity and reason in hits[0].reason


def passes(trace, rule):
    hits = score(trace, rule)
    assert hits and all(r.passed for r in hits)


def baseline(*extra, end=120):
    return scene(
        ev(0, "PersonState", state="walking", zone="door"),
        ev(0, "SessionState", phase="ENGAGED", goal="return_to_bed", strategy_index=0),
        *extra,
        ev(end, "SessionState", phase="IDLE", goal="return_to_bed", strategy_index=0),
        end=end,
    )


def test_tt1_and_tm2():
    good = baseline(
        ev(2, "Utterance", text="Hello"),
        ev(4, "Say", text="Hello there.", strategy="greet", interruptible=True),
    )
    passes(good, "TT-1")
    stats = score(good, "TM-2")[0].context
    assert stats["n"] == 1 and stats["p50"] == 2
    bad = baseline(ev(2, "Utterance", text="Hello"), end=40)
    fail(bad, "TT-1", "critical", "never composed")
    assert score(bad, "TM-2")[0].context["n"] == 0
    late = baseline(ev(2, "Utterance", text="Hello"), ev(9, "Say", text="Hello", strategy="greet"))
    fail(late, "TT-1", "major", "late reply")
    dropped = baseline(
        ev(2, "Utterance", text="Where am I?"),
        TraceEvent(
            t=3,
            kind="decision",
            type="Activity",
            data={"decision": "pending_say_dropped", "reason": "strategy_changed"},
        ),
    )
    fail(dropped, "TT-1", "critical", "dropped pending say: strategy_changed")
    assert [r for r in score(dropped, "TT-1") if not r.passed][0].context[
        "drop_reason"
    ] == "strategy_changed"


def test_tt1_deliberate_silence_passes_unless_it_answers_a_question():
    for text, severity, passed, cause in (
        ("I am waiting", "info", True, "no reply by design: reassured_enough"),
        (
            "Is anyone there",
            "review",
            False,
            "no reply by design to a question: reassured_enough",
        ),
    ):
        trace = baseline(
            ev(2, "Utterance", text=text),
            TraceEvent(
                t=3,
                kind="decision",
                type="Activity",
                data={"decision": "no_reply", "reason": "reassured_enough"},
            ),
            end=40,
        )
        result = next(r for r in score(trace, "TT-1") if cause in r.reason)
        assert (result.severity, result.passed, result.context["drop_reason"]) == (
            severity,
            passed,
            "reassured_enough",
        )


def _said(t, strategy, reply, trigger):
    return TraceEvent(
        t=t,
        kind="decision",
        type="Activity",
        data={"decision": "said", "strategy": strategy, "reply": reply, "trigger": trigger},
    )


def test_tt1_scheduled_step_is_not_a_reply():
    ladder = baseline(
        ev(2, "Utterance", text="Where is Tom?"),
        ev(4, "Say", text="Hello Jean, it's night-time.", strategy="soft_greeting"),
        _said(4, "soft_greeting", False, "strategy_advanced"),
    )
    fail(ladder, "TT-1", "critical", "only a scheduled soft_greeting step followed")
    reply = baseline(
        ev(2, "Utterance", text="Where is Tom?"),
        ev(4, "Say", text="Tom is safe.", strategy="validate_and_redirect"),
        _said(4, "validate_and_redirect", True, "utterance_reply"),
    )
    passes(reply, "TT-1")


def test_tt2_review_and_question_heuristic():
    assert is_question_or_request("Could you help me")
    assert is_question_or_request("I need the loo")
    good = baseline(
        ev(2, "Utterance", text="Where is Tom?"),
        ev(3, "Say", text="Tom is safe.", strategy="greet"),
    )
    hit = score(good, "TT-2")[0]
    assert hit.passed and hit.severity == "review" and hit.reason == "unrated"
    assert score(good, "TT-2")[0].evidence == ["utterance: Where is Tom?", "reply: Tom is safe."]
    bad = baseline(ev(2, "Utterance", text="Help me"))
    assert score(bad, "TT-2")[0].reason == "no_reply"
    assert score(good, "TT-2")[0].passed
    judged = [r for r in check_trace(good, load(), judge=lambda u, r: "answered") if r.id == "TT-2"]
    assert judged[0].reason == "answered" and judged[0].passed


def test_tt3_and_tt4():
    good = baseline(
        ev(1, "SpeechStarted"),
        ev(3, "Utterance", text="hello"),
        ev(5, "Say", text="hi", strategy="greet"),
    )
    passes(good, "TT-3")
    bad = baseline(
        ev(1, "SpeechStarted"),
        ev(2, "Say", text="hi", strategy="greet"),
        ev(3, "Utterance", text="hello"),
    )
    fail(bad, "TT-3", "major", "over speech")
    real_good = scene(
        ev(0, "Say", text="hello", strategy="greet", interruptible=True),
        ev(0.2, "Activity", kind="playback", detail="playing"),
        ev(1, "SpeechStarted"),
        ev(1.2, "Activity", kind="playback", detail="interrupted"),
        end=2,
    )
    passes(real_good, "TT-4")
    real_bad = scene(
        ev(0, "Say", text="hello", strategy="greet", interruptible=True),
        ev(0.2, "Activity", kind="playback", detail="playing"),
        ev(1, "SpeechStarted"),
        ev(2, "Activity", kind="playback", detail="ended"),
        end=3,
    )
    fail(real_bad, "TT-4", "major", "did not stop")
    passes(
        scene(ev(0, "Say", text="estimated", strategy="greet"), ev(0.5, "SpeechStarted")), "TT-4"
    )


def test_tt5_tt6_tt7():
    good = scene(
        ev(0, "PersonState", state="walking", zone="door"),
        ev(0, "Say", text="first", strategy="greet"),
        ev(12, "Say", text="second", strategy="greet"),
        end=20,
    )
    passes(good, "TT-5")
    bad = scene(
        ev(0, "Say", text="first", strategy="greet"),
        ev(3, "Say", text="second", strategy="greet"),
        end=20,
    )
    fail(bad, "TT-5", "minor", "silence gap")
    real = scene(
        ev(0, "Say", text="first", strategy="greet"),
        ev(0.1, "Activity", kind="playback", detail="playing"),
        ev(1, "Activity", kind="playback", phase="end", detail="ended"),
        ev(3, "Say", text="second", strategy="greet"),
        ev(3.1, "Activity", kind="playback", detail="playing"),
        ev(4, "Activity", kind="playback", phase="end", detail="ended"),
    )
    fail(real, "TT-5", "major", "silence gap")
    passes(good, "TT-6")
    fail(
        scene(
            ev(0, "PersonState", state="absent", zone="other"),
            ev(2, "Say", text="hello", strategy="greet"),
        ),
        "TT-6",
        "major",
        "absent",
    )
    # Recent speech counts as presence, as in the agent.
    passes(
        scene(
            ev(0, "PersonState", state="absent", zone="other"),
            ev(10, "Utterance", text="Can I go back to bed?"),
            ev(11, "Say", text="Let's go back to bed.", strategy="guided_return"),
        ),
        "TT-6",
    )
    passes(
        scene(
            ev(0, "Utterance", text="hi"),
            ev(1, "Say", text="hi", strategy="greet"),
            ev(2, "Utterance", text="again"),
        ),
        "TT-7",
    )
    fail(
        scene(
            ev(0, "Utterance", text="first"),
            ev(7, "Utterance", text="second"),
            ev(7.3, "Say", text="first reply", strategy="greet"),
        ),
        "TT-7",
        "minor",
        "stale",
    )


def test_sm1_sm2():
    good = scene(
        ev(0, "PersonState", state="in_bed", zone="bed"),
        ev(31, "Say", text="help is coming", strategy="escalate_help"),
        end=40,
    )
    passes(good, "SM-1")
    bad = scene(
        ev(0, "PersonState", state="in_bed", zone="bed"),
        ev(31, "Say", text="good night", strategy="greet"),
        end=40,
    )
    fail(bad, "SM-1", "critical", "after settling")
    advance = scene(
        ev(0, "PersonState", state="in_bed", zone="bed"),
        ev(0, "SessionState", phase="ENGAGED", goal="rest", strategy_index=0),
        ev(31, "SessionState", phase="ENGAGED", goal="rest", strategy_index=1),
        end=40,
    )
    fail(advance, "SM-1", "critical", "after settling")
    good_alert = scene(
        ev(0, "PersonState", state="on_floor", zone="other"), ev(2, "Notify", level="critical")
    )
    passes(good_alert, "SM-2")
    fail(
        scene(ev(0, "PersonState", state="in_bed", zone="bed"), ev(2, "Notify", level="attention")),
        "SM-2",
        "critical",
        "without floor",
    )


def test_sm3_sm4():
    passes(
        baseline(
            ev(10, "Say", text="hello", strategy="greet"),
            ev(70, "Say", text="hello", strategy="greet"),
        ),
        "SM-3",
    )
    fail(baseline(end=100), "SM-3", "major", "silent session")
    for gap, fails in ((130, False), (160, True)):
        escalated = scene(
            ev(0, "PersonState", state="on_floor", zone="other"),
            ev(0, "SessionState", phase="ESCALATED", goal="wait_for_caregiver"),
            ev(0, "Say", text="Help is coming.", strategy="escalate_phone"),
            ev(gap, "Say", text="I'm here with you.", strategy="reassure_waiting"),
            end=gap,
        )
        assert bool([r for r in score(escalated, "SM-3") if not r.passed]) is fails
    fail(baseline(end=70), "SM-3", "major", "silent session")
    returns_to_bed = baseline(ev(20, "PersonState", state="in_bed", zone="bed"))
    passes(returns_to_bed, "SM-3")
    fail(
        scene(
            ev(0, "SessionState", phase="ENGAGED", goal="return_to_bed", strategy_index=0),
            ev(1, "PersonState", state="in_bed", zone="bed"),
            end=120,
        ),
        "SM-3",
        "major",
        "did not end",
    )
    # A person still up at the end of a timeline is an open session, not a failure.
    still_up = scene(
        ev(0, "SessionState", phase="ENGAGED", goal="return_to_bed", strategy_index=0),
        ev(1, "PersonState", state="sitting_up", zone="bed"),
        ev(30, "Say", text="hello", strategy="greet"),
        end=60,
    )
    assert not [r for r in score(still_up, "SM-3") if not r.passed]
    good = scene(
        ev(0, "LightCommand", state="on"),
        ev(1, "PersonState", state="in_bed", zone="bed"),
        ev(35, "LightCommand", state="off"),
        end=40,
    )
    passes(good, "SM-4")
    fail(
        scene(
            ev(0, "LightCommand", state="on"),
            ev(1, "PersonState", state="in_bed", zone="bed"),
            end=40,
        ),
        "SM-4",
        "minor",
        "light remained on",
    )


def test_sm5_skip_and_veto():
    # Night window defaults to True and things_to_avoid comes from the profile, so no meta
    # is needed for a plain veto rule.
    good = baseline(ev(2, "Say", text="Hello", strategy="greet"))
    passes(good, "SM-5")
    bad = baseline(ev(2, "Say", text="Do you remember?", strategy="greet"))
    fail(bad, "SM-5", "critical", "no_memory_question")
    # guided_return after a stated toilet need is denied until the need is resolved.
    toilet = baseline(
        ev(2, "Utterance", text="I need the toilet"),
        ev(4, "Say", text="Let's go back to bed.", strategy="guided_return"),
    )
    hits = [r for r in score(toilet, "SM-5") if not r.passed]
    assert hits and hits[0].severity == "critical"
    # Something said after the request might have been an unseen wants_bed: only a suspicion.
    maybe = baseline(
        ev(2, "Utterance", text="I need the toilet"),
        ev(3, "Utterance", text="Actually I just want to sleep"),
        ev(4, "Say", text="Let's go back to bed.", strategy="guided_return"),
    )
    hits = [r for r in score(maybe, "SM-5") if not r.passed]
    assert hits and hits[0].severity == "minor" and "possible veto bypass" in hits[0].reason
    resolved = baseline(
        ev(2, "Utterance", text="I need the toilet"),
        ev(3, "GoalChanged", from_goal="return_to_bed", to_goal="restroom", reason="x"),
        ev(
            5,
            "GoalChanged",
            from_goal="restroom",
            to_goal="return_to_bed",
            reason="returned_from_bathroom",
        ),
        ev(6, "Say", text="Let's go back to bed.", strategy="guided_return"),
    )
    assert not [r for r in score(resolved, "SM-5") if not r.passed]
    # "Can I go back to bed?" while the restroom goal is active also resolves the need.
    wants_bed = baseline(
        ev(2, "Utterance", text="Okay, I'm back from the restroom."),
        ev(3, "GoalChanged", from_goal="return_to_bed", to_goal="restroom", reason="x"),
        ev(
            6,
            "GoalChanged",
            from_goal="restroom",
            to_goal="return_to_bed",
            reason="interpreted_wants_bed",
        ),
        ev(6, "Say", text="Let's go back to bed now.", strategy="guided_return"),
    )
    assert not [r for r in score(wants_bed, "SM-5") if not r.passed]


def test_sm5_flags_path_light_said_to_person_on_floor():
    trace = baseline(
        ev(1, "PersonState", state="on_floor", zone="other"),
        ev(10, "Say", text="The restroom is to the left.", strategy="path_light"),
    )
    fail(trace, "SM-5", "critical", "no_directions_from_floor")


def test_tm1_tm3():
    good = baseline(ev(2, "Utterance", text="hi"), ev(3, "Say", text="hello", strategy="greet"))
    # A slow reply with no LLM call running is decision timing, not loop lag.
    slow_no_llm = baseline(
        ev(2, "Utterance", text="hi"), ev(9, "Say", text="hello", strategy="greet")
    )
    assert not [r for r in score(slow_no_llm, "TM-1") if not r.passed]
    short_call = baseline(
        ev(1, "Activity", kind="compose", phase="start"),
        ev(2, "Utterance", text="hi"),
        ev(3, "Activity", kind="compose", phase="end", duration_ms=2000),
        ev(3, "Say", text="hello", strategy="greet"),
    )
    passes(short_call, "TM-1")
    # interpret then compose back to back: an utterance at 2 s waits until 7 s.
    chained = baseline(
        ev(1, "Activity", kind="interpret", phase="start"),
        ev(2, "Utterance", text="hi"),
        ev(4, "Activity", kind="interpret", phase="end", duration_ms=3000),
        ev(4, "Activity", kind="compose", phase="start"),
        ev(7, "Activity", kind="compose", phase="end", duration_ms=3000),
        ev(7, "Say", text="hello", strategy="greet"),
    )
    fail(chained, "TM-1", "minor", "loop lag")
    hit = [r for r in score(chained, "TM-1") if not r.passed][0]
    assert any("compose" in text and "interpret" in text for text in hit.evidence)
    assert hit.window == (2, 7)
    no_path = score(good, "TM-3")[0]
    assert no_path.passed and no_path.context["latency_s"] is None
    path = baseline(
        ev(3, "PersonState", state="walking", zone="bathroom_path"),
        ev(5, "GoalChanged", to_goal="restroom"),
    )
    assert score(path, "TM-3")[0].context["latency_s"] == 2


def test_adapters_and_playback(tmp_path):
    def entry(t, kind, data):
        return SimpleNamespace(t=t, kind=kind, data=data)

    db = SimpleNamespace(
        scenario_id="x",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        entries=[
            entry(0, "Utterance", {"text": "hi"}),
            entry(
                1,
                "Activity",
                {"kind": "decision", "detail": json.dumps({"decision": "vetoed", "rule": "r"})},
            ),
            entry(2, "AudioChunk", {"pcm16": "secret"}),
        ],
        end_t=10,
    )
    trace = from_decision_bench(db)
    assert [e.kind for e in trace.events] == ["input", "decision"]
    assert trace.events[1].data["rule"] == "r"
    path = tmp_path / "trace.jsonl"
    trace.write_jsonl(path)
    assert Trace.read_jsonl(path).events == trace.events
    sr = from_session_replay(
        [
            {
                "t": 0,
                "ts": "2026-01-01T00:00:00Z",
                "type": "IN",
                "person": {"state": "walking", "zone": "door"},
            },
            {"t": 1, "type": "IN", "heard": "hello"},
            {"t": 2, "type": "Say", "text": "hi", "strategy": "greet"},
        ],
        "s",
    )
    assert [e.type for e in sr.events] == ["PersonState", "Utterance", "Say"]
    export = from_export(
        [
            {
                "recorded_at": 100,
                "ts": "2026-01-01T00:00:00Z",
                "event_type": "Say",
                "payload": {"text": "hello", "strategy": "greet"},
            },
            {"recorded_at": 102, "event_type": "Frame", "payload": {"jpeg": "secret"}},
        ]
    )
    assert len(export.events) == 1 and export.events[0].t == 0
    logs = from_agent_log(
        [
            json.dumps(
                {
                    "service": "agent",
                    "ts": "2026-01-01T00:00:00Z",
                    "message": "vetoed say",
                    "rule": "r",
                }
            )
        ]
    )
    assert logs.events[0].kind == "decision"
    unstamped = from_agent_log(
        [
            {"service": "agent", "message": "dropped deferred Say", "reason": "max_age"},
            {"service": "agent", "message": "published Say", "event_type": "Say"},
        ]
    )
    assert [e.kind for e in unstamped.events] == ["decision", "output"]
    assert unstamped.events[0].data["reason"] == "max_age"
    estimated = playbacks(scene(ev(0, "Say", text="hello world", strategy="greet")), load())[0]
    assert estimated.estimated and estimated.end == 1
    real = playbacks(
        scene(
            ev(0, "Say", text="hello", strategy="greet"),
            ev(0.2, "Activity", kind="playback", detail="playing"),
            ev(1, "Activity", kind="playback", phase="end", detail="barge-in"),
        ),
        load(),
    )[0]
    assert not real.estimated and real.interrupted and real.start == 0.2 and real.end == 1


def test_run_artifacts_and_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENE_LAB_RUNS", str(tmp_path))
    first = RunDir("check")
    second = RunDir("check")
    assert second.id.endswith("-2")
    result = InvariantResult(
        id="TT-1",
        severity="critical",
        passed=False,
        window=(1, 2),
        evidence=["hello"],
        reason="no reply",
        context={"phase": "ENGAGED", "goal": "rest", "strategy": "greet"},
        t=1,
    )
    assert fingerprint(result) == "TT-1|ENGAGED|rest|greet|-"
    first.write_scene(
        "one", scene(ev(1, "Utterance", text="hello")), [result], {"commit": "abc", "model": "m"}
    )
    entries = results_to_entries([result, result], first.id, "one", "abc", "m")
    first.append(entries)
    assert len((first.path / "bugs.jsonl").read_text().splitlines()) == 2
    assert "(2)" in (first.path / "bugs.md").read_text()
    assert (first.path / "bugs.md").read_text().find("TT-1") < 999
    first.finish("check", "abc", "m", 1, 1)
    assert json.loads((tmp_path / "index.jsonl").read_text())["bug_counts"]["critical"] == 2
    other = InvariantResult(
        id="SM-1", severity="minor", passed=False, window=(2, 3), evidence=[], reason="later", t=2
    )
    second.append(results_to_entries([other], second.id, "two", "abc", "m"))
    rows = merge([second.path, first.path])
    assert {r["status"] for r in rows} == {"gone", "new"}
    assert "gone" in render_merge_md(rows)
    second.append(results_to_entries([result], second.id, "two", "abc", "m"))
    assert {r["status"] for r in merge([first.path, second.path])} == {"persisting", "new"}
    assert harness_error(first.id, "one", "stack crashed").origin == "harness"


def test_bug_sorting_by_severity(tmp_path):
    run = RunDir("check", root=tmp_path)
    minor = InvariantResult(
        id="TT-5",
        severity="minor",
        passed=False,
        window=(1, 1),
        evidence=[],
        reason="short gap",
        t=1,
    )
    critical = InvariantResult(
        id="TT-1",
        severity="critical",
        passed=False,
        window=(2, 2),
        evidence=[],
        reason="no reply",
        t=2,
    )
    run.append(results_to_entries([minor, critical], run.id, "one", "abc", "m"))
    report = (run.path / "bugs.md").read_text()
    assert report.index("## TT-1") < report.index("## TT-5")
