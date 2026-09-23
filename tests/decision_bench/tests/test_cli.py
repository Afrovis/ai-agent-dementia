import json

from decision_bench.__main__ import main


def test_stub_json_smoke(capsys):
    assert main(["--backend", "stub", "--scenario", "fall-01", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["scenarios"][0]["id"] == "fall-01"


def test_trace_prints(capsys):
    assert main(["--backend", "stub", "--scenario", "fall-01", "--trace"]) == 0
    output = capsys.readouterr().out
    assert "TRACE fall-01" in output
    assert "Notify[critical]" in output


def test_annotate_dry_run_dispatches_without_claude(capsys):
    assert main(["annotate", "--dry-run", "--scenario", "fall-01"]) == 0
    output = capsys.readouterr().out
    assert "annotator_prompt.md" in output
    assert "on-floor" in output


def test_stub_invariants_write_trace_and_reports(tmp_path, capsys):
    assert (
        main(
            [
                "--backend",
                "stub",
                "--scenario",
                "fall-01",
                "--invariants",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "Invariants run:" in capsys.readouterr().out
    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert (run / "bugs.jsonl").exists()
    assert (run / "bugs.md").exists()
    scene = run / "fall-01"
    assert (scene / "report.json").exists()
    events = [json.loads(line) for line in (scene / "trace.jsonl").read_text().splitlines()]
    assert any(event["kind"] == "input" for event in events)
    assert any(event["kind"] == "output" for event in events)


def test_decision_activity_survives_runner_trace():
    from datetime import datetime

    from nc_shared.events import Activity

    from decision_bench.runner import Trace, _TraceBus

    trace = Trace("test", datetime(2026, 1, 1))
    bus = _TraceBus(trace)
    bus.publish(
        Activity(
            source="agent",
            service="agent",
            kind="decision",
            phase="end",
            detail='{"decision":"vetoed","rule":"test"}',
        )
    )
    assert trace.entries[0].kind == "Activity"
    from scene_lab.trace import from_decision_bench

    assert from_decision_bench(trace).events[0].kind == "decision"


def test_latency_in_invariant_report(tmp_path, capsys):
    assert (
        main(
            [
                "--backend",
                "stub",
                "--scenario",
                "conversation-01",
                "--llm-latency",
                "fixed:2.5",
                "--invariants",
                "--runs-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "LLM latency: fixed:2.5" in capsys.readouterr().out
    run = next(path for path in tmp_path.iterdir() if path.is_dir())
    report = json.loads((run / "conversation-01" / "report.json").read_text())
    assert report["llm_latency"] == "fixed:2.5"
