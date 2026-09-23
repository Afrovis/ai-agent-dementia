"""Offline run persistence and summary behavior."""

import json

from scene_lab.offline import InvariantRun
from scene_lab.trace import Trace, TraceEvent


def test_invariant_run_writes_reports_and_summary(tmp_path):
    run = InvariantRun("session_replay", "stub", root=tmp_path)
    trace = Trace(
        id="sample",
        source="session_replay",
        events=[
            TraceEvent(
                t=0, kind="input", type="PersonState", data={"state": "standing", "zone": "bed"}
            ),
            TraceEvent(t=0, kind="output", type="SessionState", data={"phase": "ENGAGED"}),
            TraceEvent(t=1, kind="input", type="Utterance", data={"text": "Can you help?"}),
        ],
        end_t=7,
    )
    run.add("sample", trace)
    summary, text = run.close()
    assert summary["scene_count"] == 1
    assert summary["checks"]["TT-1"]["critical"] == 1
    assert "TT-1: evaluated=" in text
    assert (run.run.path / "bugs.jsonl").exists()
    assert (run.run.path / "bugs.md").exists()
    assert (run.run.path / "sample" / "trace.jsonl").exists()
    report = json.loads((run.run.path / "sample" / "report.json").read_text())
    assert report["model"] == "stub"
