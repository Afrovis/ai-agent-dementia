import json

from scene_lab.bugs import RunDir, harness_error
from scene_lab.rescore import rescore
from scene_lab.thresholds import load
from scene_lab.trace import Trace, TraceEvent


def test_rescore_rechecks_traces_and_keeps_harness_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENE_LAB_RUNS", str(tmp_path))
    run = RunDir("live", root=tmp_path)
    trace = Trace(
        id="scene-a",
        source="live",
        events=[
            TraceEvent(t=0, kind="input", type="PersonState", data={"state": "walking"}),
            TraceEvent(
                t=0,
                kind="output",
                type="SessionState",
                data={"phase": "ENGAGED", "goal": "return_to_bed", "strategy_index": 0},
            ),
            TraceEvent(t=2, kind="input", type="Utterance", data={"text": "Where am I?"}),
        ],
        end_t=40,
    )
    run.write_scene("scene-a", trace, [], {"commit": "old", "model": "m"})
    (run.path / "scene-a" / "scene.yaml").write_text("id: scene-a\n")
    run.append([harness_error(run.id, "scene-a", "mind failure: example")])

    target = rescore(run.path)

    assert target.name == run.path.name + "-rescored"
    rows = [json.loads(line) for line in (target / "bugs.jsonl").read_text().splitlines()]
    assert any(r["check"] == "TT-1" and r["severity"] == "critical" for r in rows)
    assert any(r["origin"] == "harness" and r["run"] == target.name for r in rows)
    report = json.loads((target / "scene-a" / "report.json").read_text())
    assert report["rescored_from"] == "old"
    assert report["thresholds"] == load().model_dump()
    # The original run is untouched.
    assert len((run.path / "bugs.jsonl").read_text().splitlines()) == 1
