import json

import yaml

from decision_bench.calibrate import calibrate, main


def _checkpoint(acceptable):
    return {
        "id": "checkpoint",
        "acceptable": [{"phase": acceptable}],
        "must_not": [{"say": "any"}],
        "escalate_by": None,
        "trigger": None,
        "needs_review": False,
    }


def _write_pair(local_dir, model_dir, scenario, first, second, local, *, runtime=1.5):
    model = {
        "scenario": scenario,
        "checkpoints": [_checkpoint(first)],
        "second_opinion": {"checkpoints": [_checkpoint(second)]},
    }
    local_doc = {
        "scenario": scenario,
        "annotator": {
            "runtime_seconds": runtime,
            "parse_validation_failures": ["first response was invalid"]
            if scenario == "agree"
            else [],
        },
        "checkpoints": [_checkpoint(local)],
    }
    (model_dir / f"{scenario}.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    (local_dir / f"{scenario}.yaml").write_text(yaml.safe_dump(local_doc), encoding="utf-8")


def test_calibrate_agree_disagree_and_harmful_cases(tmp_path):
    local_dir = tmp_path / "local"
    model_dir = tmp_path / "model"
    local_dir.mkdir()
    model_dir.mkdir()
    _write_pair(local_dir, model_dir, "agree", "OBSERVING", "OBSERVING", "OBSERVING")
    _write_pair(local_dir, model_dir, "disagree", "OBSERVING", "ENGAGED", "ENGAGED")
    _write_pair(local_dir, model_dir, "harmful", "OBSERVING", "ENGAGED", "OBSERVING")

    report = calibrate(local_dir, model_dir)
    by_id = {item["scenario"]: item for item in report["scenarios"]}
    assert by_id["agree"]["opus_agree"] is True
    assert by_id["agree"]["local_vs_opus1_agree"] is True
    assert by_id["disagree"]["opus_agree"] is False
    assert by_id["disagree"]["local_vs_opus1_agree"] is False
    assert by_id["harmful"]["harmful_local_agreement"] is True
    assert report["totals"]["harmful_local_agreements"] == ["harmful"]
    assert report["totals"]["scenarios"] == 3
    assert report["totals"]["local_vs_opus1_agree"]["count"] == 2
    assert report["totals"]["parse_validation_failures"] == 1


def test_calibrate_cli_writes_full_json(tmp_path, capsys):
    local_dir = tmp_path / "local"
    model_dir = tmp_path / "model"
    local_dir.mkdir()
    model_dir.mkdir()
    _write_pair(local_dir, model_dir, "agree", "OBSERVING", "OBSERVING", "OBSERVING")
    out = tmp_path / "report.json"

    assert (
        main(["--local-dir", str(local_dir), "--model-dir", str(model_dir), "--out", str(out)]) == 0
    )
    assert "Local + Opus 1 agreement" in capsys.readouterr().out
    assert json.loads(out.read_text())["totals"]["scenarios"] == 1
