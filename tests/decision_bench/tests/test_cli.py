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
