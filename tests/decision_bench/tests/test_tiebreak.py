import threading
from pathlib import Path

import yaml

from decision_bench.annotate import (
    annotate_scenario,
    build_arg_parser,
    citable_clauses,
    write_annotation_files,
)
from decision_bench.schema import GUIDELINES_PATH, load_default_profile, load_scenario
from decision_bench.tiebreak import tiebreak_main

FIXTURE = Path(__file__).parents[1] / "fixtures/scenarios/false-alarm-01.yaml"


def _run(scenario, **checkpoint_overrides):
    checkpoints = []
    for item in scenario.checkpoints:
        checkpoint = {
            "id": item.id,
            "acceptable": [{"phase": "OBSERVING"}],
            "must_not": [{"say": "any"}],
            "escalate_by": None,
            "trigger": None,
            "rationale": "Watch quietly while the person settles.",
            "cites": ["NICE-05"],
            "uncertain": "",
        }
        checkpoint.update(checkpoint_overrides)
        checkpoints.append(checkpoint)
    output = {"checkpoints": checkpoints, "scenario_notes": ""}
    return annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: {
            "structured_output": output,
            "model": "claude-opus-5",
            "session_id": "test",
            "total_cost_usd": 0.01,
        },
    )


def _setup(tmp_path, *, first_overrides, second_overrides, scenario_id="false-alarm-01"):
    scenarios_dir = tmp_path / "scenarios"
    annotations = tmp_path / "annotations"
    scenarios_dir.mkdir()
    fixture = scenarios_dir / FIXTURE.name
    fixture.write_bytes(FIXTURE.read_bytes())
    scenario = load_scenario(fixture)
    first = _run(scenario, **first_overrides)
    second = _run(scenario, **second_overrides)
    write_annotation_files(
        scenario,
        first,
        annotations_dir=annotations,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
        second=second,
    )
    return scenario, scenarios_dir, annotations, fixture


def _args(scenarios_dir, annotations, *, jobs=1, dry_run=False, scenario_ids=None, force=False):
    args = build_arg_parser().parse_args(["--tiebreak"])
    args.scenarios = scenarios_dir
    args.annotations = annotations
    args.jobs = jobs
    args.dry_run = dry_run
    args.scenario_ids = scenario_ids
    args.force = force
    return args


def test_agreeing_scenario_applies_without_any_opus_call(tmp_path):
    # Not yet applied: e.g. `annotate --no-apply` left `reviewed: false` even
    # though the two runs agree on content.
    scenario, scenarios_dir, annotations, fixture = _setup(
        tmp_path, first_overrides={}, second_overrides={}
    )
    review_path = annotations / "review" / f"{scenario.id}.yaml"
    review_doc = yaml.safe_load(review_path.read_text())
    review_doc["reviewed"] = False
    review_path.write_text(yaml.safe_dump(review_doc, sort_keys=False))

    def runner(*a):
        raise AssertionError("agree -> apply must not call the annotator")

    result = tiebreak_main(_args(scenarios_dir, annotations), runner=runner)
    assert result == 0
    loaded = load_scenario(fixture)
    assert loaded.labelled
    assert loaded.checkpoints[0].acceptable[0].value == "OBSERVING"


def test_disagreeing_scenario_runs_a_third_opinion_and_votes(tmp_path, capsys):
    scenario, scenarios_dir, annotations, fixture = _setup(
        tmp_path,
        first_overrides={"acceptable": [{"strategy": "path_light"}]},
        second_overrides={"acceptable": []},
    )

    def runner(system_prompt_path, prompt, schema, model, effort):
        output = {
            "checkpoints": [
                {
                    "id": scenario.checkpoints[0].id,
                    "acceptable": [{"strategy": "path_light"}],
                    "must_not": [{"say": "any"}],
                    "escalate_by": None,
                    "trigger": None,
                    "rationale": "Third run agrees with the first.",
                    "cites": ["NICE-05"],
                    "uncertain": "",
                }
            ],
            "scenario_notes": "",
        }
        return {
            "structured_output": output,
            "model": "claude-opus-5",
            "session_id": "third",
            "total_cost_usd": 0.01,
        }

    result = tiebreak_main(_args(scenarios_dir, annotations), runner=runner)
    assert result == 0
    printed = capsys.readouterr().out
    assert "3rd run, voted -> applied (model)" in printed
    loaded = load_scenario(fixture)
    assert loaded.checkpoints[0].acceptable[0].value == "path_light"
    model_doc = yaml.safe_load((annotations / "model" / f"{scenario.id}.yaml").read_text())
    assert model_doc["third_opinion"]["checkpoints"][0]["acceptable"] == [
        {"strategy": "path_light"}
    ]


def test_no_majority_leaves_a_review_file_and_does_not_touch_the_fixture(tmp_path, capsys):
    scenario, scenarios_dir, annotations, fixture = _setup(
        tmp_path,
        first_overrides={"acceptable": [{"strategy": "path_light"}]},
        second_overrides={"acceptable": [{"strategy": "soft_greeting"}]},
    )
    before = fixture.read_bytes()

    def runner(system_prompt_path, prompt, schema, model, effort):
        output = {
            "checkpoints": [
                {
                    "id": scenario.checkpoints[0].id,
                    "acceptable": [{"strategy": "ambient_orient"}],
                    "must_not": [{"say": "any"}],
                    "escalate_by": None,
                    "trigger": None,
                    "rationale": "A third, different proposal.",
                    "cites": ["NICE-05"],
                    "uncertain": "",
                }
            ],
            "scenario_notes": "",
        }
        return {
            "structured_output": output,
            "model": "claude-opus-5",
            "session_id": "third",
            "total_cost_usd": 0.01,
        }

    result = tiebreak_main(_args(scenarios_dir, annotations), runner=runner)
    assert result == 0
    printed = capsys.readouterr().out
    assert "no majority on" in printed
    assert str(annotations / "review" / f"{scenario.id}.yaml") in printed
    assert fixture.read_bytes() == before
    review = yaml.safe_load((annotations / "review" / f"{scenario.id}.yaml").read_text())
    assert review["reviewed"] is False
    assert review["reviewed_by"] == "human"


def test_writes_stay_serial_under_concurrent_opus_calls(tmp_path):
    # Two scenarios that both need a third opinion, run with --jobs 2. A barrier
    # forces the two fake Opus calls to genuinely overlap in time; if the file
    # writes that follow were not serial (or mixed up which scenario they
    # belonged to), one scenario's fixture would end up with the other's answer.
    scenarios_dir = tmp_path / "scenarios"
    annotations = tmp_path / "annotations"
    scenarios_dir.mkdir()
    fixtures = {
        "false-alarm-01": Path(__file__).parents[1] / "fixtures/scenarios/false-alarm-01.yaml",
        "false-alarm-02": Path(__file__).parents[1] / "fixtures/scenarios/false-alarm-02.yaml",
    }
    strategy_by_scenario = {"false-alarm-01": "path_light", "false-alarm-02": "soft_greeting"}
    scenarios = {}
    for scenario_id, source in fixtures.items():
        fixture = scenarios_dir / source.name
        fixture.write_bytes(source.read_bytes())
        scenario = load_scenario(fixture)
        scenarios[scenario_id] = scenario
        first = _run(scenario, acceptable=[{"strategy": strategy_by_scenario[scenario_id]}])
        second = _run(scenario, acceptable=[])
        write_annotation_files(
            scenario,
            first,
            annotations_dir=annotations,
            effort="high",
            guidelines_text=GUIDELINES_PATH.read_text(),
            force=False,
            second=second,
        )

    barrier = threading.Barrier(2, timeout=5)

    def runner(system_prompt_path, prompt, schema, model, effort):
        barrier.wait()  # both threads must be inside the "Opus call" at once
        scenario_id = next(sid for sid in scenarios if sid in prompt)
        checkpoint_id = scenarios[scenario_id].checkpoints[0].id
        output = {
            "checkpoints": [
                {
                    "id": checkpoint_id,
                    "acceptable": [{"strategy": strategy_by_scenario[scenario_id]}],
                    "must_not": [{"say": "any"}],
                    "escalate_by": None,
                    "trigger": None,
                    "rationale": "Third run.",
                    "cites": ["NICE-05"],
                    "uncertain": "",
                }
            ],
            "scenario_notes": "",
        }
        return {
            "structured_output": output,
            "model": "claude-opus-5",
            "session_id": "third",
            "total_cost_usd": 0.01,
        }

    result = tiebreak_main(_args(scenarios_dir, annotations, jobs=2), runner=runner)
    assert result == 0
    for scenario_id, expected in strategy_by_scenario.items():
        fixture = scenarios_dir / fixtures[scenario_id].name
        loaded = load_scenario(fixture)
        assert loaded.checkpoints[0].acceptable[0].value == expected
