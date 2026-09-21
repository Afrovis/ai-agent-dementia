from pathlib import Path

import pytest
import yaml

from decision_bench.annotate import (
    citable_clauses,
    validate_annotation,
    write_annotation_files,
)
from decision_bench.review import ReviewError, apply_scenario
from decision_bench.schema import GUIDELINES_PATH, load_scenario

FIXTURE = Path(__file__).parents[1] / "fixtures/scenarios/false-alarm-01.yaml"


def _setup(tmp_path):
    scenarios = tmp_path / "scenarios"
    annotations = tmp_path / "annotations"
    scenarios.mkdir()
    fixture = scenarios / FIXTURE.name
    fixture.write_bytes(FIXTURE.read_bytes())
    scenario = load_scenario(fixture)
    output = {
        "scenario_notes": "",
        "checkpoints": [
            {
                "id": scenario.checkpoints[0].id,
                "acceptable": [{"phase": "OBSERVING"}],
                "must_not": [{"say": "any"}],
                "escalate_by": None,
                "trigger": None,
                "rationale": "Let the person settle without waking them further.",
                "cites": ["NICE-05"],
                "uncertain": "Perhaps IDLE is also reasonable.",
            }
        ],
    }
    labelled, problems = validate_annotation(scenario, output, citable_clauses())
    assert not problems
    result = {
        "structured_output": output,
        "model": "claude-opus-5",
        "session_id": "test",
        "total_cost_usd": 0.01,
        "attempts": 1,
        "checkpoints": labelled,
    }
    _, review_path = write_annotation_files(
        scenario,
        result,
        annotations_dir=annotations,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
    )
    return scenarios, annotations, fixture, review_path


def _review(path, mutate=None):
    raw = yaml.safe_load(path.read_text())
    raw["reviewed"] = True
    if mutate:
        mutate(raw["checkpoints"][0])
    path.write_text(yaml.safe_dump(raw, sort_keys=False))


def test_unreviewed_refuses(tmp_path):
    scenarios, annotations, fixture, _ = _setup(tmp_path)
    before = fixture.read_bytes()
    with pytest.raises(ReviewError, match="not marked reviewed"):
        apply_scenario("false-alarm-01", annotations_dir=annotations, scenarios_dir=scenarios)
    assert fixture.read_bytes() == before


def test_changed_label_without_reason_leaves_fixture_untouched(tmp_path):
    scenarios, annotations, fixture, review_path = _setup(tmp_path)
    before = fixture.read_bytes()
    _review(review_path, lambda item: item.update(acceptable=[{"phase": "IDLE"}]))
    with pytest.raises(ReviewError, match="need a reason"):
        apply_scenario("false-alarm-01", annotations_dir=annotations, scenarios_dir=scenarios)
    assert fixture.read_bytes() == before


def test_accept_as_is_preserves_header_and_logs_nothing(tmp_path):
    scenarios, annotations, fixture, review_path = _setup(tmp_path)
    header = fixture.read_bytes().split(b"checkpoints:", 1)[0]
    _review(review_path)
    applied, disagreements = apply_scenario(
        "false-alarm-01", annotations_dir=annotations, scenarios_dir=scenarios
    )
    assert (applied, disagreements) == (1, 0)
    assert fixture.read_bytes().split(b"checkpoints:", 1)[0] == header
    assert load_scenario(fixture).labelled
    assert yaml.safe_load((annotations / "disagreements.yaml").read_text()) == []


def test_edit_with_reason_logs_once_on_reapply(tmp_path):
    scenarios, annotations, fixture, review_path = _setup(tmp_path)

    def mutate(item):
        item["acceptable"] = [{"phase": "IDLE"}]
        item["reason"] = "Remaining quiet in IDLE is the intended human label."

    _review(review_path, mutate)
    assert apply_scenario(
        "false-alarm-01", annotations_dir=annotations, scenarios_dir=scenarios
    ) == (1, 1)
    assert apply_scenario(
        "false-alarm-01", annotations_dir=annotations, scenarios_dir=scenarios
    ) == (1, 1)
    logged = yaml.safe_load((annotations / "disagreements.yaml").read_text())
    assert len(logged) == 1
    assert logged[0]["checkpoint"] == "settles"
    assert logged[0]["human"]["acceptable"] == [{"phase": "IDLE"}]
    assert load_scenario(fixture).labelled


def test_model_accepted_review_cannot_carry_edits(tmp_path):
    scenarios, annotations, fixture, review_path = _setup(tmp_path)
    before = fixture.read_bytes()

    def mutate(item):
        item.update(acceptable=[{"phase": "IDLE"}], reason="changed")

    _review(review_path, mutate)
    raw = yaml.safe_load(review_path.read_text())
    raw["reviewed_by"] = "model"
    review_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ReviewError, match="reviewed_by: human"):
        apply_scenario("false-alarm-01", annotations_dir=annotations, scenarios_dir=scenarios)
    assert fixture.read_bytes() == before
