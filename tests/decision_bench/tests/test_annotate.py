from pathlib import Path

import pytest
import yaml

from decision_bench.annotate import (
    ANNOTATOR_PROMPT_PATH,
    AnnotatorError,
    annotate_scenario,
    annotator_input,
    citable_clauses,
    output_schema,
    validate_annotation,
    write_annotation_files,
)
from decision_bench.annotate import (
    main as annotate_main,
)
from decision_bench.schema import (
    ACTION_VALUES,
    GUIDELINES_PATH,
    load_default_profile,
    load_scenario,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/scenarios/false-alarm-01.yaml"


def _output(scenario, **checkpoint_overrides):
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
    return {"checkpoints": checkpoints, "scenario_notes": ""}


def _fake_result(output, *, model="claude-opus-5"):
    return {
        "structured_output": output,
        "model": model,
        "session_id": "session-1",
        "total_cost_usd": 0.42,
    }


def test_prompt_is_sanitised_and_contains_questions():
    scenario = load_scenario(FIXTURE)
    labelled = _output(scenario)["checkpoints"][0]
    labelled["rationale"] = "SECRET EXISTING RATIONALE"
    checkpoint = scenario.checkpoints[0].model_copy(
        update={
            "acceptable": (),
            "must_not": scenario.checkpoints[0].must_not,
            "rationale": labelled["rationale"],
        }
    )
    # Construct a valid labelled source to prove labels are stripped from the prompt.
    checkpoint = checkpoint.model_validate(
        {
            "id": checkpoint.id,
            "question": scenario.checkpoints[0].question,
            "window": checkpoint.window,
            "must_not": [{"say": "any"}],
            "rationale": labelled["rationale"],
            "cites": ["NICE-05"],
        }
    )
    scenario = scenario.model_copy(update={"checkpoints": (checkpoint,)})
    prompt = annotator_input(
        scenario,
        load_default_profile() | scenario.profile,
        GUIDELINES_PATH.read_text(),
        citable_clauses(),
    )
    assert scenario.checkpoints[0].question in prompt
    assert scenario.checkpoints[0].id in prompt
    assert "NICE-05" in prompt
    assert "SECRET EXISTING RATIONALE" not in prompt
    assert "dwell_seconds" not in prompt
    assert "intrusiveness" not in prompt
    assert "order:" not in prompt


def test_schema_covers_actions_and_citable_ids():
    schema = output_schema(load_scenario(FIXTURE), citable_clauses())
    checkpoint = schema["properties"]["checkpoints"]["items"]
    action = checkpoint["properties"]["acceptable"]["items"]
    covered = {
        kind: set(choice["properties"][kind]["enum"])
        for choice in action["oneOf"]
        for kind in choice["properties"]
    }
    assert covered == {kind: set(values) for kind, values in ACTION_VALUES.items()}
    assert checkpoint["properties"]["cites"]["items"]["enum"] == citable_clauses()


def test_valid_output_writes_model_and_review(tmp_path):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario)

    def runner(*args):
        return _fake_result(output)

    result = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=runner,
    )
    model_path, review_path = write_annotation_files(
        scenario,
        result,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
    )
    model = yaml.safe_load(model_path.read_text())
    review = yaml.safe_load(review_path.read_text())
    assert model["annotator"]["model"] == "claude-opus-5"
    assert model["annotator"]["attempts"] == 1
    assert model["checkpoints"][0]["acceptable"] == [{"phase": "OBSERVING"}]
    assert review["reviewed"] is False
    assert review["checkpoints"][0]["reason"] == ""
    assert "uncertain" not in review["checkpoints"][0]


@pytest.mark.parametrize(
    "override,problem",
    [
        ({"id": "unknown"}, "unknown checkpoint"),
        ({"cites": []}, "cite"),
        ({"cites": ["NICE-06"]}, "not citable"),
        ({"acceptable": [{"strategy": "familiar_voice"}]}, "voice_clip"),
    ],
)
def test_invalid_output_retries_then_raises(override, problem):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, **override)
    prompts = []

    def runner(system, prompt, schema, model, effort):
        prompts.append(prompt)
        return _fake_result(output)

    with pytest.raises(AnnotatorError, match="remained invalid"):
        annotate_scenario(
            scenario,
            profile=load_default_profile(),
            guidelines_text=GUIDELINES_PATH.read_text(),
            citable=citable_clauses(),
            runner=runner,
        )
    assert len(prompts) == 2
    assert "Your previous answer was rejected" in prompts[1]
    assert problem in prompts[1]


def test_validate_rejects_trigger_without_deadline():
    scenario = load_scenario(FIXTURE)
    _, problems = validate_annotation(scenario, _output(scenario, trigger=10), citable_clauses())
    assert any("trigger requires" in problem for problem in problems)
    assert ANNOTATOR_PROMPT_PATH.exists()


def test_review_file_holds_scenario_questions_and_annotator(tmp_path):
    scenario = load_scenario(FIXTURE)
    result = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: _fake_result(_output(scenario)),
    )
    _, review_path = write_annotation_files(
        scenario,
        result,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
    )
    text = review_path.read_text()
    assert "TIMELINE" in text
    assert "Claude, claude-opus-5" in text
    assert "LOCAL MODEL" in text
    for checkpoint in scenario.checkpoints:
        assert f"CHECKPOINT {checkpoint.id}" in text
    for event in scenario.timeline:
        if event.utterance is not None:
            assert event.utterance.text in text


def test_review_only_rebuilds_from_model_file(tmp_path):
    scenario = load_scenario(FIXTURE)
    result = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: _fake_result(_output(scenario)),
    )
    _, review_path = write_annotation_files(
        scenario,
        result,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
    )
    before = yaml.safe_load(review_path.read_text())
    review_path.unlink()
    annotate_main(["--review-only", "--scenario", scenario.id, "--annotations", str(tmp_path)])
    assert yaml.safe_load(review_path.read_text()) == before


def test_second_opinion_that_agrees_is_accepted_by_triage(tmp_path):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario)
    for item in output["checkpoints"]:
        item["needs_review"] = False
    result = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: _fake_result(output),
    )
    _, review_path = write_annotation_files(
        scenario,
        result,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
        second=result,
    )
    review = yaml.safe_load(review_path.read_text())
    assert review["reviewed"] is True
    assert review["reviewed_by"] == "model"
    assert "NEEDS YOUR REVIEW" not in review_path.read_text()


def test_self_flags_alone_no_longer_block_application(tmp_path):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario)
    for item in output["checkpoints"]:
        item["needs_review"] = True
    result = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: _fake_result(output),
    )
    _, review_path = write_annotation_files(
        scenario,
        result,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
        second=result,
    )
    text = review_path.read_text()
    assert yaml.safe_load(text)["reviewed"] is True
    assert yaml.safe_load(text)["reviewed_by"] == "model"
    assert "NEEDS YOUR REVIEW" not in text


def test_flagged_checkpoint_is_marked_for_the_human(tmp_path):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario)
    second_output = _output(scenario, must_not=[])
    result = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: _fake_result(output),
    )
    second = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: _fake_result(second_output),
    )
    _, review_path = write_annotation_files(
        scenario,
        result,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
        second=second,
    )
    text = review_path.read_text()
    assert yaml.safe_load(text)["reviewed"] is False
    assert "NEEDS YOUR REVIEW" in text and "SECOND OPINION" in text


def _run(scenario, output):
    return annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=lambda *args: _fake_result(output),
    )


def test_third_opinion_that_reaches_a_majority_is_applied_with_doubtful_labels(tmp_path):
    scenario = load_scenario(FIXTURE)
    first = _run(scenario, _output(scenario, must_not=[{"say": "any"}]))
    second = _run(scenario, _output(scenario, must_not=[]))
    third = _run(scenario, _output(scenario, must_not=[{"say": "any"}]))
    _, review_path = write_annotation_files(
        scenario,
        first,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
        second=second,
        third=third,
    )
    model = yaml.safe_load((tmp_path / "model" / f"{scenario.id}.yaml").read_text())
    assert model["third_opinion"]["checkpoints"][0]["must_not"] == [{"say": "any"}]
    review = yaml.safe_load(review_path.read_text())
    assert review["reviewed"] is True
    assert review["reviewed_by"] == "model"
    assert review["checkpoints"][0]["must_not"] == [{"say": "any"}]


def test_third_opinion_without_a_majority_needs_a_human_and_shows_all_three_runs(tmp_path):
    scenario = load_scenario(FIXTURE)
    first = _run(scenario, _output(scenario, acceptable=[{"strategy": "path_light"}]))
    second = _run(scenario, _output(scenario, acceptable=[{"strategy": "soft_greeting"}]))
    third = _run(scenario, _output(scenario, acceptable=[{"strategy": "ambient_orient"}]))
    _, review_path = write_annotation_files(
        scenario,
        first,
        annotations_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
        force=False,
        second=second,
        third=third,
    )
    text = review_path.read_text()
    review = yaml.safe_load(text)
    assert review["reviewed"] is False
    assert review["reviewed_by"] == "human"
    assert "NEEDS YOUR REVIEW" in text
    assert "RUN 1" in text and "RUN 2" in text and "RUN 3" in text
    assert "no majority on acceptable" in text
