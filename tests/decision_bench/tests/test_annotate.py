import json
from pathlib import Path

import pytest
import yaml

from decision_bench.annotate import (
    ANNOTATOR_PROMPT_PATH,
    AnnotatorError,
    annotate_scenario,
    annotator_input,
    citable_clauses,
    normalize_ollama_output,
    output_schema,
    run_ollama,
    validate_annotation,
    write_annotation_files,
    write_local_annotation,
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


def test_run_ollama_sends_json_mode_schema_prompt_and_deterministic_options(monkeypatch):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": json.dumps(output)}}).encode()

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    schema = output_schema(scenario, citable_clauses())
    result = run_ollama(
        ANNOTATOR_PROMPT_PATH,
        "user prompt",
        schema,
        "qwen3.5:9b",
        "high",
    )

    payload = json.loads(requests[0][0].data)
    assert payload["format"] == "json"
    assert payload["options"] == {"temperature": 0, "num_ctx": 32768, "num_predict": 4096}
    assert "## Required output format" in payload["messages"][1]["content"]
    assert json.dumps(schema, indent=2, sort_keys=True) in payload["messages"][1]["content"]
    assert payload["think"] is True
    assert requests[0][1] == 300
    assert result["structured_output"] == output
    assert result["backend"] == "ollama"
    assert result["total_cost_usd"] == 0.0
    assert result["normalization_changed"] is False
    assert result["normalization_notes"] == []
    assert result["attempt_temperatures"] == [0]


@pytest.mark.parametrize(
    "wrapped",
    [
        "```json\n{output}\n```",
        "Here is the requested object:\n{output}\nEnd of response.",
    ],
)
def test_run_ollama_parses_fence_and_prose_wrapped_replies(monkeypatch, wrapped):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            content = wrapped.format(output=json.dumps(output))
            return json.dumps({"message": {"content": content}}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    result = run_ollama(
        ANNOTATOR_PROMPT_PATH,
        "user prompt",
        output_schema(scenario, citable_clauses()),
        "qwen3.5:9b",
        "high",
    )
    assert result["structured_output"] == output


@pytest.mark.parametrize(
    ("inner_value",),
    [
        (True,),
        (None,),
        ({},),
        (1,),
        ("ignored",),
    ],
)
def test_normalize_ollama_output_collapses_nested_action_regardless_of_inner_value(inner_value):
    action = {"say": {"correction_of_reality": inner_value}}
    original = {"checkpoints": [{"id": "repeats", "acceptable": [action], "must_not": [action]}]}
    normalized = normalize_ollama_output(original)
    assert normalized["checkpoints"][0]["acceptable"] == [{"say": "correction_of_reality"}]
    assert normalized["checkpoints"][0]["must_not"] == [{"say": "correction_of_reality"}]
    assert original == {
        "checkpoints": [{"id": "repeats", "acceptable": [action], "must_not": [action]}]
    }


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ({"strategy": {"name": "guided_return"}}, {"strategy": "guided_return"}),
        ({"say": {"name": "correction_of_reality"}}, {"say": "correction_of_reality"}),
        (
            {"strategy": {"name": "not_a_strategy"}},
            {"strategy": {"name": "not_a_strategy"}},
        ),
        (
            {"strategy": {"guided_return": "guided_return"}},
            {"strategy": "guided_return"},
        ),
    ],
)
def test_normalize_ollama_output_collapses_nested_action_value(action, expected):
    output = {"checkpoints": [{"id": "repeats", "acceptable": [action], "must_not": []}]}
    normalized = normalize_ollama_output(output)
    assert normalized["checkpoints"][0]["acceptable"] == [expected]


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ({"strategy": ["soft_greeting"]}, {"strategy": "soft_greeting"}),
        ("say correction_of_reality", {"say": "correction_of_reality"}),
        ("say: correction_of_reality", {"say": "correction_of_reality"}),
        ("bed", {"goal": "bed"}),
    ],
)
def test_normalize_ollama_output_repairs_other_known_action_shapes(action, expected):
    original = {"checkpoints": [{"id": "repeats", "acceptable": [action], "must_not": [action]}]}
    normalized = normalize_ollama_output(original)
    assert normalized["checkpoints"][0]["acceptable"] == [expected]
    assert normalized["checkpoints"][0]["must_not"] == [expected]


def test_normalize_ollama_output_leaves_unknown_nested_action_for_validation():
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)
    output["checkpoints"][0]["must_not"] = [{"say": {"not_a_real_value": {}}}]
    normalized = normalize_ollama_output(output)
    assert normalized["checkpoints"][0]["must_not"] == [{"say": {"not_a_real_value": {}}}]
    _, problems = validate_annotation(scenario, normalized, citable_clauses())
    assert problems


def test_normalize_ollama_output_fills_only_absent_scalar_fields():
    checkpoint = {
        "id": "repeats",
        "acceptable": [{"phase": "OBSERVING"}],
        "must_not": [{"say": "any"}],
    }
    normalized = normalize_ollama_output({"checkpoints": [checkpoint]})
    assert normalized["checkpoints"][0] == checkpoint | {
        "escalate_by": None,
        "trigger": None,
        "uncertain": "",
        "needs_review": False,
        "cites": [],
    }
    assert "rationale" not in normalized["checkpoints"][0]


def test_normalize_ollama_output_preserves_falsy_scalar_fields():
    checkpoint = {
        "id": "repeats",
        "acceptable": [{"phase": "OBSERVING"}],
        "must_not": [{"say": "any"}],
        "trigger": 0,
        "escalate_by": 0,
    }
    normalized = normalize_ollama_output({"checkpoints": [checkpoint]})
    assert normalized["checkpoints"][0] == checkpoint | {
        "uncertain": "",
        "needs_review": False,
        "cites": [],
    }


@pytest.mark.parametrize(
    "escalate_by", [pytest.param(None, id="null"), pytest.param(..., id="absent")]
)
@pytest.mark.parametrize("field", ["trigger", "threshold_source"])
def test_normalize_ollama_output_drops_threshold_without_escalate_by(escalate_by, field):
    checkpoint = {
        "id": "wants-home",
        "acceptable": [{"phase": "OBSERVING"}],
        "must_not": [{"say": "any"}],
        field: 5,
    }
    if escalate_by is not ...:
        checkpoint["escalate_by"] = escalate_by
    notes = set()

    normalized = normalize_ollama_output({"checkpoints": [checkpoint]}, notes=notes)

    assert normalized["checkpoints"][0][field] is None
    assert f"wants-home: dropped {field} without escalate_by" in notes


@pytest.mark.parametrize("escalate_by", [0, 180])
def test_normalize_ollama_output_keeps_thresholds_with_escalate_by(escalate_by):
    checkpoint = {
        "id": "wants-home",
        "acceptable": [{"phase": "OBSERVING"}],
        "must_not": [{"say": "any"}],
        "escalate_by": escalate_by,
        "trigger": 0,
        "threshold_source": "caregiver",
    }
    notes = set()

    normalized = normalize_ollama_output({"checkpoints": [checkpoint]}, notes=notes)

    assert normalized["checkpoints"][0]["trigger"] == 0
    assert normalized["checkpoints"][0]["threshold_source"] == "caregiver"
    assert not any("dropped" in note for note in notes)


def test_normalize_ollama_output_preserves_explicit_nulls():
    checkpoint = {
        "id": "repeats",
        "acceptable": [{"phase": "OBSERVING"}],
        "must_not": [{"say": "any"}],
        "escalate_by": None,
        "trigger": None,
        "uncertain": None,
        "needs_review": None,
        "cites": None,
        "rationale": None,
    }
    assert normalize_ollama_output({"checkpoints": [checkpoint]}) == {"checkpoints": [checkpoint]}


@pytest.mark.parametrize(
    ("cites", "expected", "note"),
    [
        (
            ["NICE-01", "NICE-06", "DICE-01"],
            ["NICE-01", "DICE-01"],
            "chest-pain: dropped 1 invalid citation (NICE-06)",
        ),
        (
            ["NICE-06", "MADE-UP"],
            [],
            "chest-pain: dropped 2 invalid citations (NICE-06, MADE-UP)",
        ),
    ],
)
def test_normalize_ollama_output_drops_invalid_citations(cites, expected, note):
    checkpoint = {
        "id": "chest-pain",
        "acceptable": [{"phase": "OBSERVING"}],
        "must_not": [{"say": "any"}],
        "escalate_by": None,
        "trigger": None,
        "uncertain": "",
        "needs_review": False,
        "cites": cites,
    }
    notes = set()

    normalized = normalize_ollama_output(
        {"checkpoints": [checkpoint]}, notes=notes, citable=["NICE-01", "DICE-01"]
    )

    assert normalized["checkpoints"][0]["cites"] == expected
    assert notes == {note}


@pytest.mark.parametrize("missing", ["id", "acceptable", "must_not"])
def test_normalize_ollama_output_leaves_checkpoint_missing_core_field_alone(missing):
    checkpoint = {
        "id": "repeats",
        "acceptable": [{"say": {"correction_of_reality": {}}}],
        "must_not": [],
    }
    del checkpoint[missing]
    assert normalize_ollama_output({"checkpoints": [checkpoint]}) == {"checkpoints": [checkpoint]}


def test_normalize_ollama_output_leaves_valid_actions_untouched():
    output = {
        "checkpoints": [{"acceptable": [{"phase": "OBSERVING"}], "must_not": [{"say": "any"}]}]
    }
    assert normalize_ollama_output(output) == output


def test_normalize_ollama_output_leaves_unknown_junk_for_validation():
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)
    output["checkpoints"][0]["acceptable"] = ["utter nonsense"]
    normalized = normalize_ollama_output(output)
    assert normalized["checkpoints"][0]["acceptable"] == ["utter nonsense"]
    _, problems = validate_annotation(scenario, normalized, citable_clauses())
    assert problems


def test_run_ollama_retries_with_validation_feedback(monkeypatch):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)
    invalid = _output(scenario, needs_review=False)
    invalid["checkpoints"][0]["acceptable"] = [{"say": "not_a_real_action"}]
    responses = iter([json.dumps(invalid), json.dumps(output)])

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": next(responses)}}).encode()

    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = run_ollama(
        ANNOTATOR_PROMPT_PATH,
        "user prompt",
        output_schema(scenario, citable_clauses()),
        "qwen3.5:9b",
        "high",
    )
    assert len(calls) == 2
    assert result["runner_attempts"] == 2
    assert result["attempt_temperatures"] == [0, 0.3]
    assert len(result["parse_validation_failures"]) == 1
    first_payload = json.loads(calls[0].data)
    retry_payload = json.loads(calls[1].data)
    assert first_payload["options"]["temperature"] == 0
    assert retry_payload["options"]["temperature"] == 0.3
    assert "seed" not in first_payload["options"]
    assert "seed" not in retry_payload["options"]
    retry_prompt = retry_payload["messages"][1]["content"]
    assert "Your previous answer was rejected" in retry_prompt
    assert "does not match exactly one schema" in retry_prompt
    assert json.dumps(invalid) in retry_prompt


def test_run_ollama_records_action_normalization(monkeypatch):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)
    output["checkpoints"][0]["must_not"] = [{"say": {"correction_of_reality": True}}]
    del output["checkpoints"][0]["trigger"]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": json.dumps(output)}}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    result = run_ollama(
        ANNOTATOR_PROMPT_PATH,
        "user prompt",
        output_schema(scenario, citable_clauses()),
        "qwen3.5:9b",
        "high",
    )
    assert result["structured_output"]["checkpoints"][0]["must_not"] == [
        {"say": "correction_of_reality"}
    ]
    assert result["normalization_changed"] is True
    assert result["normalization_notes"] == [
        "settles: collapsed nested say action",
        "settles: filled missing trigger",
    ]


def test_run_ollama_filters_citations_using_schema(monkeypatch):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)
    output["checkpoints"][0]["cites"] = ["NICE-05", "NICE-06"]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"message": {"content": json.dumps(output)}}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    result = run_ollama(
        ANNOTATOR_PROMPT_PATH,
        "user prompt",
        output_schema(scenario, citable_clauses()),
        "qwen3.5:9b",
        "high",
    )

    assert result["structured_output"]["checkpoints"][0]["cites"] == ["NICE-05"]
    assert result["normalization_notes"] == ["settles: dropped 1 invalid citation (NICE-06)"]


def test_local_yaml_surfaces_normalization_notes(tmp_path):
    scenario = load_scenario(FIXTURE)
    output = _output(scenario, needs_review=False)

    def runner(*args):
        return _fake_result(output, model="gemma4:e4b-mlx") | {
            "backend": "ollama",
            "normalization_changed": True,
            "normalization_notes": ["settles: filled missing trigger"],
        }

    result = annotate_scenario(
        scenario,
        profile=load_default_profile(),
        guidelines_text=GUIDELINES_PATH.read_text(),
        citable=citable_clauses(),
        runner=runner,
    )
    path = write_local_annotation(
        scenario,
        result,
        out_dir=tmp_path,
        effort="high",
        guidelines_text=GUIDELINES_PATH.read_text(),
    )
    annotator = yaml.safe_load(path.read_text())["annotator"]
    assert annotator["normalization_changed"] is True
    assert annotator["normalization_notes"] == ["settles: filled missing trigger"]


def test_ollama_refuses_model_annotation_out_dir():
    with pytest.raises(SystemExit, match="must not be annotations/model"):
        annotate_main(
            [
                "--backend",
                "ollama",
                "--model",
                "qwen3.5:9b",
                "--out-dir",
                str(Path(__file__).parents[1] / "annotations/model"),
            ]
        )


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
