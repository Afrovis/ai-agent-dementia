from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from decision_bench.schema import (
    CATEGORIES,
    PROFILE_FIELDS,
    SAY_PATTERNS,
    STRATEGIES,
    Action,
    Checkpoint,
    Scenario,
    citation_problems,
    guideline_clauses,
    load_default_profile,
    load_scenario,
    load_scenarios,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _scenario(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "id": "sample-01",
        "category": "restroom",
        "summary": "A sample night.",
        "start": "02:00",
        "timeline": [
            {"t": 0, "person": {"state": "sitting_up", "zone": "bed"}},
            {"t": 10, "utterance": {"text": "I need the toilet."}},
        ],
        "checkpoints": [{"id": "first", "window": [10, 40], "question": "What now?"}],
    }
    raw.update(overrides)
    return raw


def _labelled(**overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "id": "first",
        "window": [10, 40],
        "acceptable": [{"strategy": "path_light"}],
        "must_not": [{"say": "correction_of_reality"}],
        "rationale": "Light the way.",
        "cites": ["TOIL-02"],
    }
    raw.update(overrides)
    return raw


# --- fixtures in the repository ---


def test_pilot_scenarios_load():
    scenarios = load_scenarios()
    assert len(scenarios) == 7
    by_category = {c: sum(s.category == c for s in scenarios) for c in CATEGORIES}
    assert by_category == {c: 2 if c == "false_alarm" else 1 for c in CATEGORIES}


def test_labels_come_from_the_annotation_workflow():
    # A labelled clean scenario needs an annotator draft and a reviewed human copy,
    # so labels are never hand-copied from what the agent does.
    annotations = Path(__file__).parents[1] / "annotations"
    scenarios = load_scenarios()
    for scenario in scenarios:
        if not scenario.labelled or scenario.noise_of is not None:
            continue
        assert (annotations / "model" / f"{scenario.id}.yaml").exists(), scenario.id
        review = yaml.safe_load((annotations / "review" / f"{scenario.id}.yaml").read_text())
        assert review["reviewed"] is True, scenario.id
    assert citation_problems(scenarios, guideline_clauses()) == []


def test_default_profile_loads():
    profile = load_default_profile()
    assert profile["preferred_address"] == "Jean"


def test_guideline_clauses_parse():
    clauses = guideline_clauses()
    assert len(clauses) >= 20
    assert {c.split("-")[0] for c in clauses} == {
        "NICE",
        "AA",
        "VAL",
        "PCC",
        "DICE",
        "FALL",
        "TOIL",
    }


def test_fixture_cites_name_real_clauses():
    clauses = guideline_clauses()
    unknown = [p for p in citation_problems(load_scenarios(), clauses) if "unknown" in p]
    assert unknown == []


def test_guidelines_list_every_say_pattern():
    text = (REPO_ROOT / "tests/decision_bench/guidelines.md").read_text()
    for pattern in SAY_PATTERNS - {"any"}:
        assert f"`{pattern}`" in text, pattern


# --- vocabulary stays in step with the agent and dialogue_bench ---


def test_strategies_match_the_example_config():
    path = REPO_ROOT / "config/strategies.example.yaml"
    if not path.exists():
        pytest.skip("strategies.example.yaml not present")
    raw = yaml.safe_load(path.read_text())
    entries = raw["strategies"] if isinstance(raw, dict) else raw
    ids = {entry["id"] for entry in entries} if isinstance(entries, list) else set(entries)
    assert ids == STRATEGIES


def test_profile_fields_match_the_agent():
    profile = pytest.importorskip("agent.profile")
    agent_fields = {f.name for f in fields(profile.PersonProfile)} - {"enable_cloud_fallback"}
    assert agent_fields == PROFILE_FIELDS


def test_say_patterns_include_every_dialogue_check():
    checks = pytest.importorskip("dialogue_bench.checks")
    assert checks.CHECK_NAMES <= SAY_PATTERNS


# --- scenario validation ---


def test_start_must_be_a_quoted_time():
    with pytest.raises(ValidationError, match="quoted local time"):
        Scenario.model_validate(_scenario(start=134))


def test_timeline_must_be_in_order():
    timeline = [
        {"t": 10, "person": {"state": "standing", "zone": "bed"}},
        {"t": 5, "person": {"state": "sitting_up", "zone": "bed"}},
    ]
    with pytest.raises(ValidationError, match="time order"):
        Scenario.model_validate(_scenario(timeline=timeline))


def test_timeline_event_needs_exactly_one_input():
    both = {"t": 0, "person": {"state": "in_bed", "zone": "bed"}, "utterance": {"text": "hi"}}
    with pytest.raises(ValidationError, match="exactly one"):
        Scenario.model_validate(_scenario(timeline=[both]))
    with pytest.raises(ValidationError, match="exactly one"):
        Scenario.model_validate(_scenario(timeline=[{"t": 0}]))


def test_unknown_person_state_is_rejected():
    timeline = [{"t": 0, "person": {"state": "dancing", "zone": "bed"}}]
    with pytest.raises(ValidationError):
        Scenario.model_validate(_scenario(timeline=timeline))


def test_unknown_profile_field_is_rejected():
    with pytest.raises(ValidationError, match="unknown profile fields"):
        Scenario.model_validate(_scenario(profile={"favourite_colour": "blue"}))


def test_duplicate_checkpoint_ids_are_rejected():
    checkpoint = {"id": "first", "window": [0, 10], "question": "?"}
    with pytest.raises(ValidationError, match="duplicate checkpoint"):
        Scenario.model_validate(_scenario(checkpoints=[checkpoint, checkpoint]))


def test_familiar_voice_needs_a_clip():
    checkpoint = _labelled(acceptable=[{"strategy": "familiar_voice"}], cites=["AA-04"])
    with pytest.raises(ValidationError, match="voice_clip"):
        Scenario.model_validate(_scenario(checkpoints=[checkpoint]))
    scenario = Scenario.model_validate(_scenario(checkpoints=[checkpoint], voice_clip=True))
    assert scenario.labelled


def test_load_scenario_checks_the_file_name(tmp_path):
    path = tmp_path / "other-name.yaml"
    path.write_text(yaml.safe_dump(_scenario()))
    with pytest.raises(ValueError, match="does not match the file name"):
        load_scenario(path)


def test_noise_of_must_name_a_clean_scenario(tmp_path):
    (tmp_path / "sample-01.yaml").write_text(yaml.safe_dump(_scenario()))
    noisy = _scenario(id="sample-01-noisy", noise_of="missing-01")
    (tmp_path / "sample-01-noisy.yaml").write_text(yaml.safe_dump(noisy))
    with pytest.raises(ValueError, match="unknown scenario"):
        load_scenarios(tmp_path)
    noisy["noise_of"] = "sample-01"
    (tmp_path / "sample-01-noisy.yaml").write_text(yaml.safe_dump(noisy))
    assert [s.id for s in load_scenarios(tmp_path)] == ["sample-01", "sample-01-noisy"]


# --- actions and checkpoints ---


@pytest.mark.parametrize(
    "raw",
    [
        {"strategy": "path_light"},
        {"phase": "ESCALATED"},
        {"goal": "comfort"},
        {"notify": "any"},
        {"say": "correction_of_reality"},
    ],
)
def test_valid_actions(raw):
    action = Action.model_validate(raw)
    [(kind, value)] = raw.items()
    assert (action.kind, action.value) == (kind, value)


@pytest.mark.parametrize(
    "raw",
    [
        {"strategy": "sing_a_song"},
        {"phase": "PANIC"},
        {"notify": "urgent"},
        {"say": "rude"},
        {"strategy": "path_light", "notify": "any"},
        {},
    ],
)
def test_invalid_actions(raw):
    with pytest.raises(ValidationError):
        Action.model_validate(raw)


def test_unlabelled_checkpoint_needs_a_question():
    with pytest.raises(ValidationError, match="needs a question"):
        Checkpoint.model_validate({"id": "first", "window": [0, 10]})


def test_labelled_checkpoint_needs_rationale_and_cites():
    with pytest.raises(ValidationError, match="rationale"):
        Checkpoint.model_validate(_labelled(rationale=None))
    with pytest.raises(ValidationError, match="cite"):
        Checkpoint.model_validate(_labelled(cites=[]))
    assert Checkpoint.model_validate(_labelled()).labelled


def test_cites_must_look_like_clause_ids():
    with pytest.raises(ValidationError, match="clause id"):
        Checkpoint.model_validate(_labelled(cites=["NICE 1.7.1"]))


def test_window_must_not_run_backwards():
    with pytest.raises(ValidationError, match="starts after"):
        Checkpoint.model_validate({"id": "first", "window": [40, 10], "question": "?"})


def test_escalate_by_must_come_from_the_caregiver():
    base = {"id": "caregiver", "escalate_by": 300, "rationale": "Tell someone."}
    with pytest.raises(ValidationError, match="threshold_source"):
        Checkpoint.model_validate(base)
    checkpoint = Checkpoint.model_validate(base | {"threshold_source": "caregiver", "trigger": 48})
    assert checkpoint.labelled
    assert checkpoint.deadline_from == 48


def test_deadline_defaults_to_window_start():
    checkpoint = Checkpoint.model_validate(
        _labelled(escalate_by=120, threshold_source="caregiver", window=[48, 200])
    )
    assert checkpoint.deadline_from == 48


def test_threshold_source_without_deadline_is_rejected():
    with pytest.raises(ValidationError, match="only go with escalate_by"):
        Checkpoint.model_validate(_labelled(threshold_source="caregiver"))


# --- guideline pack parsing ---


def test_citation_problems_flag_unknown_and_unchecked(tmp_path):
    scenario = Scenario.model_validate(
        _scenario(checkpoints=[_labelled(cites=["TOIL-02", "AA-01", "AA-99"])])
    )
    clauses = {"TOIL-02": True, "AA-01": False}
    assert citation_problems([scenario], clauses) == [
        "sample-01/first cites AA-01, which is not checked yet",
        "sample-01/first cites unknown clause AA-99",
    ]


def test_guideline_clauses_read_checked_boxes(tmp_path):
    path = tmp_path / "guidelines.md"
    path.write_text(
        "### AA-01 · One\n\n- **Checked:** [x]\n\n### AA-02 · Two\n\n- **Checked:** [ ]\n"
    )
    assert guideline_clauses(path) == {"AA-01": True, "AA-02": False}


def test_guideline_clause_without_checked_line_is_rejected(tmp_path):
    path = tmp_path / "guidelines.md"
    path.write_text("### AA-01 · One\n\nNo box here.\n")
    with pytest.raises(ValueError, match="no Checked line"):
        guideline_clauses(path)
