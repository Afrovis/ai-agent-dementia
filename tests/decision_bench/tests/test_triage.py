from decision_bench.triage import checkpoint_flags, scenario_flags


def _cp(**overrides):
    base = {
        "id": "cp",
        "acceptable": [{"goal": "restroom"}, {"strategy": "path_light"}],
        "must_not": [{"notify": "critical"}],
        "escalate_by": None,
        "trigger": None,
        "needs_review": False,
    }
    return base | overrides


def test_agreeing_runs_without_flags_pass_triage():
    assert checkpoint_flags(_cp(), _cp(acceptable=[{"goal": "restroom"}])) == []


def test_self_flag_and_second_opinion_flag():
    assert checkpoint_flags(_cp(needs_review=True), _cp()) == ["the annotator asked for review"]
    assert checkpoint_flags(_cp(), _cp(needs_review=True)) == [
        "the second opinion asked for review"
    ]


def test_disagreements_that_can_change_a_critical_result_are_flagged():
    assert "must_not differs" in checkpoint_flags(_cp(), _cp(must_not=[]))[0]
    assert "deadline" in checkpoint_flags(_cp(), _cp(escalate_by=120, trigger=10))[0]
    assert "share nothing" in checkpoint_flags(_cp(), _cp(acceptable=[{"phase": "IDLE"}]))[0]


def test_deadline_number_alone_is_not_flagged():
    assert checkpoint_flags(_cp(escalate_by=60), _cp(escalate_by=300)) == []


def test_triage_never_accepts_what_it_cannot_check():
    legacy = {key: value for key, value in _cp().items() if key != "needs_review"}
    assert "predates self-flagging" in checkpoint_flags(legacy, _cp())[0]
    assert "no second opinion" in checkpoint_flags(_cp(), None)[0]
    assert scenario_flags({"checkpoints": [_cp()]}) == {
        "cp": ["there is no second opinion to compare with"]
    }
    assert (
        scenario_flags({"checkpoints": [_cp()], "second_opinion": {"checkpoints": [_cp()]}}) == {}
    )
