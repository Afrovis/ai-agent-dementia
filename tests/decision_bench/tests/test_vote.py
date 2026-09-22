from decision_bench.vote import vote_checkpoint, vote_scenario


def _run(**overrides):
    base = {
        "id": "cp",
        "acceptable": [],
        "must_not": [],
        "escalate_by": None,
        "trigger": None,
        "rationale": "because",
        "cites": ["NICE-05"],
    }
    return base | overrides


def test_action_reaching_two_of_three_wins_its_field():
    vote = vote_checkpoint(
        [
            _run(acceptable=[{"goal": "restroom"}]),
            _run(acceptable=[{"goal": "restroom"}]),
            _run(acceptable=[]),
        ]
    )
    assert vote.acceptable == [{"goal": "restroom"}]
    assert vote.doubtful_acceptable == []


def test_must_not_majority_wins_over_a_lone_acceptable_vote():
    vote = vote_checkpoint(
        [
            _run(must_not=[{"notify": "critical"}]),
            _run(must_not=[{"notify": "critical"}]),
            _run(acceptable=[{"notify": "critical"}]),
        ]
    )
    assert vote.must_not == [{"notify": "critical"}]
    assert vote.doubtful_acceptable == []
    assert vote.doubtful_must_not == []


def test_lone_vote_with_no_majority_either_way_is_doubtful():
    vote = vote_checkpoint(
        [
            _run(acceptable=[{"strategy": "path_light"}]),
            _run(),
            _run(),
        ]
    )
    assert vote.acceptable == []
    assert vote.doubtful_acceptable == [{"strategy": "path_light"}]
    assert vote.doubtful_must_not == []


def test_accepted_by_one_and_forbidden_by_one_is_doubtful_both_ways():
    vote = vote_checkpoint(
        [
            _run(acceptable=[{"strategy": "guided_return"}]),
            _run(must_not=[{"strategy": "guided_return"}]),
            _run(),
        ]
    )
    assert vote.acceptable == []
    assert vote.must_not == []
    assert vote.doubtful_acceptable == [{"strategy": "guided_return"}]
    assert vote.doubtful_must_not == [{"strategy": "guided_return"}]


def test_accepted_by_two_and_forbidden_by_one_is_acceptable_only():
    vote = vote_checkpoint(
        [
            _run(acceptable=[{"strategy": "guided_return"}]),
            _run(acceptable=[{"strategy": "guided_return"}]),
            _run(must_not=[{"strategy": "guided_return"}]),
        ]
    )
    assert vote.acceptable == [{"strategy": "guided_return"}]
    assert vote.doubtful_acceptable == []
    assert vote.doubtful_must_not == []


def test_forbidden_by_two_and_accepted_by_one_is_must_not_only():
    vote = vote_checkpoint(
        [
            _run(must_not=[{"strategy": "guided_return"}]),
            _run(must_not=[{"strategy": "guided_return"}]),
            _run(acceptable=[{"strategy": "guided_return"}]),
        ]
    )
    assert vote.must_not == [{"strategy": "guided_return"}]
    assert vote.doubtful_acceptable == []
    assert vote.doubtful_must_not == []


def test_escalate_by_needs_two_runs_and_takes_the_lower_median():
    # Only one run sets it: unset.
    assert vote_checkpoint([_run(escalate_by=60), _run(), _run()]).escalate_by is None
    # Two runs set it: the smaller of the two.
    assert vote_checkpoint([_run(escalate_by=60), _run(escalate_by=180), _run()]).escalate_by == 60
    # Three runs set it: the middle value.
    vote = vote_checkpoint(
        [_run(escalate_by=60), _run(escalate_by=180), _run(escalate_by=120)]
    )
    assert vote.escalate_by == 120


def test_trigger_majority_and_no_majority_when_all_three_differ():
    vote = vote_checkpoint(
        [
            _run(escalate_by=60, trigger=10),
            _run(escalate_by=60, trigger=10),
            _run(escalate_by=60, trigger=20),
        ]
    )
    assert vote.trigger == 10
    assert vote.no_majority == []

    vote = vote_checkpoint(
        [
            _run(escalate_by=60, trigger=10),
            _run(escalate_by=60, trigger=20),
            _run(escalate_by=60, trigger=30),
        ]
    )
    assert vote.trigger is None
    assert "no majority on trigger" in vote.no_majority


def test_trigger_disagreement_does_not_matter_without_a_majority_escalate_by():
    # Only one run sets escalate_by, so it stays unset and trigger is never voted.
    vote = vote_checkpoint(
        [_run(escalate_by=60, trigger=10), _run(trigger=20), _run(trigger=30)]
    )
    assert vote.escalate_by is None
    assert vote.no_majority == []


def test_no_majority_on_acceptable_when_every_vote_is_a_lone_disagreement():
    vote = vote_checkpoint(
        [
            _run(acceptable=[{"strategy": "path_light"}]),
            _run(acceptable=[{"strategy": "soft_greeting"}]),
            _run(acceptable=[{"strategy": "ambient_orient"}]),
        ]
    )
    assert vote.acceptable == []
    assert "no majority on acceptable" in vote.no_majority


def test_empty_acceptable_agreed_by_all_three_is_not_a_disagreement():
    vote = vote_checkpoint([_run(), _run(), _run()])
    assert vote.acceptable == []
    assert vote.no_majority == []


def test_checkpoint_missing_from_two_or_more_runs_has_no_majority():
    vote = vote_checkpoint([_run(), None, None])
    assert vote.no_majority == ["checkpoint missing from 2+ runs"]
    assert vote.acceptable == []


def test_rationale_and_cites_come_from_the_first_run():
    vote = vote_checkpoint(
        [
            _run(rationale="first says so", cites=["NICE-01"]),
            _run(rationale="second says something else", cites=["AA-01"]),
            _run(rationale="third", cites=["VAL-01"]),
        ]
    )
    assert vote.rationale == "first says so"
    assert vote.cites == ["NICE-01"]


def test_vote_record_counts_actions_and_escalate_by_setters():
    vote = vote_checkpoint(
        [
            _run(acceptable=[{"goal": "restroom"}], escalate_by=60),
            _run(acceptable=[{"goal": "restroom"}]),
            _run(),
        ]
    )
    assert vote.record["acceptable_votes"] == {"goal: restroom": 2}
    assert vote.record["escalate_by_set_by"] == ["annotator"]


def test_vote_scenario_votes_every_checkpoint_across_three_runs():
    model_doc = {
        "checkpoints": [_run(id="a", acceptable=[{"goal": "restroom"}])],
        "second_opinion": {"checkpoints": [_run(id="a", acceptable=[{"goal": "restroom"}])]},
        "third_opinion": {"checkpoints": [_run(id="a")]},
    }
    votes = vote_scenario(model_doc)
    assert votes["a"].acceptable == [{"goal": "restroom"}]
