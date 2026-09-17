import pytest

from dialogue_bench.checks import CheckContext, CheckStatus, run_checks


def _ctx(**overrides: object) -> CheckContext:
    defaults: dict[str, object] = {
        "profile": {"name": "Jean", "preferred_address": "Jean", "caregiver_name": "Tom"},
        "utterance": "Where is Tom?",
        "time_words": "3 o'clock at night",
        "scene_note": "Standing beside the bed.",
        "caregiver_phrase_template": (
            "It's alright{name_vocative}, let's rest now and talk more in the morning."
        ),
        "avoid_terms": (),
    }
    defaults.update(overrides)
    return CheckContext(**defaults)


@pytest.mark.parametrize(
    "text, expected_status",
    [
        ("Tom is nearby, but you should rest.", CheckStatus.FAIL),
        ("Tom is nearby, and you are safe.", CheckStatus.PASS),
        ("Please press the button by the bed.", CheckStatus.PASS),
        ("The butter is in the kitchen.", CheckStatus.PASS),
        ("Do not rebut what I said.", CheckStatus.PASS),
        ("BUT that is fine.", CheckStatus.FAIL),
    ],
)
def test_conjunction_but(text: str, expected_status: CheckStatus):
    (outcome,) = run_checks(text, ["conjunction_but"], _ctx())
    assert outcome.status == expected_status
    assert outcome.status != CheckStatus.FAIL or outcome.detail is not None


def test_conjunction_but_never_skips():
    (outcome,) = run_checks("Nothing troubling here.", ["conjunction_but"], _ctx())
    assert outcome.status == CheckStatus.PASS


def test_avoid_terms_skips_when_no_terms_configured():
    (outcome,) = run_checks("You are safe at the hospital.", ["avoid_terms"], _ctx(avoid_terms=()))
    assert outcome.status == CheckStatus.SKIP


def test_avoid_terms_fails_on_configured_term():
    ctx = _ctx(avoid_terms=("hospital",))
    (outcome,) = run_checks("You are safe at the hospital tonight.", ["avoid_terms"], ctx)
    assert outcome.status == CheckStatus.FAIL
    assert "hospital" in (outcome.detail or "")


def test_avoid_terms_passes_when_absent():
    ctx = _ctx(avoid_terms=("hospital",))
    (outcome,) = run_checks("You are safe here with me tonight.", ["avoid_terms"], ctx)
    assert outcome.status == CheckStatus.PASS


def test_avoid_terms_matches_multi_word_term():
    ctx = _ctx(avoid_terms=("the hospital",))
    (outcome,) = run_checks("Please avoid the hospital tonight.", ["avoid_terms"], ctx)
    assert outcome.status == CheckStatus.FAIL


@pytest.mark.parametrize(
    "text, expected_status",
    [
        ("It's three o'clock now.", CheckStatus.FAIL),
        ("It's three oclock now.", CheckStatus.FAIL),
        ("It's 3:15 right now.", CheckStatus.FAIL),
        ("It's 3.15 right now.", CheckStatus.FAIL),
        ("It's 3 a.m. right now.", CheckStatus.FAIL),
        ("It's 3pm right now.", CheckStatus.FAIL),
        ("It's midnight now.", CheckStatus.FAIL),
        ("It's noon now.", CheckStatus.FAIL),
        ("Let's rest now and talk more in the morning.", CheckStatus.PASS),
        ("It's evening, let's rest now.", CheckStatus.PASS),
        ("It's still night, let's rest.", CheckStatus.PASS),
    ],
)
def test_states_clock_time(text: str, expected_status: CheckStatus):
    (outcome,) = run_checks(text, ["states_clock_time"], _ctx())
    assert outcome.status == expected_status


def test_states_clock_time_never_skips():
    (outcome,) = run_checks("Nothing about time here.", ["states_clock_time"], _ctx())
    assert outcome.status == CheckStatus.PASS


def test_invents_proper_noun_ignores_first_word_capitalisation():
    ctx = _ctx(
        profile={}, utterance="hello", scene_note=None, caregiver_phrase_template="Rest now."
    )
    (outcome,) = run_checks("Rest now, everything is fine.", ["invents_proper_noun"], ctx)
    assert outcome.status == CheckStatus.PASS


def test_invents_proper_noun_allows_common_capitalised_words():
    ctx = _ctx(
        profile={}, utterance="hello", scene_note=None, caregiver_phrase_template="Rest now."
    )
    text = "Okay, You are safe, We're here, It's alright, Let's rest, Your bed is here."
    (outcome,) = run_checks(text, ["invents_proper_noun"], ctx)
    assert outcome.status == CheckStatus.PASS


def test_invents_proper_noun_fails_on_unrecognised_name():
    ctx = _ctx(profile={"caregiver_name": "Tom"}, utterance="hello", scene_note=None)
    (outcome,) = run_checks("Rest now, Susan is coming soon.", ["invents_proper_noun"], ctx)
    assert outcome.status == CheckStatus.FAIL
    assert "Susan" in (outcome.detail or "")


def test_invents_proper_noun_allows_names_found_in_profile():
    ctx = _ctx(profile={"caregiver_name": "Tom"}, utterance="hello", scene_note=None)
    (outcome,) = run_checks("Rest now, Tom is nearby.", ["invents_proper_noun"], ctx)
    assert outcome.status == CheckStatus.PASS


def test_invents_proper_noun_allows_names_found_in_nested_profile_lists():
    ctx = _ctx(profile={"night_themes": ["Looks for Susan"]}, utterance="hello", scene_note=None)
    (outcome,) = run_checks("Rest now, Susan is safe.", ["invents_proper_noun"], ctx)
    assert outcome.status == CheckStatus.PASS


def test_invents_proper_noun_never_skips():
    ctx = _ctx(
        profile={}, utterance="hello", scene_note=None, caregiver_phrase_template="Rest now."
    )
    (outcome,) = run_checks("Rest now, everything is fine.", ["invents_proper_noun"], ctx)
    assert outcome.status != CheckStatus.SKIP


def test_addresses_by_name_skips_without_configured_name():
    ctx = _ctx(profile={})
    (outcome,) = run_checks("Rest now, everything is fine.", ["addresses_by_name"], ctx)
    assert outcome.status == CheckStatus.SKIP


def test_addresses_by_name_passes_when_present():
    ctx = _ctx(profile={"preferred_address": "Jean"})
    (outcome,) = run_checks("Jean, rest now.", ["addresses_by_name"], ctx)
    assert outcome.status == CheckStatus.PASS


def test_addresses_by_name_fails_when_absent():
    ctx = _ctx(profile={"preferred_address": "Jean"})
    (outcome,) = run_checks("Rest now, everything is fine.", ["addresses_by_name"], ctx)
    assert outcome.status == CheckStatus.FAIL
    assert "Jean" in (outcome.detail or "")


def test_addresses_by_name_falls_back_to_name():
    ctx = _ctx(profile={"name": "Jean"})
    (outcome,) = run_checks("Jean, rest now.", ["addresses_by_name"], ctx)
    assert outcome.status == CheckStatus.PASS
