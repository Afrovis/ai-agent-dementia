from dialogue_bench.checks import CheckContext

from decision_bench.checks import check_pattern


def _context() -> CheckContext:
    return CheckContext(
        profile={"name": "Jean", "preferred_address": "Jean"},
        utterance="I need help",
        time_words="3 o'clock at night",
        scene_note=None,
        caregiver_phrase_template="Let's rest.",
        avoid_terms=("hospital",),
    )


def test_dialogue_patterns_map_failure_to_shown_and_name_pass_to_shown():
    ctx = _context()
    assert check_pattern("I hear you, but let's rest.", "conjunction_but", ctx).status == "shown"
    assert check_pattern("Jean, let's rest.", "addresses_by_name", ctx).status == "shown"
    assert check_pattern("Let's rest.", "addresses_by_name", ctx).status == "not_shown"
    assert check_pattern("The hospital can wait.", "avoid_terms", ctx).status == "shown"
    assert check_pattern("It is 3:12.", "states_clock_time", ctx).status == "shown"
    assert check_pattern("Let's visit Paris.", "invents_proper_noun", ctx).status == "shown"


def test_memory_question_positive_and_negative_phrases():
    ctx = _context()
    positives = [
        "Do you remember Tom?",
        "Don't you remember?",
        "Can you remember the garden?",
        "Remember when we went home?",
        "Don't you know that?",
        "Do you know what day it is?",
        "Do you recall the room?",
    ]
    for text in positives:
        assert check_pattern(text, "memory_question", ctx).status == "shown", text
    for text in ("I remember the garden.", "You know the way.", "Let's rest now."):
        assert check_pattern(text, "memory_question", ctx).status == "not_shown", text


def test_blunt_refusal_positive_and_allowed_phrases():
    ctx = _context()
    for text in (
        "No, stay here.",
        "No.",
        "You can't go.",
        "You cannot leave.",
        "You're not allowed outside.",
        "You are not allowed to go.",
        "It is not allowed to happen.",
        "Don't go.",
        "Stop that.",
    ):
        assert check_pattern(text, "blunt_refusal", ctx).status == "shown", text
    for text in (
        "No need to worry.",
        "No rush, Jean.",
        "No hurry now.",
        "I know the way.",
        "Nobody is rushing.",
        "It is not far.",
    ):
        assert check_pattern(text, "blunt_refusal", ctx).status == "not_shown", text


def test_semantic_patterns_always_request_review():
    ctx = _context()
    assert check_pattern("Let's rest.", "correction_of_reality", ctx).status == "review"
    assert check_pattern("Let's rest.", "infantilising", ctx).status == "review"
