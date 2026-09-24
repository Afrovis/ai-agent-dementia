"""Deterministic hearing-request recognition."""

import pytest

from agent.questions import is_hearing_request


@pytest.mark.parametrize(
    "text",
    [
        "Speak up, dear.",
        "Louder!",
        "I can't hear you good.",
        "I cannot hear",
        "I can not hear",
        "I didn't catch it.",
        "I did not catch that",
        "Say that again",
        "Say it again",
        "What was that?",
        "What's that? Dear, I didn't catch it.",
        "What's that?",
        "Pardon?",
        "Come again?",
        "What?",
        "eh? dear",
        "Huh?",
    ],
)
def test_hearing_requests(text):
    assert is_hearing_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "What time is it?",
        "What's that noise outside?",
        "What is that?",
        "What? Is Tom here?",
        "What's that on the table?",
    ],
)
def test_other_questions(text):
    assert not is_hearing_request(text)


def test_what_was_that_is_a_hearing_request_only_on_its_own():
    from agent.questions import is_hearing_request

    assert is_hearing_request("What was that? I can't hear you good.")
    assert is_hearing_request("What was that, dear?")
    assert not is_hearing_request("What was that noise outside?")
