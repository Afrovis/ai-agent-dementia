"""Small direct-question check shared by reply routing and trace checks."""

import re

_HEARING_PHRASES = (
    "speak up",
    "louder",
    "can't hear",
    "cannot hear",
    "can not hear",
    "didn't catch",
    "did not catch",
    "say that again",
    "say it again",
    "pardon",
    "come again",
)


def is_hearing_request(text: str) -> bool:
    """Recognize explicit hearing requests and short standalone clarifications."""
    lower = text.lower().replace("’", "'")
    if any(phrase in lower for phrase in _HEARING_PHRASES):
        return True
    # "What's that?" / "What was that?" are clarifications only when the clause
    # ends there, not in "what's that noise?" or "what was that sound?".
    that = r"what(?:'s| was) that"
    if re.search(rf"\b{that}\s*[?.!,;:]\s*(?:dear\b|$)", lower) or re.fullmatch(
        rf"\s*{that}\s*(?:dear)?\s*[?.!,;:]*\s*", lower
    ):
        return True
    return bool(re.fullmatch(r"\s*(?:what|eh|huh)\s*[?.!,;:]*\s*(?:dear\b)?\s*[?.!,;:]*\s*", lower))


def is_direct_question(text: str) -> bool:
    """Recognize a request for an answer even when transcription omits `?`."""
    lower = text.lower()
    return "?" in text or any(
        phrase in lower
        for phrase in (
            "is anyone",
            "is someone",
            "can you hear",
            "where is",
            "are you there",
            "is that you",
            "who's there",
            "when will",
            "what time",
            "where am i",
            "what's the time",
            "is it morning",
        )
    )
