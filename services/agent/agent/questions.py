"""Small direct-question check shared by reply routing and trace checks."""


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
