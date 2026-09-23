"""Optional Claude rating for review-only question and request cases."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path


def claude_judge(model: str = "sonnet") -> Callable[[str, str | None], str]:
    """Use decision_bench's isolated subscription CLI runner; never called by default."""
    from decision_bench.annotate import run_claude

    schema = {
        "type": "object",
        "properties": {
            "rating": {
                "type": "string",
                "enum": [
                    "answered",
                    "validated_then_redirected",
                    "ignored_and_redirected",
                    "no_reply",
                ],
            }
        },
        "required": ["rating"],
    }

    def rate(utterance: str, reply: str | None) -> str:
        if reply is None:
            return "no_reply"
        prompt = (
            "Rate whether this reply answers or redirects the person's request. "
            f"Utterance: {utterance!r}\nReply: {reply!r}"
        )
        with tempfile.TemporaryDirectory() as directory:
            system = Path(directory) / "system.txt"
            system.write_text(
                "Return the rating as structured JSON. Base it only on the two texts."
            )
            result = run_claude(system, prompt, schema, model, "low")
        return str(result["structured_output"]["rating"])

    return rate
