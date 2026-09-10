"""Vision-LLM scene notes via a local Ollama server (issue #9).

PLAN.md section 6.1: "When the state changes, or every 60 s while
`ENGAGED`, `perceive` sends one frame to the vision LLM with a fixed
prompt... The output is attached to the `PersonState` event as
`scene_note`." HANDOFF.md section 11 forbids a cloud vision model outright,
so the only client this module ships talks to Ollama on the host, reached
at `OLLAMA_URL` (`http://host.docker.internal:11434` by default) -- never
any other endpoint.

`VisionClient` is the tiny interface `perceive.scene_notes.SceneNoteCache`
drives: one JPEG in, one sentence (or `None`) out. `OllamaVisionClient` is
the real implementation; `FakeVisionClient` is what every test uses instead
(HANDOFF.md section 4: no service may require Ollama to run its tests).

Frame bytes are base64-encoded in memory for exactly as long as it takes to
build the request body, sent to `OLLAMA_URL` and nowhere else, and then
dropped -- never written to disk, never logged (HANDOFF.md rule 2 and the
hard rule repeated in `perceive.main`'s module docstring).

`OllamaVisionClient.describe` must never raise into its caller: a dead or
slow vision model degrades `scene_note` to `None`, the same "fail safe"
posture `perceive.backends._decode_rgb` applies to a corrupt frame -- it
must not cost `perceive` a single `PersonState` publish (HANDOFF.md rule 4).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

SERVICE_NAME = "perceive"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

VISION_PROMPT = (
    "Describe in one sentence what the person is doing and whether they "
    "look distressed. Do not describe their identity."
)
"""The fixed prompt, verbatim from PLAN.md section 6.1. Not configurable:
a caregiver-editable prompt could drift into describing identity, which
this system must never do, and a consistent prompt is what makes the
sanitisation in `_sanitize_note` (first sentence, no quotes) a reasonable
thing to rely on."""

MAX_SCENE_NOTE_LENGTH = 160
"""Maximum length of a sanitised `scene_note`, in characters. The prompt
asks for one sentence; this is the backstop for a model that ignores it.
160 is generous for one plain-English sentence describing an action and a
distress read (HANDOFF.md section 6's `Say` one-sentence rule uses a
20-word budget for spoken text; a scene note is read by the caregiver, not
spoken to the person at night, so it can run a little longer than that,
but still needs a hard ceiling so a rambling model output never balloons
into something that reads like a transcript)."""

_OLLAMA_GENERATE_PATH = "/api/generate"
_QUOTE_CHARS = "\"'“”‘’ "


class VisionClient(Protocol):
    """The interface `perceive.scene_notes.SceneNoteCache` drives.

    `None` means the call failed, timed out, or returned nothing usable --
    the caller treats that identically to "no scene note this round",
    never as an error to propagate.
    """

    def describe(self, jpeg: bytes) -> str | None:
        """Return a one-sentence description of `jpeg`, or `None`."""
        ...


def _sanitize_note(raw: str) -> str | None:
    """Turn a model's raw text output into a safe-to-publish `scene_note`.

    The prompt asks for one sentence; this does not trust the model to
    comply. Collapses all whitespace (including newlines a chatty model
    might add), strips surrounding quote characters, keeps only the text up
    to and including the first sentence-ending punctuation mark (if any),
    and truncates to `MAX_SCENE_NOTE_LENGTH`. Returns `None` if nothing
    usable is left.
    """
    collapsed = " ".join(raw.split())
    collapsed = collapsed.strip(_QUOTE_CHARS)
    if not collapsed:
        return None

    match = re.search(r"[.!?]", collapsed)
    if match:
        collapsed = collapsed[: match.end()]

    collapsed = collapsed.strip()
    if not collapsed:
        return None

    if len(collapsed) > MAX_SCENE_NOTE_LENGTH:
        collapsed = collapsed[:MAX_SCENE_NOTE_LENGTH].rstrip()

    return collapsed or None


def _log_failure(reason: str, **fields: object) -> None:
    """Log a structured warning for a failed vision call. Never includes
    frame bytes -- only the reason and, where relevant, status/error text."""
    logger.warning(
        json.dumps(
            {
                "service": SERVICE_NAME,
                "message": "vision call failed, scene_note stays unset",
                "reason": reason,
                **fields,
            }
        )
    )


class OllamaVisionClient:
    """Posts one frame to a local Ollama server's `/api/generate` endpoint.

    Stateless and synchronous: one HTTP POST per `describe` call, with an
    explicit timeout. Runs off the frame-processing thread entirely --
    `perceive.scene_notes.SceneNoteCache` is what keeps this from stalling
    the frame loop; this class itself makes no attempt to be fast.

    Any transport error, timeout, non-200 response, unparseable body, or
    empty `response` field logs a warning and returns `None`. Nothing here
    ever raises into the caller.
    """

    def __init__(self, *, ollama_url: str, model: str, timeout_seconds: float = 10.0) -> None:
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def describe(self, jpeg: bytes) -> str | None:
        """Send `jpeg` to Ollama with the fixed prompt and return a sanitised
        one-sentence note, or `None` on any failure."""
        payload = {
            "model": self._model,
            "prompt": VISION_PROMPT,
            "images": [base64.b64encode(jpeg).decode("ascii")],
            "stream": False,
        }
        try:
            response = httpx.post(
                f"{self._ollama_url}{_OLLAMA_GENERATE_PATH}",
                json=payload,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - any transport failure degrades to None
            _log_failure("request failed", error=str(exc))
            return None

        if response.status_code != 200:
            _log_failure("non-200 response", status_code=response.status_code)
            return None

        try:
            body = response.json()
        except ValueError as exc:
            _log_failure("response was not JSON", error=str(exc))
            return None

        raw = body.get("response") if isinstance(body, dict) else None
        if not raw or not isinstance(raw, str):
            _log_failure("response had no usable text")
            return None

        note = _sanitize_note(raw)
        if note is None:
            _log_failure("response sanitised to nothing")
        return note


class FakeVisionClient:
    """Returns canned responses in order, for tests. No Ollama, no network.

    `responses` may include `None` entries to script a failed/empty call.
    Every call (including its `jpeg` argument) is recorded in `calls`, so
    tests can assert exactly how many requests fired and with what frame.
    Once `responses` is exhausted, further calls return `None`.
    """

    def __init__(self, responses: Sequence[str | None] = ()) -> None:
        self._responses = list(responses)
        self.calls: list[bytes] = []

    def describe(self, jpeg: bytes) -> str | None:
        self.calls.append(jpeg)
        if not self._responses:
            return None
        return self._responses.pop(0)
