"""Local vision-model second opinion for an ambiguous floor situation.

The pose pipeline (`perceive.classify`) is the authority: this module can
only *upgrade* an ambiguous frame to `on_floor`, never veto or clear a
pose-confirmed one, and it must never block the frame loop the way
`perceive.scene_notes.SceneNoteCache` already promises for scene notes --
the same "at most one request in flight, fire-and-forget, main thread reads
the completed result" split applies here, so `FloorCheckScheduler` mirrors
`SceneNoteCache` closely.

Three pieces:

- `FloorCheckTrigger` decides, from a classified frame and the tracker's
  own read-only signals, whether *this frame* warrants a check, and why:
  a fresh `floor_suspect`, a person lost outside the bed (and not through
  the door) within the last 30 s, or `sitting_up` outside the bed with a
  low height ratio persisting a few seconds.
- `FloorCheckClient` (`OllamaFloorCheckClient`/`FakeFloorCheckClient`) is
  the same "one JPEG in, one answer or `None` out" shape as
  `perceive.vision.VisionClient`, with a boolean-plus-confidence answer
  instead of a sentence.
- `FloorCheckScheduler` is the non-blocking scheduling: fires at most one
  request at a time, respects a cooldown between triggers, and hands the
  completed answer back to `perceive.main.run_once` to re-validate and
  apply -- `StateTracker.confirm_floor` is the only thing that ever acts
  on it, and only after the trigger condition is checked again against the
  frame current when the answer arrives.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Protocol

import httpx

from perceive.classify import PersonStateName, StateTracker
from perceive.zones import ZoneName

SERVICE_NAME = "perceive"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

FLOOR_CHECK_PROMPT = (
    "You check a bedroom safety camera frame. Is a person lying, sitting or "
    "kneeling ON THE FLOOR (not on a bed, chair or other furniture)? If no "
    "person is visible, answer false. Answer JSON only."
)
"""Fixed, not configurable -- same reasoning as `perceive.vision.VISION_PROMPT`:
a caregiver-editable prompt could drift, and `OllamaFloorCheckClient` relies
on this exact question pairing with the fixed `_FLOOR_CHECK_FORMAT` schema
below."""

_FLOOR_CHECK_FORMAT = {
    "type": "object",
    "properties": {
        "person_on_floor": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["person_on_floor", "confidence"],
}

_OLLAMA_CHAT_PATH = "/api/chat"


@dataclass(frozen=True)
class FloorCheckResult:
    """One answer from a `FloorCheckClient`."""

    person_on_floor: bool
    confidence: float


class FloorCheckClient(Protocol):
    """The interface `FloorCheckScheduler` drives. `None` means the call
    failed, timed out, or returned nothing usable -- treated identically to
    "no opinion this round", never as an error to propagate."""

    def check(self, jpeg: bytes) -> FloorCheckResult | None:
        """Return a floor-check answer for `jpeg`, or `None`."""
        ...


def _log_failure(reason: str, **fields: object) -> None:
    """Log a structured warning for a failed floor-check call. Never
    includes frame bytes, matching `perceive.vision._log_failure`."""
    logger.warning(
        json.dumps(
            {
                "service": SERVICE_NAME,
                "message": "floor check call failed, no second opinion this round",
                "reason": reason,
                **fields,
            }
        )
    )


def _first_balanced_object(content: str) -> str | None:
    """Return the first balanced ``{...}`` substring in ``content``.

    Scan from each opening brace so a doubled leading ``{{`` with only one
    closing brace can still recover the valid object beginning at the second
    brace. Braces inside JSON strings do not affect balancing.
    """
    for start, character in enumerate(content):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(content)):
            character = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return content[start : index + 1]
    return None


def _parse_content(content: str) -> object:
    """Parse an exact JSON answer, falling back to its first balanced object."""
    try:
        return json.loads(content)
    except ValueError:
        candidate = _first_balanced_object(content)
        if candidate is None:
            raise
        return json.loads(candidate)


class OllamaFloorCheckClient:
    """Posts one frame to a local Ollama server's `/api/chat` endpoint with
    a JSON schema `format` and `temperature=0`, for a deterministic
    boolean-plus-confidence answer rather than `perceive.vision`'s free-text
    sentence.

    Stateless and synchronous, same as `perceive.vision.OllamaVisionClient`:
    one HTTP POST per `check` call. `FloorCheckScheduler` is what keeps this
    off the frame-processing thread. Never raises into the caller -- any
    transport error, timeout, non-200 response, unparseable body, or
    malformed answer logs a warning and returns `None`.
    """

    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        timeout_seconds: float = 45.0,
        think: bool | None = False,
    ) -> None:
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._think = think

    def check(self, jpeg: bytes) -> FloorCheckResult | None:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": FLOOR_CHECK_PROMPT,
                    "images": [base64.b64encode(jpeg).decode("ascii")],
                }
            ],
            "format": _FLOOR_CHECK_FORMAT,
            "options": {"temperature": 0},
            "stream": False,
        }
        if self._think is not None:
            payload["think"] = self._think
        try:
            response = httpx.post(
                f"{self._ollama_url}{_OLLAMA_CHAT_PATH}",
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

        content = None
        if isinstance(body, dict):
            message = body.get("message")
            if isinstance(message, dict):
                content = message.get("content")
        if not content or not isinstance(content, str):
            _log_failure("response had no usable content")
            return None

        try:
            parsed = _parse_content(content)
        except ValueError as exc:
            _log_failure("content was not JSON", error=str(exc))
            return None

        if (
            not isinstance(parsed, dict)
            or "person_on_floor" not in parsed
            or "confidence" not in parsed
        ):
            _log_failure("content missing required fields")
            return None

        try:
            return FloorCheckResult(
                person_on_floor=bool(parsed["person_on_floor"]),
                confidence=float(parsed["confidence"]),
            )
        except (TypeError, ValueError) as exc:
            _log_failure("content fields had the wrong type", error=str(exc))
            return None


class FakeFloorCheckClient:
    """Returns canned `FloorCheckResult`s (or `None`) in order, for tests.
    No Ollama, no network. Every call is recorded in `calls`. Once
    `responses` is exhausted, further calls return `None`."""

    def __init__(self, responses: list[FloorCheckResult | None] = ()) -> None:
        self._responses = list(responses)
        self.calls: list[bytes] = []

    def check(self, jpeg: bytes) -> FloorCheckResult | None:
        self.calls.append(jpeg)
        if not self._responses:
            return None
        return self._responses.pop(0)


@dataclass
class FloorCheckTrigger:
    """Decides whether the current frame warrants a floor-check vision
    request, and why.

    Reads only read-only signals off `StateTracker`
    (`floor_suspect`, `last_seen_zone`, `seconds_since_last_detection`,
    `last_height_ratio`) plus the frame's own classified state and zone --
    never mutates the tracker. Keeps only one piece of state of its own:
    the timestamp condition (c)'s "persisting >= 3 s" is measured from,
    since that is specific to this trigger, not something `StateTracker`
    itself needs to track for any other purpose.

    Never fires while the current state is `on_floor` (nothing to upgrade)
    or the zone is `bed` or `door`: a person in bed is already covered by
    the pose pipeline's own `in_bed` rules, and near the door is heading
    out, not down.
    """

    low_ratio_persist_seconds: float = 3.0
    low_ratio_threshold: float = 0.75
    lost_max_age_seconds: float = 30.0

    _low_ratio_since: float | None = field(default=None, init=False, repr=False)

    def reason(
        self,
        tracker: StateTracker,
        state: PersonStateName,
        zone: ZoneName,
        now: float,
    ) -> str | None:
        """Return why to fire a check this frame (`"floor_suspect"`,
        `"lost_outside_bed"`, `"low_height_ratio"`), or `None`."""
        if zone in ("bed", "door") or state == "on_floor":
            self._low_ratio_since = None
            return None

        if tracker.floor_suspect is not None:
            # (a) A fall drop was seen just before the person was lost --
            # `StateTracker`'s own suspicion, already bounded by
            # `floor_suspect_seconds`.
            return "floor_suspect"

        if state == "absent":
            # (b) Lost, but recently seen somewhere that is not the bed
            # (and not through the door).
            self._low_ratio_since = None
            last_zone = tracker.last_seen_zone
            since = tracker.seconds_since_last_detection(now)
            if last_zone not in ("bed", "door") and since <= self.lost_max_age_seconds:
                return "lost_outside_bed"
            return None

        if state == "sitting_up":
            # (c) Sitting, outside the bed, low to the ground for a few
            # seconds running -- not just one shaky frame.
            ratio = tracker.last_height_ratio
            if ratio is None or ratio > self.low_ratio_threshold:
                self._low_ratio_since = None
                return None
            if self._low_ratio_since is None:
                self._low_ratio_since = now
            if now - self._low_ratio_since >= self.low_ratio_persist_seconds:
                return "low_height_ratio"
            return None

        self._low_ratio_since = None
        return None


SubmitFn = Callable[[Callable[[], None]], None]


def synchronous_submit(job: Callable[[], None]) -> None:
    """Run `job` immediately, on the calling thread -- what tests use, same
    convention as `perceive.scene_notes.synchronous_submit`."""
    job()


def _thread_submit(job: Callable[[], None]) -> None:
    """The default `submit`: run `job` on a new daemon thread, so a slow or
    hung floor check never blocks process shutdown or the frame loop."""
    Thread(target=job, daemon=True).start()


@dataclass
class FloorCheckScheduler:
    """Fires at most one background `FloorCheckClient.check` call at a time,
    with a cooldown between triggers, and hands the latest completed answer
    back to the caller to act on.

    Mirrors `perceive.scene_notes.SceneNoteCache`'s split deliberately: the
    background thread only ever calls `client.check` and stores the result;
    it never touches `StateTracker` itself. Applying an answer --
    `StateTracker.confirm_floor` and the `PersonState` publish that follows
    -- happens on the main frame-processing thread in `perceive.main.run_once`,
    which is also where the trigger condition is re-checked against the
    frame current when the answer arrives, so nothing here needs a lock
    shared with the tracker.

    A single positive answer is not enough to force `on_floor` (a local
    vision model benchmark found a false positive on a person who had just
    fallen onto the *bed*, not the floor): `required_positives` consecutive
    qualifying answers, on distinct frames at least `follow_up_min_interval_seconds`
    apart, are required while the trigger condition keeps holding --
    tracked here via `note_positive`/`reset_positive_streak`, mutated only
    from the main thread in `run_once`, never from the background job.
    While a streak is running, `maybe_trigger` skips the normal
    `cooldown_seconds` (down to `follow_up_min_interval_seconds`) so the
    follow-up fires as soon as the previous call returns and a new frame is
    available, rather than waiting out the full cooldown again.
    """

    client: FloorCheckClient
    min_confidence: float = 0.6
    cooldown_seconds: float = 10.0
    answer_max_age_seconds: float = 30.0
    required_positives: int = 2
    follow_up_min_interval_seconds: float = 1.0
    submit: SubmitFn = _thread_submit

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _in_flight: bool = field(default=False, init=False, repr=False)
    _last_trigger_at: float | None = field(default=None, init=False, repr=False)
    _pending: tuple[FloorCheckResult | None, str, float, float] | None = field(
        default=None, init=False, repr=False
    )
    """`(result, reason, triggered_at, latency_ms)` of the most recently
    completed check not yet consumed by `take_result`."""

    _positive_streak_count: int = field(default=0, init=False, repr=False)
    """Consecutive qualifying (`person_on_floor=True`, confidence at or
    above `min_confidence`) answers seen so far, with the trigger condition
    still holding each time -- reset by `reset_positive_streak` on any
    negative/low-confidence/stale answer or once the trigger condition
    itself no longer holds. Only ever read or written from the main
    thread; the background job (`_run`) never touches it."""

    def maybe_trigger(self, jpeg: bytes, reason: str, now: float) -> bool:
        """Fire a background check for `jpeg` if warranted, and return
        whether it did. Never fires while a check is already in flight.
        Otherwise gated by `cooldown_seconds` since the last trigger --
        except while a positive streak (`positive_streak_count > 0`) is
        running, when only the much shorter `follow_up_min_interval_seconds`
        applies, so a follow-up check fires promptly."""
        with self._lock:
            if self._in_flight:
                return False
            min_gap = (
                self.follow_up_min_interval_seconds
                if self._positive_streak_count > 0
                else self.cooldown_seconds
            )
            if self._last_trigger_at is not None and (now - self._last_trigger_at) < min_gap:
                return False
            self._in_flight = True
            self._last_trigger_at = now

        self.submit(lambda: self._run(jpeg, reason, now))
        return True

    @property
    def positive_streak_count(self) -> int:
        """How many consecutive qualifying positive answers have been
        recorded so far via `note_positive`, not yet reset. Read-only."""
        return self._positive_streak_count

    def note_positive(self) -> int:
        """Record one qualifying positive answer and return the new streak
        count. Called by `perceive.main.run_once` only after confirming the
        answer qualifies (`person_on_floor=True`, confidence at or above
        `min_confidence`, not stale) and the trigger condition still
        holds."""
        self._positive_streak_count += 1
        return self._positive_streak_count

    def reset_positive_streak(self) -> None:
        """Clear the positive streak: a negative or unqualifying answer, or
        the trigger condition no longer holding, restarts the count from
        zero."""
        self._positive_streak_count = 0

    def _run(self, jpeg: bytes, reason: str, triggered_at: float) -> None:
        """The background job `submit` runs: call the client, time it, store
        the result. Never lets an unexpected exception from `client` escape
        -- a dead vision model must not cost `perceive` anything beyond a
        missing second opinion."""
        start = time.monotonic()
        try:
            result = self.client.check(jpeg)
        except Exception as exc:  # noqa: BLE001 - a client must never take perceive down
            logger.warning(
                json.dumps(
                    {
                        "service": SERVICE_NAME,
                        "message": "floor check client raised, no second opinion this round",
                        "error": str(exc),
                    }
                )
            )
            result = None
        latency_ms = (time.monotonic() - start) * 1000.0

        with self._lock:
            self._pending = (result, reason, triggered_at, latency_ms)
            self._in_flight = False

    def take_result(self) -> tuple[FloorCheckResult | None, str, float, float] | None:
        """Return and clear the latest completed `(result, reason,
        triggered_at, latency_ms)`, or `None` if nothing has completed since
        the last call. Consumed exactly once, like
        `SceneNoteCache._note`'s lock-guarded read -- but popped rather than
        held, since a floor-check answer is acted on once, not repeated on
        every subsequent publish the way a scene note is."""
        with self._lock:
            pending = self._pending
            self._pending = None
        return pending
