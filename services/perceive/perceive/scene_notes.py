"""Non-blocking scheduling for vision-LLM scene notes (issue #9).

A vision call takes on the order of a second or more -- `perceive` consumes
frames at up to 2 fps and must not stall behind it, or it falls behind the
`frames` stream and blows the 2 s latency target in PLAN.md section 12.
`SceneNoteCache` is what keeps `perceive.main.run_once` from ever blocking
on `perceive.vision.VisionClient.describe`: it fires at most one request in
the background and hands back whatever the last completed one said, never
waiting on the call itself.

`SessionPhaseTracker` is the other half of "every 60 s while `ENGAGED`":
`agent` publishes `SessionState` on the `session` stream (HANDOFF.md
section 5), but `agent` is not built until M2, so that stream is empty for
now -- the tracker simply stays at its `IDLE` default and the state-change
trigger above is what actually fires `scene_note` requests today. It is
still wired up properly, under its own consumer group, so nothing needs to
change here once `agent` starts publishing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock, Thread

from perceive.vision import VisionClient

SERVICE_NAME = "perceive"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

SESSION_STREAM = "session"
SESSION_GROUP = "perceive-session"

SCENE_NOTE_MAX_AGE_SECONDS = 90.0
"""A `scene_note` older than this is treated as if there were none at all.

`perceive` publishes a heartbeat `PersonState` every
`PERCEIVE_HEARTBEAT_SECONDS` (60 by default) even when nothing changed, so
without a staleness check a stalled or slow vision model would leave a
description from minutes -- or, over a quiet stretch of the night, much
longer -- ago attached to every publish after it, misrepresenting what is
happening in the room right now. 90 s is one and a half heartbeat
intervals at the default: generous enough that a single slow Ollama call
does not immediately blank the note, short enough that a `scene_note`
never describes a scene from a materially different moment of the night.
"""

_NO_STATE = object()
"""Sentinel for "no request has ever been made" -- distinct from any real
`PersonState.state` string, so the very first observed state always counts
as a change and fires a request, per PLAN.md 6.1's "when the state
changes"."""

SubmitFn = Callable[[Callable[[], None]], None]


def synchronous_submit(job: Callable[[], None]) -> None:
    """Run `job` immediately, on the calling thread.

    The injected-executor hook tests use: fully synchronous, no real
    threads, no sleeping, no flakiness -- a `SceneNoteCache` built with this
    as `submit` behaves exactly like one running a real background thread
    except that `maybe_request` does not return until the "background" call
    has finished.
    """
    job()


def _thread_submit(job: Callable[[], None]) -> None:
    """The default `submit`: run `job` on a new daemon thread.

    Daemon so a slow or hung vision call never blocks process shutdown --
    matching HANDOFF.md rule 4's "fail quiet" posture: losing the vision
    thread at shutdown must not be the thing that keeps `perceive` from
    exiting.
    """
    Thread(target=job, daemon=True).start()


@dataclass
class SceneNoteCache:
    """Fires at most one background `VisionClient.describe` call at a time
    and remembers the latest completed result.

    `maybe_request` never blocks: it decides whether to fire, and if so
    hands the work to `submit` and returns immediately. A frame arriving
    while a call is already in flight is skipped outright, not queued --
    there is always at most one request outstanding, by construction, so
    `perceive` never falls behind the vision model even if it is slower
    than the frame rate.
    """

    vision_client: VisionClient
    interval_seconds: float = 60.0
    submit: SubmitFn = _thread_submit

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _in_flight: bool = field(default=False, init=False, repr=False)
    _last_requested_state: object = field(default=_NO_STATE, init=False, repr=False)
    _last_request_at: float | None = field(default=None, init=False, repr=False)
    _note: str | None = field(default=None, init=False, repr=False)
    _note_frame_at: float | None = field(default=None, init=False, repr=False)
    """When the frame the current note describes was *taken*, not when the
    call returned. Age is what makes a note stale, and a note describes the
    moment it saw, so a slow model must not buy its own output extra life."""

    def maybe_request(self, jpeg: bytes, state: str, now: float, *, engaged: bool = False) -> bool:
        """Fire a background vision request for `jpeg` if warranted, and
        return whether it did.

        Fires when `state` differs from the state the last request was made
        for, or when `engaged` is true and `interval_seconds` have passed
        since the last request -- whichever condition is met first. Never
        fires while a request is already in flight.
        """
        with self._lock:
            if self._in_flight:
                return False

            state_changed = state != self._last_requested_state
            interval_elapsed = (
                engaged
                and self._last_request_at is not None
                and (now - self._last_request_at) >= self.interval_seconds
            )
            if not (state_changed or interval_elapsed):
                return False

            self._in_flight = True
            self._last_requested_state = state
            self._last_request_at = now

        self.submit(lambda: self._run(jpeg, now))
        return True

    def _run(self, jpeg: bytes, requested_at: float) -> None:
        """The background job `submit` runs: call the vision client, store
        the result. Never lets an unexpected exception from `vision_client`
        escape -- a dead vision model must not cost `perceive` anything
        more than a missing `scene_note` (HANDOFF.md rule 4)."""
        try:
            note = self.vision_client.describe(jpeg)
        except Exception as exc:  # noqa: BLE001 - a vision client must never take perceive down
            logger.warning(
                json.dumps(
                    {
                        "service": SERVICE_NAME,
                        "message": "vision client raised, scene_note stays unset",
                        "error": str(exc),
                    }
                )
            )
            note = None

        with self._lock:
            self._note = note
            self._note_frame_at = requested_at
            self._in_flight = False

    def current(self, now: float) -> str | None:
        """Return the latest completed scene note, or `None` if there isn't
        one yet or it is older than `SCENE_NOTE_MAX_AGE_SECONDS`."""
        with self._lock:
            note = self._note
            frame_at = self._note_frame_at
        if note is None or frame_at is None:
            return None
        if (now - frame_at) > SCENE_NOTE_MAX_AGE_SECONDS:
            return None
        return note


@dataclass
class SessionPhaseTracker:
    """Tracks the latest `SessionState.phase` seen on the `session` stream.

    Defaults to `IDLE` when nothing has ever been published -- true for
    every night until `agent` (M2) exists. `engaged` is the only thing
    `perceive` actually asks of this: whether the periodic "every 60 s"
    vision trigger in PLAN.md 6.1 should be armed.
    """

    phase: str = "IDLE"
    group_ready: bool = False
    """Whether the `session` consumer group has been created yet. `run()`
    calls `read_session_phase` on every loop iteration, so without this the
    service would issue a redundant XGROUP CREATE against Redis roughly
    twenty times a second for the whole night."""

    @property
    def engaged(self) -> bool:
        return self.phase == "ENGAGED"


def read_session_phase(
    bus,
    tracker: SessionPhaseTracker,
    *,
    consumer: str = "perceive-1",
    count: int = 10,
    block_ms: int = 1,
) -> None:
    """Read whatever is newly available on the `session` stream and update
    `tracker` with the latest `SessionState.phase` seen, if any.

    A short, effectively non-blocking read (`block_ms=1`, not `0`, which
    would block a real Redis connection indefinitely): `perceive.main.run()`
    calls this once per loop iteration alongside `run_once`, and reading
    `session` must never be what makes the frame loop miss PLAN.md section
    12's 2 s latency target. The stream also carries `GoalChanged`, which
    this ignores; every message read is acked regardless of type, since
    nothing else reads `session` on this consumer group.
    """
    if not tracker.group_ready:
        bus.ensure_group(SESSION_STREAM, SESSION_GROUP)
        tracker.group_ready = True
    messages = bus.read(SESSION_STREAM, SESSION_GROUP, consumer, count=count, block_ms=block_ms)
    for msg_id, event in messages:
        bus.ack(SESSION_STREAM, SESSION_GROUP, msg_id)
        phase = getattr(event, "phase", None)
        if phase is not None:
            tracker.phase = phase
