"""Pure bedside eyes state derived from bus events and a monotonic clock."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from nc_shared.events import (
    Ack,
    Gaze,
    Notify,
    PersonState,
    SessionState,
    Show,
    SpeechStarted,
    Utterance,
)

OVERRIDE_SECONDS = 15.0
"""How long listening (from SpeechStarted) or a Show face override lasts."""


class EyesState:
    """Keep the latest eyes state; emit a message only when it changes.

    A Show listening/speaking override lasts until the next Show with a
    different face, and at most 15 seconds: strategies send these faces but
    no closing Show, so without the cap the eyes could stay "listening" over
    a sleeper all night. The page's own speaking flag covers longer speech.
    SpeechStarted temporarily takes precedence over the override;
    Utterance or a 15-second timeout reveals it again. A subsequent Show
    speaking ends that listening interval.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.posture = "open"
        self.show_override: str | None = None
        self.show_override_at: float | None = None
        self.listening_at: float | None = None
        self.escalated = False
        self.alert = False
        self.alert_acknowledged = False
        self.recent_notifies: deque[tuple[float, str]] = deque()
        self.escalation_ids: set[str] = set()
        self.gaze = {"target": "none", "x": None, "y": None}
        self._state = self._snapshot()

    @property
    def state(self) -> dict:
        return self._state.copy()

    def _snapshot(self) -> dict:
        expression = self.posture
        if self.show_override is not None:
            expression = self.show_override
        if self.listening_at is not None:
            expression = "listening"
        return {
            "type": "eyes",
            "expression": expression,
            "alert": self.alert,
            "gaze": self.gaze.copy(),
        }

    def _changed(self) -> dict | None:
        state = self._snapshot()
        if state == self._state:
            return None
        self._state = state
        return self.state

    def _expire(self, now: float) -> None:
        if self.listening_at is not None and now - self.listening_at >= OVERRIDE_SECONDS:
            self.listening_at = None
        if self.show_override_at is not None and now - self.show_override_at >= OVERRIDE_SECONDS:
            self.show_override = None
            self.show_override_at = None
        while self.recent_notifies and now - self.recent_notifies[0][0] > 10:
            self.recent_notifies.popleft()

    def tick(self, now: float | None = None) -> dict | None:
        """Evaluate timeouts even when no new bus event arrives."""
        self._expire(self.clock() if now is None else now)
        return self._changed()

    def consume(self, event, msg_id: str | None = None, now: float | None = None) -> dict | None:
        """Apply one event. Notify IDs must be their Redis stream message IDs."""
        now = self.clock() if now is None else now
        self._expire(now)
        if isinstance(event, PersonState):
            self.posture = {"in_bed": "sleeping", "sitting_up": "sleepy"}.get(event.state, "open")
        elif isinstance(event, Gaze):
            self.gaze = {"target": event.target, "x": event.x, "y": event.y}
        elif isinstance(event, SpeechStarted):
            self.listening_at = now
        elif isinstance(event, Utterance):
            self.listening_at = None
        elif isinstance(event, Show):
            self.show_override = event.face if event.face in {"listening", "speaking"} else None
            self.show_override_at = now if self.show_override is not None else None
            if event.face == "speaking":
                self.listening_at = None
        elif isinstance(event, SessionState):
            entering = event.phase == "ESCALATED" and not self.escalated
            leaving = event.phase != "ESCALATED" and self.escalated
            if entering:
                self.escalation_ids = {notify_id for _, notify_id in self.recent_notifies}
                self.alert_acknowledged = False
                self.alert = True
            elif leaving:
                self.escalation_ids.clear()
                self.alert = False
                self.alert_acknowledged = False
            self.escalated = event.phase == "ESCALATED"
        elif isinstance(event, Notify) and msg_id is not None:
            self.recent_notifies.append((now, msg_id))
            if self.escalated:
                self.escalation_ids.add(msg_id)
        elif isinstance(event, Ack) and self.escalated and event.notify_id in self.escalation_ids:
            self.alert_acknowledged = True
            self.alert = False
        return self._changed()
