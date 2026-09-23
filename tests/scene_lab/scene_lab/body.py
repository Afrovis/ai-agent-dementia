"""Deterministic, frame-driven approximation of the perceive person stream."""

from __future__ import annotations

import random
from collections.abc import Callable

from nc_shared.events import PersonState

STATES = ("in_bed", "sitting_up", "standing", "walking", "on_floor", "absent")
ZONES = ("bed", "door", "bathroom_path", "other")

# A rise out of bed and the return trip get visible intermediate postures.
# Other changes are direct because there is no defensible intermediate pose.
TRANSITIONS = {
    ("in_bed", "walking"): ("sitting_up", "standing", "walking"),
    ("sitting_up", "walking"): ("standing", "walking"),
    ("walking", "in_bed"): ("standing", "sitting_up", "in_bed"),
    ("standing", "in_bed"): ("sitting_up", "in_bed"),
}


class Body:
    def __init__(
        self,
        publish: Callable[[PersonState], None],
        clock: Callable[[], float],
        fps: float = 2.0,
        confirm_frames: int = 3,
        heartbeat_s: float = 60.0,
        noise: dict[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        if fps <= 0 or confirm_frames < 1 or heartbeat_s <= 0:
            raise ValueError("fps, confirm_frames, and heartbeat_s must be positive")
        self.publish = publish
        self.clock = clock
        self.fps = fps
        self.confirm_frames = confirm_frames
        self.heartbeat_s = heartbeat_s
        self.noise = noise or {}
        for value in self.noise.values():
            if not 0 <= value <= 1:
                raise ValueError("noise probabilities must be in [0, 1]")
        self.rng = random.Random(seed)
        self.target_state = "in_bed"
        self.target_zone = "bed"
        self._steps: tuple[str, ...] = ()
        self._move_at = 0.0
        self._move_duration = 0.0
        self._current: str | None = None
        self._pending: str | None = None
        self._pending_count = 0
        self._confidence = 0.95
        self._zone = "bed"
        self._last_published: float | None = None

    def move(self, state: str, zone: str, over_s: float = 0) -> None:
        if state not in STATES or zone not in ZONES or over_s < 0:
            raise ValueError("invalid movement")
        origin = self.target_state
        self.target_state, self.target_zone = state, zone
        self._steps = TRANSITIONS.get((origin, state), (state,))
        self._move_at = self.clock()
        self._move_duration = over_s

    def _intended_state(self, now: float) -> str:
        if not self._move_duration:
            return self.target_state
        progress = max(0.0, (now - self._move_at) / self._move_duration)
        index = min(len(self._steps) - 1, int(progress * len(self._steps)))
        return self._steps[index]

    def tick(self, now: float) -> PersonState | None:
        """Consume one frame and return a published event, if any.

        Cadence follows perceive/main.py lines 373 (tracker update), 496-505
        (publish result with zone), 525-555 (heartbeat snapshot), 662-670
        (last publish or heartbeat), and classify.py 1303-1318 (N frames).
        Confidence threshold is main.py lines 128, 177 and classify.py's
        ClassifyThresholds.min_confidence; low samples cannot confirm state.
        """
        raw = self._intended_state(now)
        if self.rng.random() < self.noise.get("flicker", 0):
            raw = self.rng.choice([state for state in STATES if state != raw])
        confidence = 0.3 if self.rng.random() < self.noise.get("low_confidence", 0) else 0.95
        if confidence < 0.5:
            # A weak pose does not provide a posture vote. Perceive may hold
            # the current reading, but it cannot establish a new one.
            raw = self._current
            if raw is None:
                return None
        if raw == self._current:
            self._pending, self._pending_count = None, 0
        elif raw == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = raw, 1
        changed = self._pending_count >= self.confirm_frames
        if changed:
            self._current = raw
            self._pending, self._pending_count = None, 0
        if self._current is None:
            return None
        # Perceive updates its snapshot zone and confidence on every frame.
        self._confidence, self._zone = confidence, self.target_zone
        due = self._last_published is None or now - self._last_published >= self.heartbeat_s
        if not changed and not due:
            return None
        event = PersonState(
            source="scene_lab",
            state=self._current,
            confidence=self._confidence,
            zone=self._zone,
            scene_note=None,
        )
        self.publish(event)
        self._last_published = now
        return event
