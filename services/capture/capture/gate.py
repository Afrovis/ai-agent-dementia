"""The motion gate: decides which frames `capture` actually publishes.

Issue #7 asks for "2 fps normal, 0.5 fps when static for 30s. Publish
frames only on change." Those two requirements sound like they conflict --
a genuinely static scene has no change to publish -- so `MotionGate`
reconciles them like this: while the scene is moving, admit at
`active_fps`; once `static_seconds` have passed with no motion, drop to a
slow heartbeat at `idle_fps` instead of stopping altogether. `perceive`
needs a periodic frame even from an unchanging room to keep confirming
`in_bed`/`absent`, and a capture service that goes silent on a quiet night
is indistinguishable from a crashed one -- which HANDOFF.md rule 4
explicitly forbids ("a crash must never look like a quiet night"). So
"publish only on change" is read as "publish at the rate motion demands",
with `idle_fps` as the floor, not as "publish only when pixels differ".

Deterministic and clock-free by design: `admit` takes `now` as an explicit
float (unix-epoch-like, but any monotonically increasing float works) so
tests can drive the gate through synthetic time without `time.sleep` or
mocking `time.time`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from capture.motion import frame_signature, motion_score


@dataclass
class MotionGate:
    """Rate-limits and motion-gates a sequence of frames.

    Call `admit(jpeg, now)` for every frame a source produces; it returns
    `True` for frames that should be published (as `Frame` on `frames`) and
    `False` for frames to drop silently. Not thread-safe; one instance per
    source stream.
    """

    active_fps: float = 2.0
    idle_fps: float = 0.5
    static_seconds: float = 30.0
    motion_threshold: float = 0.02

    _last_signature: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _last_published_at: float | None = field(default=None, init=False, repr=False)
    _last_motion_at: float | None = field(default=None, init=False, repr=False)
    _current_fps: float = field(default=0.0, init=False, repr=False)

    def admit(self, jpeg: bytes, now: float) -> bool:
        """Return whether the frame at `jpeg`/`now` should be published.

        The very first frame is always admitted, so a fresh gate (or one
        recovering from a restart) never waits a full period before saying
        anything. After that: compute the frame's signature, compare it to
        the last *published* frame's signature (not merely the last frame
        seen -- comparing against a dropped frame would let motion sneak in
        unnoticed between admits), and use the result to pick the current
        rate (`active_fps` if motion was seen within `static_seconds`,
        `idle_fps` otherwise). The frame is admitted only if enough time has
        passed at that rate since the last publish; faster frames are
        dropped.
        """
        signature = frame_signature(jpeg)

        if self._last_signature is None:
            self._last_signature = signature
            self._last_published_at = now
            self._last_motion_at = now
            self._current_fps = self.active_fps
            return True

        score = motion_score(signature, self._last_signature)
        if score >= self.motion_threshold:
            self._last_motion_at = now

        static_for = now - self._last_motion_at if self._last_motion_at is not None else 0.0
        self._current_fps = self.idle_fps if static_for >= self.static_seconds else self.active_fps

        min_period = 1.0 / self._current_fps
        assert self._last_published_at is not None  # set on the first admitted frame
        if now - self._last_published_at < min_period:
            return False

        self._last_signature = signature
        self._last_published_at = now
        return True

    @property
    def current_fps(self) -> float:
        """The effective publish rate the gate is currently applying, for logging."""
        return self._current_fps
