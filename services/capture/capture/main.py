"""Entry point for the `capture` service (issue #7).

Wires a `FrameSource` (browser bridge, USB camera, or RTSP camera; see
`capture.sources`) through a `capture.gate.MotionGate` and publishes the
frames it admits as `Frame` events on the capped `frames` stream, which is
what `perceive` reads. See HANDOFF.md section 5 for the contract: `capture`
is the sole producer of `frames`.

The loop body is split into small, injectable, single-iteration functions
(`run_once`, `maybe_emit_health`) precisely so `run()` -- the real, infinite,
real-time loop Docker runs -- has nothing left to unit test directly; tests
drive `run_once`/`maybe_emit_health` a bounded number of times against a
`FakeBus` and a `capture.sources.FrameSource` test double instead.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import redis
from nc_shared.bus import Bus
from nc_shared.events import Frame, Health
from nc_shared.replay import CAPPED_MAXLEN

from capture.gate import MotionGate
from capture.sources import FrameSource, build_source

SERVICE_NAME = "capture"
HEALTH_INTERVAL_S = 30.0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, level: int = logging.INFO, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4).

    Never pass frame bytes here: only sizes and dimensions, matching the
    hard rule that camera frames are never logged as bytes.
    """
    logger.log(level, json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


@dataclass(frozen=True)
class CaptureConfig:
    """`capture`'s env-driven configuration (HANDOFF.md section 4: env, then
    yaml, then code defaults; `capture` has no yaml config yet, so env then
    these defaults)."""

    source: str = "browser"
    device: str = "0"
    active_fps: float = 2.0
    idle_fps: float = 0.5
    static_seconds: float = 30.0
    motion_threshold: float = 0.02

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CaptureConfig:
        """Build a `CaptureConfig` from environment variables, defaults otherwise."""
        env = os.environ if env is None else env
        return cls(
            source=env.get("CAPTURE_SOURCE", "browser"),
            device=env.get("CAPTURE_DEVICE", "0"),
            active_fps=float(env.get("CAPTURE_FPS", "2.0")),
            idle_fps=float(env.get("CAPTURE_IDLE_FPS", "0.5")),
            static_seconds=float(env.get("CAPTURE_STATIC_SECONDS", "30")),
            motion_threshold=float(env.get("CAPTURE_MOTION_THRESHOLD", "0.02")),
        )


def build_gate(config: CaptureConfig) -> MotionGate:
    """Build the `MotionGate` described by `config`."""
    return MotionGate(
        active_fps=config.active_fps,
        idle_fps=config.idle_fps,
        static_seconds=config.static_seconds,
        motion_threshold=config.motion_threshold,
    )


def run_once(
    source: FrameSource,
    gate: MotionGate,
    bus,
    *,
    now_fn: Callable[[], float] = time.time,
) -> bool:
    """Read one frame from `source` and publish it as `Frame` if the gate admits it.

    Returns `True` if a `Frame` was published, `False` if the source had
    nothing ready or the gate dropped the frame. Side-effect-free beyond the
    one `bus.publish` call, so tests can call it directly and in a loop with
    a `FakeBus` and a fake `FrameSource`, instead of going through the
    infinite, real-time `run()` loop.
    """
    frame = source.read()
    if frame is None:
        return False
    jpeg, width, height, source_kind = frame

    now = now_fn()
    if not gate.admit(jpeg, now):
        # Dropping is the normal case, not an incident: at 2 fps in from the
        # browser and a 0.5 fps idle rate, three frames in four are dropped on
        # a quiet night. Logged at debug so a night of stillness does not bury
        # the events that matter in tens of thousands of lines.
        _log(
            "dropped frame",
            level=logging.DEBUG,
            event_type="Frame",
            reason="motion_gate",
            fps=round(gate.current_fps, 3),
            source_kind=source_kind,
        )
        return False

    event = Frame(
        source=SERVICE_NAME,
        jpeg=jpeg,
        width=width,
        height=height,
        source_kind=source_kind,
    )
    bus.publish(event, maxlen=CAPPED_MAXLEN["frames"])
    _log(
        "published Frame",
        event_type="Frame",
        width=width,
        height=height,
        jpeg_bytes=len(jpeg),
        fps=round(gate.current_fps, 3),
        source_kind=source_kind,
    )
    return True


def maybe_emit_health(
    bus,
    last_emitted_at: float | None,
    now: float,
    *,
    interval: float = HEALTH_INTERVAL_S,
    ok: bool = True,
    detail: str = "running",
) -> float | None:
    """Publish a `Health` heartbeat if `interval` seconds have passed since the last one.

    Returns the (possibly updated) `last_emitted_at`, so callers thread it
    through their own loop state. `last_emitted_at=None` always emits, which
    covers both "just started" and tests that only care about one emission.
    """
    if last_emitted_at is not None and now - last_emitted_at < interval:
        return last_emitted_at
    bus.publish(Health(source=SERVICE_NAME, service=SERVICE_NAME, ok=ok, detail=detail))
    _log("published Health", event_type="Health", ok=ok, detail=detail)
    return now


def run() -> None:
    """Connect to Redis and the configured source, then loop forever.

    Reads `CAPTURE_SOURCE`, `CAPTURE_DEVICE`, `CAPTURE_FPS`,
    `CAPTURE_IDLE_FPS`, `CAPTURE_STATIC_SECONDS`, and
    `CAPTURE_MOTION_THRESHOLD` from the environment (defaults documented in
    `.env.example`). A short sleep between iterations when a source has
    nothing ready avoids a busy loop without materially affecting latency
    at 2 fps.
    """
    config = CaptureConfig.from_env()
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    _log("capture starting", source=config.source, device=config.device)

    bus = Bus(redis.Redis.from_url(redis_url))
    source = build_source(config.source, config.device, bus)
    gate = build_gate(config)

    last_health_at: float | None = None
    while True:
        published = run_once(source, gate, bus)
        last_health_at = maybe_emit_health(bus, last_health_at, time.time())
        if not published:
            time.sleep(0.05)


if __name__ == "__main__":
    run()
