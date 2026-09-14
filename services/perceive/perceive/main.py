"""Entry point for the `perceive` service (issue #8).

Reads `Frame` events `capture` publishes on `frames`, runs them through a
`perceive.backends.PoseBackend` and `perceive.classify.StateTracker`
(caregiver-drawn `perceive.zones.ZoneMap` in between, to decide bed/door/
bathroom-path/other), and publishes `PersonState` on `person` when the
tracker reports a state change, plus a heartbeat at most every
`PERCEIVE_HEARTBEAT_SECONDS` otherwise -- a silent `perceive` must not be
indistinguishable from a calm night (HANDOFF.md rule 4).

Split into small, injectable, single-iteration functions (`run_once`,
`maybe_emit_person_heartbeat`, `maybe_emit_health`,
`perceive.scene_notes.read_session_phase`) the same way `capture.main` is:
`run()` is the real, infinite, real-time loop Docker runs, and has nothing
left to unit test directly. Tests drive `run_once` against a `FakeBus` and
`perceive.backends.ScriptedBackend` -- no camera, no Redis, no model
weights, no Ollama, per HANDOFF.md section 4.

Hard rule, same as `capture`: frames are never written to disk and never
logged as bytes. Only dimensions, state, confidence, and zone are logged.

Issue #9 adds `scene_note`: on a confirmed state change, or every
`PERCEIVE_VISION_INTERVAL_SECONDS` while the last known `SessionState`
(read from the `session` stream; empty until `agent` exists in M2) is
`ENGAGED`, one frame goes to a local Ollama vision model via
`perceive.vision.VisionClient` and the resulting one-sentence note is
attached to subsequent `PersonState` publishes until it goes stale or is
replaced. The call itself runs off the frame-processing path entirely --
see `perceive.scene_notes.SceneNoteCache` -- so a slow or dead vision model
degrades `scene_note` to `None`, never `perceive`'s frame latency.
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
from nc_shared.events import Health, PersonState

from perceive.backends import PoseBackend, build_backend
from perceive.classify import ClassifyThresholds, StateTracker, centroid_of
from perceive.scene_notes import SceneNoteCache, SessionPhaseTracker, read_session_phase
from perceive.vision import OllamaVisionClient, VisionClient
from perceive.zones import ZoneMap, ZoneName, load_zones

SERVICE_NAME = "perceive"
HEALTH_INTERVAL_S = 30.0

FRAME_STREAM = "frames"
FRAME_GROUP = "perceive"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, level: int = logging.INFO, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4).

    Never pass frame bytes here: only dimensions, state, confidence, and
    zone, matching the hard rule that camera frames are never logged as
    bytes.
    """
    logger.log(level, json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


@dataclass(frozen=True)
class PerceiveConfig:
    """`perceive`'s env-driven configuration (HANDOFF.md section 4: env, then
    yaml -- `zones.yaml` is the yaml here -- then defaults in code)."""

    pose_backend: str = "mediapipe"
    yolo_model: str = "yolov8n-pose.pt"
    mediapipe_video_mode: bool = False
    min_confidence: float = 0.5
    presence_confidence: float = 0.25
    floor_top_y: float = 1.01
    absent_confirm_seconds: float = 3.0
    bed_vanish_hold: bool = False
    hold_floor: bool = True
    confirm_frames: int = 3
    bed_hold_seconds: float = 0.0
    walk_threshold: float = 0.15
    heartbeat_seconds: float = 60.0
    zones_path: str | None = None
    vision_enabled: bool = True
    vision_model: str = "moondream"
    vision_interval_seconds: float = 60.0
    vision_timeout_seconds: float = 10.0
    ollama_url: str = "http://host.docker.internal:11434"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> PerceiveConfig:
        """Build a `PerceiveConfig` from environment variables, defaults otherwise."""
        env = os.environ if env is None else env
        return cls(
            pose_backend=env.get("PERCEIVE_POSE_BACKEND", "mediapipe"),
            yolo_model=env.get("PERCEIVE_YOLO_MODEL", "yolov8n-pose.pt"),
            mediapipe_video_mode=(
                env.get("PERCEIVE_MEDIAPIPE_VIDEO_MODE", "false").strip().lower() == "true"
            ),
            min_confidence=float(env.get("PERCEIVE_MIN_CONFIDENCE", "0.5")),
            presence_confidence=float(env.get("PERCEIVE_PRESENCE_CONFIDENCE", "0.25")),
            floor_top_y=float(env.get("PERCEIVE_FLOOR_TOP_Y", "1.01")),
            absent_confirm_seconds=float(env.get("PERCEIVE_ABSENT_CONFIRM_SECONDS", "3")),
            bed_vanish_hold=env.get("PERCEIVE_BED_VANISH_HOLD", "false").strip().lower() == "true",
            hold_floor=env.get("PERCEIVE_HOLD_FLOOR", "true").strip().lower() != "false",
            confirm_frames=int(env.get("PERCEIVE_CONFIRM_FRAMES", "3")),
            bed_hold_seconds=float(env.get("PERCEIVE_BED_HOLD_SECONDS", "0")),
            walk_threshold=float(env.get("PERCEIVE_WALK_THRESHOLD", "0.15")),
            heartbeat_seconds=float(env.get("PERCEIVE_HEARTBEAT_SECONDS", "60")),
            zones_path=env.get("ZONES_PATH"),
            vision_enabled=env.get("PERCEIVE_VISION_ENABLED", "true").strip().lower() != "false",
            vision_model=env.get("PERCEIVE_VISION_MODEL", "moondream"),
            vision_interval_seconds=float(env.get("PERCEIVE_VISION_INTERVAL_SECONDS", "60")),
            vision_timeout_seconds=float(env.get("PERCEIVE_VISION_TIMEOUT_SECONDS", "10")),
            ollama_url=env.get("OLLAMA_URL", "http://host.docker.internal:11434"),
        )


def build_tracker(config: PerceiveConfig) -> StateTracker:
    """Build the `StateTracker` described by `config`."""
    thresholds = ClassifyThresholds(
        min_confidence=config.min_confidence,
        presence_confidence=config.presence_confidence,
        floor_top_y=config.floor_top_y,
        absent_confirm_seconds=config.absent_confirm_seconds,
        bed_vanish_hold=config.bed_vanish_hold,
        hold_floor=config.hold_floor,
        walk_displacement_threshold=config.walk_threshold,
    )
    return StateTracker(
        thresholds=thresholds,
        confirm_frames=config.confirm_frames,
        bed_hold_seconds=config.bed_hold_seconds,
    )


def build_vision_client(config: PerceiveConfig) -> VisionClient | None:
    """Build the configured `VisionClient`, or `None` if vision is disabled.

    `None` is the whole of `PERCEIVE_VISION_ENABLED=false`'s behaviour: no
    `OllamaVisionClient` is ever constructed, so there is no Ollama
    dependency at runtime, and every caller downstream already treats "no
    vision client" the same as "no scene note this round".
    """
    if not config.vision_enabled:
        return None
    return OllamaVisionClient(
        ollama_url=config.ollama_url,
        model=config.vision_model,
        timeout_seconds=config.vision_timeout_seconds,
    )


def run_once(
    bus,
    backend: PoseBackend,
    zones: ZoneMap,
    tracker: StateTracker,
    *,
    consumer: str = "perceive-1",
    count: int = 1,
    block_ms: int = 200,
    now_fn: Callable[[], float] = time.time,
    scene_cache: SceneNoteCache | None = None,
    engaged: bool = False,
) -> PersonState | None:
    """Read one `Frame`, classify it, and publish `PersonState` if the tracker
    reports a change.

    Returns the published event, or `None` if there was nothing to read or
    the tracker is still waiting on hysteresis to confirm a change --
    either way, side-effect-free beyond the one `bus.publish` call (and,
    when `scene_cache` is given, at most one non-blocking vision request),
    so tests can call it directly and in a loop with a `FakeBus` and a
    `perceive.backends.ScriptedBackend` instead of going through the
    infinite, real-time `run()` loop.

    Acks the `Frame` message as soon as it has been decoded into a pose:
    nothing downstream of `perceive` reads `frames` again, so there is
    nothing left to redeliver it for, mirroring
    `capture.sources.BrowserBusSource`.

    When `scene_cache` is given (`None` when `PERCEIVE_VISION_ENABLED=false`,
    per `build_vision_client`), this frame is offered to
    `scene_cache.maybe_request` keyed on the tracker's confirmed current
    state -- the same state a `PersonState` publish would carry -- so a
    vision request fires exactly on a confirmed state change or, with
    `engaged=True`, on the configured interval; either way `maybe_request`
    never blocks this function. The latest completed note, if any and not
    stale, is attached to every `PersonState` this function publishes.
    """
    messages = bus.read(FRAME_STREAM, FRAME_GROUP, consumer, count=count, block_ms=block_ms)
    if not messages:
        return None
    msg_id, frame = messages[0]
    bus.ack(FRAME_STREAM, FRAME_GROUP, msg_id)

    pose = backend.detect(frame.jpeg)

    zone: ZoneName = "other"
    if pose is not None:
        centroid_x, centroid_y = centroid_of(pose)
        zone = zones.zone_for_point(centroid_x, centroid_y)

    now = now_fn()
    result = tracker.update(pose, zone, now)

    # One line per frame would be ~170k lines a night at the active rate.
    # The PersonState publishes below carry what actually matters.
    _log(
        "classified frame",
        level=logging.DEBUG,
        event_type="Frame",
        width=frame.width,
        height=frame.height,
        detected=pose is not None,
        zone=zone,
    )

    if scene_cache is not None:
        snapshot = tracker.snapshot()
        if snapshot is not None:
            current_state, _confidence, _zone = snapshot
            scene_cache.maybe_request(frame.jpeg, current_state, now, engaged=engaged)

    if result is None:
        return None

    state, confidence = result
    scene_note = scene_cache.current(now) if scene_cache is not None else None
    event = PersonState(
        source=SERVICE_NAME,
        state=state,
        confidence=confidence,
        zone=zone,
        scene_note=scene_note,
    )
    bus.publish(event)
    _log(
        "published PersonState",
        event_type="PersonState",
        state=state,
        confidence=round(confidence, 3),
        zone=zone,
    )
    return event


def maybe_emit_person_heartbeat(
    bus,
    tracker: StateTracker,
    last_emitted_at: float | None,
    now: float,
    *,
    interval: float,
    scene_cache: SceneNoteCache | None = None,
) -> float | None:
    """Publish a heartbeat `PersonState` -- repeating the tracker's last known
    state, confidence, and zone -- if `interval` seconds have passed with
    nothing new to report.

    Without this, a quiet, uneventful night and a crashed `perceive`
    process both look like silence on the `person` stream, which HANDOFF.md
    rule 4 forbids ("a crash must never look like a quiet night"). Returns
    the (possibly updated) `last_emitted_at`, threaded through by the
    caller like `capture.main.maybe_emit_health`.

    Carries `scene_cache.current(now)` (or `None` if `scene_cache` is
    `None`, or the last note is stale) the same way `run_once` does, so a
    completed scene note keeps showing up on every publish -- including
    the heartbeat -- until it goes stale or is replaced.
    """
    if last_emitted_at is not None and now - last_emitted_at < interval:
        return last_emitted_at

    snapshot = tracker.snapshot()
    if snapshot is None:
        return last_emitted_at  # nothing classified yet; nothing to repeat

    state, confidence, zone = snapshot
    event = PersonState(
        source=SERVICE_NAME,
        state=state,
        confidence=confidence,
        zone=zone,
        scene_note=scene_cache.current(now) if scene_cache is not None else None,
    )
    bus.publish(event)
    _log(
        "published PersonState heartbeat",
        event_type="PersonState",
        state=state,
        confidence=round(confidence, 3),
        zone=zone,
    )
    return now


def maybe_emit_health(
    bus,
    last_emitted_at: float | None,
    now: float,
    *,
    interval: float = HEALTH_INTERVAL_S,
    ok: bool = True,
    detail: str = "running",
) -> float | None:
    """Publish a `Health` heartbeat if `interval` seconds have passed since the
    last one. Same shape as `capture.main.maybe_emit_health`."""
    if last_emitted_at is not None and now - last_emitted_at < interval:
        return last_emitted_at
    bus.publish(Health(source=SERVICE_NAME, service=SERVICE_NAME, ok=ok, detail=detail))
    _log("published Health", event_type="Health", ok=ok, detail=detail)
    return now


def run() -> None:
    """Connect to Redis, build the configured pose backend and zones, then
    loop forever.

    Reads `PERCEIVE_POSE_BACKEND`, `PERCEIVE_YOLO_MODEL`,
    `PERCEIVE_MEDIAPIPE_VIDEO_MODE`, `PERCEIVE_MIN_CONFIDENCE`,
    `PERCEIVE_PRESENCE_CONFIDENCE`, `PERCEIVE_CONFIRM_FRAMES`, `PERCEIVE_BED_HOLD_SECONDS`,
    `PERCEIVE_WALK_THRESHOLD`, `PERCEIVE_HEARTBEAT_SECONDS`, `ZONES_PATH`,
    `PERCEIVE_VISION_ENABLED`, `PERCEIVE_VISION_MODEL`,
    `PERCEIVE_VISION_INTERVAL_SECONDS`, `PERCEIVE_VISION_TIMEOUT_SECONDS`,
    and `OLLAMA_URL` from the environment (defaults documented in
    `.env.example`). A short sleep between iterations when nothing was
    published avoids a busy loop.
    """
    config = PerceiveConfig.from_env()
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    _log(
        "perceive starting", pose_backend=config.pose_backend, vision_enabled=config.vision_enabled
    )

    bus = Bus(redis.Redis.from_url(redis_url))
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = build_backend(
        config.pose_backend,
        model_path=config.yolo_model,
        static_image_mode=not config.mediapipe_video_mode,
    )
    zones = load_zones(config.zones_path)
    tracker = build_tracker(config)

    vision_client = build_vision_client(config)
    scene_cache = (
        SceneNoteCache(vision_client=vision_client, interval_seconds=config.vision_interval_seconds)
        if vision_client is not None
        else None
    )
    session_tracker = SessionPhaseTracker()

    last_health_at: float | None = None
    last_heartbeat_at: float | None = None
    while True:
        read_session_phase(bus, session_tracker)
        published = run_once(
            bus,
            backend,
            zones,
            tracker,
            scene_cache=scene_cache,
            engaged=session_tracker.engaged,
        )
        now = time.time()
        if published is not None:
            last_heartbeat_at = now
        else:
            last_heartbeat_at = maybe_emit_person_heartbeat(
                bus,
                tracker,
                last_heartbeat_at,
                now,
                interval=config.heartbeat_seconds,
                scene_cache=scene_cache,
            )
        last_health_at = maybe_emit_health(bus, last_health_at, now)
        if published is None:
            time.sleep(0.05)


if __name__ == "__main__":
    run()
