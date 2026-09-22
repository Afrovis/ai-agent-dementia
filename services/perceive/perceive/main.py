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

A second, distinct local vision call -- `perceive.floor_check` -- can
*upgrade* an ambiguous frame to `on_floor` (a fall drop then lost, a
person lost outside the bed, or a low, sustained height ratio while
`sitting_up` outside the bed), but never vetoes or blocks on a
pose-confirmed one: it runs the same non-blocking, at-most-one-in-flight
way `scene_note` does, and only applies through
`perceive.classify.StateTracker.confirm_floor` after re-checking its
trigger condition against the frame current when the answer arrives.
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
from nc_shared.events import Health, PersonState, PoseDebug

from perceive.backends import PoseBackend, build_backend
from perceive.classify import (
    ClassifyThresholds,
    StateTracker,
    ground_zone_for_pose,
    zone_for_pose,
)
from perceive.floor_check import (
    FloorCheckClient,
    FloorCheckScheduler,
    FloorCheckTrigger,
    OllamaFloorCheckClient,
)
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


def _parse_ground_line(value: str) -> tuple[float, float] | None:
    """Parse `PERCEIVE_GROUND_LINE="a,b"` into a `(a, b)` override for
    `perceive.classify.StateTracker.ground_line`, or `None` for the default
    empty string -- online self-calibration stays in charge then."""
    value = value.strip()
    if not value:
        return None
    try:
        a_str, b_str = value.split(",", 1)
        return float(a_str), float(b_str)
    except ValueError:
        _log("invalid PERCEIVE_GROUND_LINE, ignoring", level=logging.WARNING, value=value)
        return None


def _parse_optional_bool(value: str | None, *, default: bool) -> bool | None:
    """Parse an optional env bool; empty, ``auto``, or ``unset`` means omit."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "auto", "unset"}:
        return None
    return normalized == "true"


@dataclass(frozen=True)
class PerceiveConfig:
    """`perceive`'s env-driven configuration (HANDOFF.md section 4: env, then
    yaml -- `zones.yaml` is the yaml here -- then defaults in code)."""

    pose_backend: str = "yolo"
    yolo_model: str = "yolo11s-pose.pt"
    yolo_imgsz: int = 640
    phantoms_file: str | None = None
    phantom_max_confidence: float = 0.7
    mediapipe_video_mode: bool = False
    min_confidence: float = 0.5
    presence_confidence: float = 0.25
    floor_top_y: float = 1.01
    absent_confirm_seconds: float = 3.0
    bed_vanish_hold: bool = False
    sitting_thigh_ratio: float = 0.0
    hold_floor: bool = True
    confirm_frames: int = 3
    bed_hold_seconds: float = 0.0
    walk_threshold: float = 0.15
    walk_mode: str = "legacy"
    walk_motion_threshold: float = 0.35
    walk_window_seconds: float = 2.0
    bed_latch: bool = False
    upright_zone_from_feet: bool = False
    heartbeat_seconds: float = 60.0
    zones_path: str | None = None
    vision_enabled: bool = True
    vision_model: str = "moondream"
    vision_interval_seconds: float = 60.0
    vision_timeout_seconds: float = 10.0
    ollama_url: str = "http://host.docker.internal:11434"
    floor_height_ratio: float = 0.6
    fall_window_seconds: float = 2.5
    fall_drop: float = 0.15
    floor_suspect_seconds: float = 20.0
    ground_line: tuple[float, float] | None = None
    ground_line_file: str | None = None
    floor_check_enabled: bool = True
    floor_check_model: str = "gemma4:e4b-mlx"
    floor_check_think: bool | None = False
    floor_check_timeout_seconds: float = 20.0
    floor_check_cooldown_seconds: float = 10.0
    floor_check_min_confidence: float = 0.6
    floor_check_required_positives: int = 2

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> PerceiveConfig:
        """Build a `PerceiveConfig` from environment variables, defaults otherwise."""
        env = os.environ if env is None else env
        return cls(
            pose_backend=env.get("PERCEIVE_POSE_BACKEND", "yolo"),
            yolo_model=env.get("PERCEIVE_YOLO_MODEL", "yolo11s-pose.pt"),
            yolo_imgsz=int(env.get("PERCEIVE_YOLO_IMGSZ", "640")),
            phantoms_file=env.get("PERCEIVE_PHANTOMS_FILE", "").strip() or None,
            phantom_max_confidence=float(env.get("PERCEIVE_PHANTOM_MAX_CONFIDENCE", "0.7")),
            mediapipe_video_mode=(
                env.get("PERCEIVE_MEDIAPIPE_VIDEO_MODE", "false").strip().lower() == "true"
            ),
            min_confidence=float(env.get("PERCEIVE_MIN_CONFIDENCE", "0.5")),
            presence_confidence=float(env.get("PERCEIVE_PRESENCE_CONFIDENCE", "0.25")),
            floor_top_y=float(env.get("PERCEIVE_FLOOR_TOP_Y", "1.01")),
            absent_confirm_seconds=float(env.get("PERCEIVE_ABSENT_CONFIRM_SECONDS", "3")),
            bed_vanish_hold=env.get("PERCEIVE_BED_VANISH_HOLD", "false").strip().lower() == "true",
            sitting_thigh_ratio=float(env.get("PERCEIVE_SITTING_THIGH_RATIO", "0")),
            hold_floor=env.get("PERCEIVE_HOLD_FLOOR", "true").strip().lower() != "false",
            confirm_frames=int(env.get("PERCEIVE_CONFIRM_FRAMES", "3")),
            bed_hold_seconds=float(env.get("PERCEIVE_BED_HOLD_SECONDS", "0")),
            walk_threshold=float(env.get("PERCEIVE_WALK_THRESHOLD", "0.15")),
            walk_mode=env.get("PERCEIVE_WALK_MODE", "legacy").strip().lower() or "legacy",
            walk_motion_threshold=float(env.get("PERCEIVE_WALK_MOTION_THRESHOLD", "0.35")),
            walk_window_seconds=float(env.get("PERCEIVE_WALK_WINDOW_SECONDS", "2")),
            bed_latch=env.get("PERCEIVE_BED_LATCH", "false").strip().lower() == "true",
            upright_zone_from_feet=(
                env.get("PERCEIVE_UPRIGHT_ZONE_FROM_FEET", "false").strip().lower() == "true"
            ),
            heartbeat_seconds=float(env.get("PERCEIVE_HEARTBEAT_SECONDS", "60")),
            zones_path=env.get("ZONES_PATH"),
            vision_enabled=env.get("PERCEIVE_VISION_ENABLED", "true").strip().lower() != "false",
            vision_model=env.get("PERCEIVE_VISION_MODEL", "moondream"),
            vision_interval_seconds=float(env.get("PERCEIVE_VISION_INTERVAL_SECONDS", "60")),
            vision_timeout_seconds=float(env.get("PERCEIVE_VISION_TIMEOUT_SECONDS", "10")),
            ollama_url=env.get("OLLAMA_URL", "http://host.docker.internal:11434"),
            floor_height_ratio=float(env.get("PERCEIVE_FLOOR_HEIGHT_RATIO", "0.6")),
            fall_window_seconds=float(env.get("PERCEIVE_FALL_WINDOW_SECONDS", "2.5")),
            fall_drop=float(env.get("PERCEIVE_FALL_DROP", "0.15")),
            floor_suspect_seconds=float(env.get("PERCEIVE_FLOOR_SUSPECT_SECONDS", "20")),
            ground_line=_parse_ground_line(env.get("PERCEIVE_GROUND_LINE", "")),
            ground_line_file=env.get("PERCEIVE_GROUND_LINE_FILE") or None,
            floor_check_enabled=(
                env.get("PERCEIVE_FLOOR_CHECK_ENABLED", "true").strip().lower() != "false"
            ),
            floor_check_model=env.get("PERCEIVE_FLOOR_CHECK_MODEL", "gemma4:e4b-mlx"),
            floor_check_think=_parse_optional_bool(
                env.get("PERCEIVE_FLOOR_CHECK_THINK"), default=False
            ),
            floor_check_timeout_seconds=float(
                env.get("PERCEIVE_FLOOR_CHECK_TIMEOUT_SECONDS", "20")
            ),
            floor_check_cooldown_seconds=float(
                env.get("PERCEIVE_FLOOR_CHECK_COOLDOWN_SECONDS", "10")
            ),
            floor_check_min_confidence=float(env.get("PERCEIVE_FLOOR_CHECK_MIN_CONFIDENCE", "0.6")),
            floor_check_required_positives=int(
                env.get("PERCEIVE_FLOOR_CHECK_REQUIRED_POSITIVES", "2")
            ),
        )


def build_tracker(config: PerceiveConfig) -> StateTracker:
    """Build the `StateTracker` described by `config`."""
    thresholds = ClassifyThresholds(
        min_confidence=config.min_confidence,
        presence_confidence=config.presence_confidence,
        floor_top_y=config.floor_top_y,
        absent_confirm_seconds=config.absent_confirm_seconds,
        bed_vanish_hold=config.bed_vanish_hold,
        sitting_thigh_ratio=config.sitting_thigh_ratio,
        hold_floor=config.hold_floor,
        walk_displacement_threshold=config.walk_threshold,
        walk_mode=config.walk_mode,
        walk_motion_threshold=config.walk_motion_threshold,
        walk_window_seconds=config.walk_window_seconds,
        bed_latch=config.bed_latch,
        upright_zone_from_feet=config.upright_zone_from_feet,
        floor_height_ratio=config.floor_height_ratio,
        fall_window_seconds=config.fall_window_seconds,
        fall_drop=config.fall_drop,
        floor_suspect_seconds=config.floor_suspect_seconds,
    )
    return StateTracker(
        thresholds=thresholds,
        confirm_frames=config.confirm_frames,
        bed_hold_seconds=config.bed_hold_seconds,
        ground_line=config.ground_line,
        ground_line_file=config.ground_line_file,
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


def build_floor_check_client(config: PerceiveConfig) -> FloorCheckClient | None:
    """Build the configured `FloorCheckClient`, or `None` if the floor
    check is disabled. Same shape as `build_vision_client`: `None` means no
    `OllamaFloorCheckClient` is ever constructed, so there is no Ollama
    dependency at runtime, and `run_once` already treats "no floor check
    client" the same as "no second opinion this round"."""
    if not config.floor_check_enabled:
        return None
    return OllamaFloorCheckClient(
        ollama_url=config.ollama_url,
        model=config.floor_check_model,
        timeout_seconds=config.floor_check_timeout_seconds,
        think=config.floor_check_think,
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
    floor_scheduler: FloorCheckScheduler | None = None,
    floor_trigger: FloorCheckTrigger | None = None,
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

    When `floor_scheduler`/`floor_trigger` are both given (`None` when
    `PERCEIVE_FLOOR_CHECK_ENABLED=false`, per `build_floor_check_client`),
    this function does two more things, neither of which blocks it: it
    asks `floor_trigger.reason(...)` whether the tracker's now-current
    state warrants a vision second opinion and, if so, offers this frame to
    `floor_scheduler.maybe_trigger` (a no-op if one is already in flight, or
    the cooldown -- or, mid-streak, the shorter follow-up interval -- has
    not elapsed); if the trigger condition does *not* currently hold, any
    running positive streak is reset. It also collects any check that
    completed since the last call via `floor_scheduler.take_result()`. A
    completed answer only counts towards the streak
    (`floor_scheduler.note_positive()`) if it says `person_on_floor=True` at
    or above `floor_scheduler.min_confidence`, the answer is not older than
    `floor_scheduler.answer_max_age_seconds`, and `floor_trigger.reason(...)`
    says the trigger condition still holds against the frame current *now*,
    not the one the check was fired against -- any other answer resets the
    streak. Only once `floor_scheduler.required_positives` consecutive
    qualifying answers have accumulated does this apply --
    `tracker.confirm_floor` and an immediate `PersonState` publish,
    `scene_note` marked `"vision: person on floor"` so downstream can tell
    it did not come from pose -- and reset the streak: the vision model can
    only ever upgrade an ambiguous frame, never veto a pose-based
    `on_floor` (already published without waiting) or apply itself against
    a situation that has since resolved or a single, possibly mistaken,
    answer. One structured log line covers every completed check, whether
    or not it was applied, and never includes frame bytes.
    """
    messages = bus.read(FRAME_STREAM, FRAME_GROUP, consumer, count=count, block_ms=block_ms)
    if not messages:
        return None
    msg_id, frame = messages[0]
    bus.ack(FRAME_STREAM, FRAME_GROUP, msg_id)

    started = time.perf_counter()
    pose = backend.detect(frame.jpeg)

    zone: ZoneName = "other"
    ground_zone: ZoneName | None = None
    if pose is not None:
        zone = zone_for_pose(zones, pose)
        ground_zone = ground_zone_for_pose(zones, pose)

    now = now_fn()
    result = tracker.update(pose, zone, now, ground_zone=ground_zone)

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

    floor_check_event: PersonState | None = None
    if floor_scheduler is not None and floor_trigger is not None:
        floor_snapshot = tracker.snapshot()
        if floor_snapshot is not None:
            floor_state, _floor_confidence, floor_zone = floor_snapshot
            trigger_reason = floor_trigger.reason(tracker, floor_state, floor_zone, now)
            if trigger_reason is not None:
                floor_scheduler.maybe_trigger(frame.jpeg, trigger_reason, now)
            else:
                # The trigger condition itself no longer holds -- any
                # positive streak in progress no longer means anything.
                floor_scheduler.reset_positive_streak()

        completed = floor_scheduler.take_result()
        if completed is not None:
            vision_result, reason, triggered_at, latency_ms = completed
            applied = False
            streak_count = floor_scheduler.positive_streak_count
            answer_log: dict[str, object] | None = None
            if vision_result is not None:
                answer_log = {
                    "person_on_floor": vision_result.person_on_floor,
                    "confidence": round(vision_result.confidence, 3),
                }
            qualifies = (
                vision_result is not None
                and vision_result.person_on_floor
                and vision_result.confidence >= floor_scheduler.min_confidence
                and (now - triggered_at) <= floor_scheduler.answer_max_age_seconds
            )
            still_holds = False
            if qualifies:
                recheck_snapshot = tracker.snapshot()
                if recheck_snapshot is not None:
                    recheck_state, _recheck_confidence, recheck_zone = recheck_snapshot
                    still_holds = (
                        floor_trigger.reason(tracker, recheck_state, recheck_zone, now) is not None
                    )
            if qualifies and still_holds:
                streak_count = floor_scheduler.note_positive()
                if streak_count >= floor_scheduler.required_positives:
                    confirmed = tracker.confirm_floor(
                        now, confidence=vision_result.confidence, source="vision"
                    )
                    floor_scheduler.reset_positive_streak()
                    if confirmed is not None:
                        applied = True
                        confirmed_state, confirmed_confidence = confirmed
                        floor_check_event = PersonState(
                            source=SERVICE_NAME,
                            state=confirmed_state,
                            confidence=confirmed_confidence,
                            zone=recheck_zone,
                            scene_note="vision: person on floor",
                        )
                        bus.publish(floor_check_event)
                        _log(
                            "published PersonState",
                            event_type="PersonState",
                            state=confirmed_state,
                            confidence=round(confirmed_confidence, 3),
                            zone=recheck_zone,
                        )
            else:
                floor_scheduler.reset_positive_streak()
            _log(
                "floor check",
                event_type="FloorCheck",
                reason=reason,
                latency_ms=round(latency_ms, 1),
                answer=answer_log,
                positive_streak_count=streak_count,
                applied=applied,
            )

    snapshot = tracker.snapshot()
    tracked_state = snapshot[0] if snapshot is not None else None
    bus.publish(
        PoseDebug(
            source=SERVICE_NAME,
            landmarks=(
                {
                    name: (point.x, point.y, point.visibility)
                    for name, point in pose.landmarks.items()
                }
                if pose is not None
                else {}
            ),
            bbox=pose.bbox if pose is not None else None,
            confidence=pose.confidence if pose is not None else 0.0,
            detected=pose is not None,
            candidate_state=tracked_state,
            state=tracked_state,
            zone=snapshot[2] if snapshot is not None else None,
            frame_ts=frame.ts.isoformat(),
            latency_ms=(time.perf_counter() - started) * 1000,
        ),
        maxlen=50,
    )

    if result is None:
        return floor_check_event

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

    Reads `PERCEIVE_POSE_BACKEND`, `PERCEIVE_YOLO_MODEL`, `PERCEIVE_YOLO_IMGSZ`,
    `PERCEIVE_PHANTOMS_FILE`, `PERCEIVE_PHANTOM_MAX_CONFIDENCE`,
    `PERCEIVE_MEDIAPIPE_VIDEO_MODE`, `PERCEIVE_MIN_CONFIDENCE`,
    `PERCEIVE_PRESENCE_CONFIDENCE`, `PERCEIVE_CONFIRM_FRAMES`, `PERCEIVE_BED_HOLD_SECONDS`,
    `PERCEIVE_WALK_THRESHOLD`, `PERCEIVE_HEARTBEAT_SECONDS`, `ZONES_PATH`,
    `PERCEIVE_VISION_ENABLED`, `PERCEIVE_VISION_MODEL`,
    `PERCEIVE_VISION_INTERVAL_SECONDS`, `PERCEIVE_VISION_TIMEOUT_SECONDS`,
    `OLLAMA_URL`, `PERCEIVE_FLOOR_HEIGHT_RATIO`, `PERCEIVE_FALL_WINDOW_SECONDS`,
    `PERCEIVE_FALL_DROP`, `PERCEIVE_FLOOR_SUSPECT_SECONDS`,
    `PERCEIVE_GROUND_LINE`, `PERCEIVE_GROUND_LINE_FILE`,
    `PERCEIVE_FLOOR_CHECK_ENABLED`, `PERCEIVE_FLOOR_CHECK_MODEL`,
    `PERCEIVE_FLOOR_CHECK_THINK`,
    `PERCEIVE_FLOOR_CHECK_TIMEOUT_SECONDS`,
    `PERCEIVE_FLOOR_CHECK_COOLDOWN_SECONDS`,
    `PERCEIVE_FLOOR_CHECK_MIN_CONFIDENCE` and
    `PERCEIVE_FLOOR_CHECK_REQUIRED_POSITIVES` from the environment (defaults
    documented in `.env.example`). A short sleep between iterations when
    nothing was published avoids a busy loop.
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
        yolo_imgsz=config.yolo_imgsz,
        phantoms_file=config.phantoms_file,
        phantom_max_confidence=config.phantom_max_confidence,
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

    floor_check_client = build_floor_check_client(config)
    floor_scheduler = (
        FloorCheckScheduler(
            client=floor_check_client,
            min_confidence=config.floor_check_min_confidence,
            cooldown_seconds=config.floor_check_cooldown_seconds,
            required_positives=config.floor_check_required_positives,
        )
        if floor_check_client is not None
        else None
    )
    floor_trigger = FloorCheckTrigger() if floor_scheduler is not None else None

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
            floor_scheduler=floor_scheduler,
            floor_trigger=floor_trigger,
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
