"""Tests for `perceive.main`'s loop body, using `FakeBus` and `ScriptedBackend`.

No camera, no Redis, no model weights, no Ollama, per HANDOFF.md section 4.
`test_full_scripted_night_publishes_the_expected_person_state_sequence` is
the acceptance evidence for issue #8: it drives a full scripted sequence --
lying in bed, sitting up, standing, walking toward the door, then on the
floor -- through `run_once` and asserts the exact `PersonState` sequence
published, including that `on_floor` bypasses hysteresis.
"""

from nc_shared.bus import FakeBus
from nc_shared.events import Frame, Health, PersonState
from nc_shared.replay import CAPPED_MAXLEN

from perceive.backends import Landmark, PoseResult, ScriptedBackend
from perceive.classify import ClassifyThresholds, StateTracker
from perceive.main import (
    FRAME_GROUP,
    FRAME_STREAM,
    PerceiveConfig,
    build_tracker,
    maybe_emit_health,
    maybe_emit_person_heartbeat,
    run_once,
)
from perceive.zones import ZoneMap

BED_ZONE = ZoneMap(polygons={"bed": [(0.0, 0.2), (0.4, 0.2), (0.4, 0.8), (0.0, 0.8)]})


def _pose(points: dict[str, tuple[float, float]], confidence: float = 0.9) -> PoseResult:
    landmarks = {name: Landmark(x=x, y=y, visibility=0.9) for name, (x, y) in points.items()}
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return PoseResult(landmarks=landmarks, bbox=bbox, confidence=confidence)


def in_bed_pose() -> PoseResult:
    return _pose(
        {
            "nose": (0.05, 0.50),
            "left_shoulder": (0.08, 0.48),
            "right_shoulder": (0.08, 0.52),
            "left_hip": (0.30, 0.50),
            "right_hip": (0.30, 0.54),
            "left_knee": (0.35, 0.50),
            "right_knee": (0.35, 0.54),
            "left_ankle": (0.38, 0.50),
            "right_ankle": (0.38, 0.54),
        }
    )


def sitting_up_pose() -> PoseResult:
    return _pose(
        {
            "nose": (0.50, 0.30),
            "left_shoulder": (0.48, 0.35),
            "right_shoulder": (0.52, 0.35),
            "left_hip": (0.49, 0.55),
            "right_hip": (0.51, 0.55),
            "left_knee": (0.49, 0.60),
            "right_knee": (0.51, 0.60),
            "left_ankle": (0.49, 0.58),
            "right_ankle": (0.51, 0.58),
        }
    )


def standing_pose(centroid_x: float = 0.50) -> PoseResult:
    offset = centroid_x - 0.50
    return _pose(
        {
            "nose": (0.50 + offset, 0.10),
            "left_shoulder": (0.48 + offset, 0.20),
            "right_shoulder": (0.52 + offset, 0.20),
            "left_hip": (0.49 + offset, 0.50),
            "right_hip": (0.51 + offset, 0.50),
            "left_knee": (0.49 + offset, 0.75),
            "right_knee": (0.51 + offset, 0.75),
            "left_ankle": (0.49 + offset, 0.95),
            "right_ankle": (0.51 + offset, 0.95),
        }
    )


def on_floor_pose() -> PoseResult:
    return _pose(
        {
            "nose": (0.70, 0.85),
            "left_shoulder": (0.65, 0.83),
            "right_shoulder": (0.75, 0.83),
            "left_hip": (0.68, 0.86),
            "right_hip": (0.78, 0.86),
            "left_knee": (0.72, 0.87),
            "right_knee": (0.82, 0.87),
            "left_ankle": (0.75, 0.88),
            "right_ankle": (0.85, 0.88),
        }
    )


def _publish_frame(bus) -> None:
    bus.publish(
        Frame(source="capture", jpeg=b"\xff\xd8\xff", width=320, height=240, source_kind="browser"),
        maxlen=CAPPED_MAXLEN["frames"],
    )


def test_full_scripted_night_publishes_the_expected_person_state_sequence():
    """Lie in bed, sit up, stand, walk toward the door, then fall.

    This is issue #8's acceptance line: the exact `PersonState.state`
    sequence `run_once` publishes must match what a caregiver watching the
    room would call each moment, and `on_floor` must land on the very
    frame it appears on rather than waiting for `PERCEIVE_CONFIRM_FRAMES`
    like every other state does.
    """
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    bus.ensure_group("person", "test-consumer")

    poses = [
        in_bed_pose(),
        in_bed_pose(),
        sitting_up_pose(),
        sitting_up_pose(),
        standing_pose(0.50),
        standing_pose(0.50),
        standing_pose(0.65),
        standing_pose(0.80),
        on_floor_pose(),
    ]
    backend = ScriptedBackend(poses)
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=2)

    published_states: list[str] = []
    for i in range(len(poses)):
        _publish_frame(bus)
        event = run_once(bus, backend, BED_ZONE, tracker, now_fn=lambda i=i: float(i))
        if event is not None:
            published_states.append(event.state)

    assert published_states == ["in_bed", "sitting_up", "standing", "walking", "on_floor"]

    # `on_floor` must have bypassed the confirm_frames=2 hysteresis: it
    # published on the single frame it appeared on (frame index 8, the
    # ninth and last frame), not after a repeat.
    read = bus.read("person", "test-consumer", "consumer-1")
    events = [PersonState.model_validate_json(e.data) for e in bus._streams["person"]]  # noqa: SLF001
    assert events[-1].state == "on_floor"
    assert events[0].state == "in_bed"
    assert events[0].zone == "bed"
    assert events[-1].zone != "bed"
    assert len(read) == len(events)  # every publish landed on the "person" stream once


def test_run_once_returns_none_when_no_frame_is_available():
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = ScriptedBackend([])
    tracker = build_tracker(PerceiveConfig())

    assert run_once(bus, backend, ZoneMap(), tracker, now_fn=lambda: 0.0) is None


def test_run_once_acks_the_frame_message():
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    _publish_frame(bus)
    backend = ScriptedBackend([in_bed_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)

    run_once(bus, backend, BED_ZONE, tracker, now_fn=lambda: 0.0)

    assert bus.pending(FRAME_STREAM, FRAME_GROUP) == []


def test_run_once_returns_none_while_hysteresis_is_pending():
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    _publish_frame(bus)
    backend = ScriptedBackend([in_bed_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=3)

    assert run_once(bus, backend, BED_ZONE, tracker, now_fn=lambda: 0.0) is None


def test_perceive_config_from_env_uses_defaults_when_unset():
    config = PerceiveConfig.from_env(env={})
    assert config.pose_backend == "mediapipe"
    assert config.min_confidence == 0.5
    assert config.confirm_frames == 3
    assert config.walk_threshold == 0.15
    assert config.heartbeat_seconds == 60.0
    assert config.zones_path is None


def test_perceive_config_from_env_reads_every_key():
    env = {
        "PERCEIVE_POSE_BACKEND": "scripted",
        "PERCEIVE_MIN_CONFIDENCE": "0.4",
        "PERCEIVE_CONFIRM_FRAMES": "5",
        "PERCEIVE_WALK_THRESHOLD": "0.2",
        "PERCEIVE_HEARTBEAT_SECONDS": "30",
        "ZONES_PATH": "/app/config/zones.yaml",
    }
    config = PerceiveConfig.from_env(env=env)
    assert config.pose_backend == "scripted"
    assert config.min_confidence == 0.4
    assert config.confirm_frames == 5
    assert config.walk_threshold == 0.2
    assert config.heartbeat_seconds == 30.0
    assert config.zones_path == "/app/config/zones.yaml"


def test_build_tracker_wires_config_thresholds_through():
    config = PerceiveConfig(min_confidence=0.7, confirm_frames=4, walk_threshold=0.25)
    tracker = build_tracker(config)
    assert tracker.confirm_frames == 4
    assert tracker.thresholds.min_confidence == 0.7
    assert tracker.thresholds.walk_displacement_threshold == 0.25


def test_maybe_emit_health_emits_on_first_call():
    bus = FakeBus()
    last = maybe_emit_health(bus, None, now=0.0)

    assert last == 0.0
    entries = bus._streams["health"]  # noqa: SLF001
    assert len(entries) == 1
    event = Health.model_validate_json(entries[0].data)
    assert event.service == "perceive"
    assert event.ok is True


def test_maybe_emit_health_waits_for_the_interval():
    bus = FakeBus()
    last = maybe_emit_health(bus, None, now=0.0, interval=30.0)
    last = maybe_emit_health(bus, last, now=10.0, interval=30.0)

    assert last == 0.0
    assert len(bus._streams["health"]) == 1  # noqa: SLF001


def test_maybe_emit_person_heartbeat_does_nothing_before_any_classification():
    bus = FakeBus()
    tracker = StateTracker(thresholds=ClassifyThresholds())

    last = maybe_emit_person_heartbeat(bus, tracker, None, now=0.0, interval=60.0)

    assert last is None
    assert bus._streams == {}  # noqa: SLF001


def test_maybe_emit_person_heartbeat_repeats_the_last_confirmed_state():
    bus = FakeBus()
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    tracker.update(in_bed_pose(), "bed", now=0.0)

    last = maybe_emit_person_heartbeat(bus, tracker, None, now=100.0, interval=60.0)

    assert last == 100.0
    entries = bus._streams["person"]  # noqa: SLF001
    assert len(entries) == 1
    event = PersonState.model_validate_json(entries[0].data)
    assert event.state == "in_bed"
    assert event.zone == "bed"


def test_maybe_emit_person_heartbeat_waits_for_the_interval():
    bus = FakeBus()
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    tracker.update(in_bed_pose(), "bed", now=0.0)

    last = maybe_emit_person_heartbeat(bus, tracker, 100.0, now=110.0, interval=60.0)

    assert last == 100.0
    assert bus._streams.get("person", []) == []  # noqa: SLF001
