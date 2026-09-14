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
    build_vision_client,
    maybe_emit_health,
    maybe_emit_person_heartbeat,
    run_once,
)
from perceive.scene_notes import SCENE_NOTE_MAX_AGE_SECONDS, SceneNoteCache, synchronous_submit
from perceive.vision import FakeVisionClient
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
    assert config.pose_backend == "yolo"
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


# --- issue #9: scene notes wired into run_once/maybe_emit_person_heartbeat ---


def test_state_change_fires_one_vision_request_and_the_note_lands_on_the_next_person_state():
    """Acceptance evidence 1: a state change fires exactly one vision
    request, and the returned note lands on the next published `PersonState`."""
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = ScriptedBackend([in_bed_pose(), sitting_up_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    vision_client = FakeVisionClient(["Lying still in bed.", "The person is sitting up in bed."])
    scene_cache = SceneNoteCache(vision_client=vision_client, submit=synchronous_submit)

    _publish_frame(bus)
    event_1 = run_once(
        bus, backend, BED_ZONE, tracker, now_fn=lambda: 0.0, scene_cache=scene_cache, engaged=False
    )
    assert event_1.state == "in_bed"
    assert len(vision_client.calls) == 1  # first-ever state counts as a change
    assert event_1.scene_note == "Lying still in bed."

    _publish_frame(bus)
    event_2 = run_once(
        bus, backend, BED_ZONE, tracker, now_fn=lambda: 1.0, scene_cache=scene_cache, engaged=False
    )
    assert event_2.state == "sitting_up"
    assert len(vision_client.calls) == 2  # the state change fired a second request
    assert event_2.scene_note == "The person is sitting up in bed."


def test_second_frame_while_a_vision_request_is_in_flight_does_not_fire_a_second_request():
    """Acceptance evidence 2."""
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = ScriptedBackend([in_bed_pose(), in_bed_pose(), sitting_up_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    vision_client = FakeVisionClient(["note."])
    held_jobs = []
    scene_cache = SceneNoteCache(vision_client=vision_client, submit=held_jobs.append)

    for i in range(3):
        _publish_frame(bus)
        run_once(
            bus, backend, BED_ZONE, tracker, now_fn=lambda i=i: float(i), scene_cache=scene_cache
        )

    # Only the very first frame (first-ever state) fired a request; it never
    # completed (the held job was never run), so nothing after it -- not
    # even the confirmed `sitting_up` state change -- fires a second one.
    assert len(held_jobs) == 1
    assert len(vision_client.calls) == 0


def test_engaged_session_fires_on_interval_with_no_state_change():
    """Acceptance evidence 3."""
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = ScriptedBackend([in_bed_pose(), in_bed_pose(), in_bed_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    vision_client = FakeVisionClient(["first.", "second."])
    scene_cache = SceneNoteCache(
        vision_client=vision_client, interval_seconds=60.0, submit=synchronous_submit
    )

    _publish_frame(bus)
    run_once(
        bus, backend, BED_ZONE, tracker, now_fn=lambda: 0.0, scene_cache=scene_cache, engaged=True
    )
    assert len(vision_client.calls) == 1  # first-ever state

    _publish_frame(bus)
    run_once(
        bus, backend, BED_ZONE, tracker, now_fn=lambda: 30.0, scene_cache=scene_cache, engaged=True
    )
    assert len(vision_client.calls) == 1  # interval not elapsed yet, no state change

    _publish_frame(bus)
    run_once(
        bus, backend, BED_ZONE, tracker, now_fn=lambda: 60.0, scene_cache=scene_cache, engaged=True
    )
    assert len(vision_client.calls) == 2  # interval elapsed


def test_a_failing_vision_client_leaves_scene_note_none_and_person_state_still_publishes():
    """Acceptance evidence 4."""
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = ScriptedBackend([in_bed_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)

    class RaisingClient:
        def describe(self, jpeg):
            raise RuntimeError("Ollama is unreachable")

    scene_cache = SceneNoteCache(vision_client=RaisingClient(), submit=synchronous_submit)

    _publish_frame(bus)
    event = run_once(bus, backend, BED_ZONE, tracker, now_fn=lambda: 0.0, scene_cache=scene_cache)

    assert event is not None
    assert event.state == "in_bed"
    assert event.scene_note is None


def test_stale_scene_note_is_not_attached_to_person_state():
    """Acceptance evidence 5. A note attaches to the heartbeat too, so this
    checks it on the heartbeat path once the note is old enough to be stale
    -- without triggering a fresh vision request that would mask it."""
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = ScriptedBackend([in_bed_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    vision_client = FakeVisionClient(["a note."])
    scene_cache = SceneNoteCache(vision_client=vision_client, submit=synchronous_submit)

    _publish_frame(bus)
    first = run_once(bus, backend, BED_ZONE, tracker, now_fn=lambda: 0.0, scene_cache=scene_cache)
    assert first.scene_note == "a note."

    far_future = SCENE_NOTE_MAX_AGE_SECONDS + 1000.0
    last = maybe_emit_person_heartbeat(
        bus, tracker, None, now=far_future, interval=1.0, scene_cache=scene_cache
    )

    assert last == far_future
    entries = bus._streams["person"]  # noqa: SLF001
    heartbeat_event = PersonState.model_validate_json(entries[-1].data)
    assert heartbeat_event.state == "in_bed"
    assert heartbeat_event.scene_note is None


def test_vision_disabled_by_config_produces_no_vision_calls_at_all():
    """Acceptance evidence 6."""
    config = PerceiveConfig.from_env(env={"PERCEIVE_VISION_ENABLED": "false"})
    assert build_vision_client(config) is None

    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    backend = ScriptedBackend([in_bed_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)

    _publish_frame(bus)
    # No scene_cache at all -- the same shape `run()` builds when vision is
    # disabled -- so there is nothing here that could call Ollama.
    event = run_once(bus, backend, BED_ZONE, tracker, now_fn=lambda: 0.0)

    assert event is not None
    assert event.scene_note is None
