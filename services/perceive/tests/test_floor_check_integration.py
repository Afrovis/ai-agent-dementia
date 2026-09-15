"""Integration tests for `perceive.main.run_once`'s floor-check wiring:
`FloorCheckScheduler`/`FloorCheckTrigger` driving `StateTracker.confirm_floor`
through a fake `FloorCheckClient`, end to end, with `synchronous_submit` so
there are no real threads and no sleeping.

No camera, no Redis, no model weights, no Ollama, per HANDOFF.md section 4.
"""

from nc_shared.bus import FakeBus
from nc_shared.events import Frame, PersonState
from nc_shared.replay import CAPPED_MAXLEN

from perceive.backends import Landmark, PoseResult, ScriptedBackend
from perceive.classify import ClassifyThresholds, StateTracker
from perceive.floor_check import (
    FakeFloorCheckClient,
    FloorCheckResult,
    FloorCheckScheduler,
    FloorCheckTrigger,
    synchronous_submit,
)
from perceive.main import FRAME_GROUP, FRAME_STREAM, run_once
from perceive.zones import ZoneMap

BED_ZONE = ZoneMap(polygons={"bed": [(0.0, 0.2), (0.4, 0.2), (0.4, 0.8), (0.0, 0.8)]})
FULL_FRAME_BED_ZONE = ZoneMap(polygons={"bed": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]})


def _pose(points: dict[str, tuple[float, float]], confidence: float = 0.9) -> PoseResult:
    landmarks = {name: Landmark(x=x, y=y, visibility=0.9) for name, (x, y) in points.items()}
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return PoseResult(landmarks=landmarks, bbox=bbox, confidence=confidence)


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


def standing_pose() -> PoseResult:
    return _pose(
        {
            "nose": (0.50, 0.10),
            "left_shoulder": (0.48, 0.20),
            "right_shoulder": (0.52, 0.20),
            "left_hip": (0.49, 0.50),
            "right_hip": (0.51, 0.50),
            "left_knee": (0.49, 0.75),
            "right_knee": (0.51, 0.75),
            "left_ankle": (0.49, 0.95),
            "right_ankle": (0.51, 0.95),
        }
    )


def _publish_frame(bus) -> None:
    bus.publish(
        Frame(source="capture", jpeg=b"\xff\xd8\xff", width=640, height=480, source_kind="browser"),
        maxlen=CAPPED_MAXLEN["frames"],
    )


def _bus() -> FakeBus:
    bus = FakeBus()
    bus.ensure_group(FRAME_STREAM, FRAME_GROUP)
    return bus


def _tracker_with_floor_suspect(now: float) -> StateTracker:
    """A tracker reporting `absent`, with `floor_suspect` already armed and
    the last-seen zone outside the bed -- the state `run_once` would be in
    after a fall drop was observed and the person then lost entirely."""
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    tracker.update(sitting_up_pose(), "other", now=now - 5.0)
    tracker._floor_suspect_since = now - 1.0  # noqa: SLF001
    tracker._last_seen_zone = "other"  # noqa: SLF001
    tracker._undetected_since = now - 1.0  # noqa: SLF001
    tracker._current = "absent"  # noqa: SLF001
    return tracker


def test_floor_check_fires_on_floor_suspect_and_confirms_after_two_positives():
    bus = _bus()
    bus.ensure_group("person", "test-consumer")
    backend = ScriptedBackend([None, None])
    tracker = _tracker_with_floor_suspect(now=0.0)

    client = FakeFloorCheckClient(
        [
            FloorCheckResult(person_on_floor=True, confidence=0.9),
            FloorCheckResult(person_on_floor=True, confidence=0.9),
        ]
    )
    scheduler = FloorCheckScheduler(client=client, submit=synchronous_submit)
    trigger = FloorCheckTrigger()

    times = iter([0.0, 1.5])
    for _ in range(2):
        _publish_frame(bus)
        run_once(
            bus,
            backend,
            BED_ZONE,
            tracker,
            now_fn=lambda: next(times),
            floor_scheduler=scheduler,
            floor_trigger=trigger,
        )

    assert len(client.calls) == 2
    events = [PersonState.model_validate_json(e.data) for e in bus._streams["person"]]  # noqa: SLF001
    assert events[-1].state == "on_floor"
    assert events[-1].scene_note == "vision: person on floor"


def test_floor_check_does_not_confirm_after_a_single_positive():
    bus = _bus()
    bus.ensure_group("person", "test-consumer")
    backend = ScriptedBackend([None])
    tracker = _tracker_with_floor_suspect(now=0.0)

    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.9)])
    scheduler = FloorCheckScheduler(client=client, submit=synchronous_submit)
    trigger = FloorCheckTrigger()

    _publish_frame(bus)
    run_once(
        bus,
        backend,
        BED_ZONE,
        tracker,
        now_fn=lambda: 0.0,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert bus._streams.get("person", []) == []  # noqa: SLF001
    assert scheduler.positive_streak_count == 1


def test_floor_check_negative_answer_resets_the_streak():
    bus = _bus()
    backend = ScriptedBackend([None, None])
    tracker = _tracker_with_floor_suspect(now=0.0)

    client = FakeFloorCheckClient(
        [
            FloorCheckResult(person_on_floor=True, confidence=0.9),
            FloorCheckResult(person_on_floor=False, confidence=0.9),
        ]
    )
    scheduler = FloorCheckScheduler(client=client, submit=synchronous_submit)
    trigger = FloorCheckTrigger()

    times = iter([0.0, 1.5])
    for _ in range(2):
        _publish_frame(bus)
        run_once(
            bus,
            backend,
            BED_ZONE,
            tracker,
            now_fn=lambda: next(times),
            floor_scheduler=scheduler,
            floor_trigger=trigger,
        )

    assert scheduler.positive_streak_count == 0
    assert bus._streams.get("person", []) == []  # noqa: SLF001


def test_floor_check_low_confidence_answer_does_not_count_as_positive():
    bus = _bus()
    backend = ScriptedBackend([None])
    tracker = _tracker_with_floor_suspect(now=0.0)

    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.2)])
    scheduler = FloorCheckScheduler(client=client, submit=synchronous_submit, min_confidence=0.6)
    trigger = FloorCheckTrigger()

    _publish_frame(bus)
    run_once(
        bus,
        backend,
        BED_ZONE,
        tracker,
        now_fn=lambda: 0.0,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert scheduler.positive_streak_count == 0


def test_floor_check_never_triggers_in_bed_zone():
    bus = _bus()
    backend = ScriptedBackend([sitting_up_pose(), sitting_up_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)

    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.9)])
    scheduler = FloorCheckScheduler(client=client, submit=synchronous_submit)
    trigger = FloorCheckTrigger()

    _publish_frame(bus)
    run_once(
        bus,
        backend,
        FULL_FRAME_BED_ZONE,
        tracker,
        now_fn=lambda: 0.0,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )
    # Sitting up inside the bed zone -- the trigger must never fire here.
    _publish_frame(bus)
    run_once(
        bus,
        backend,
        FULL_FRAME_BED_ZONE,
        tracker,
        now_fn=lambda: 0.1,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert len(client.calls) == 0


def test_floor_check_never_triggers_when_already_on_floor():
    """`on_floor_pose` (pose-confirmed) must never get a redundant vision
    check -- the check can only ever upgrade an ambiguous frame."""
    bus = _bus()
    on_floor_pose = _pose(
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
    backend = ScriptedBackend([on_floor_pose])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)

    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.9)])
    scheduler = FloorCheckScheduler(client=client, submit=synchronous_submit)
    trigger = FloorCheckTrigger()

    _publish_frame(bus)
    run_once(
        bus,
        backend,
        ZoneMap(),
        tracker,
        now_fn=lambda: 0.0,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert len(client.calls) == 0


def test_floor_check_client_error_is_a_no_op():
    bus = _bus()
    backend = ScriptedBackend([None])
    tracker = _tracker_with_floor_suspect(now=0.0)

    class RaisingClient:
        def check(self, jpeg):
            raise RuntimeError("model process died")

    scheduler = FloorCheckScheduler(client=RaisingClient(), submit=synchronous_submit)
    trigger = FloorCheckTrigger()

    _publish_frame(bus)
    event = run_once(
        bus,
        backend,
        BED_ZONE,
        tracker,
        now_fn=lambda: 0.0,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert event is None
    assert bus._streams.get("person", []) == []  # noqa: SLF001
    assert scheduler.positive_streak_count == 0


def test_floor_check_stale_answer_is_ignored():
    bus = _bus()
    backend = ScriptedBackend([None])
    tracker = _tracker_with_floor_suspect(now=0.0)
    tracker._floor_suspect_since = -100.0  # noqa: SLF001 - still armed, but irrelevant here

    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.9)])
    scheduler = FloorCheckScheduler(
        client=client, submit=synchronous_submit, answer_max_age_seconds=30.0
    )
    trigger = FloorCheckTrigger()

    # Trigger fires "now" at t=0, but by the time run_once is called again
    # the answer is already older than answer_max_age_seconds.
    scheduler.maybe_trigger(b"jpeg", "floor_suspect", now=0.0)

    _publish_frame(bus)
    run_once(
        bus,
        backend,
        BED_ZONE,
        tracker,
        now_fn=lambda: 100.0,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert scheduler.positive_streak_count == 0
    assert bus._streams.get("person", []) == []  # noqa: SLF001


def test_floor_check_ignored_when_condition_no_longer_holds_on_arrival():
    """The person got up between the check firing and the answer arriving:
    a still-true `person_on_floor` answer must not force `on_floor` then."""
    bus = _bus()
    backend = ScriptedBackend([standing_pose(), standing_pose()])
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)

    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.9)])
    held_jobs = []
    scheduler = FloorCheckScheduler(client=client, submit=held_jobs.append)
    trigger = FloorCheckTrigger()

    # Arm floor_suspect manually and hold the check in flight.
    tracker._floor_suspect_since = 0.0  # noqa: SLF001
    tracker._last_seen_zone = "other"  # noqa: SLF001
    tracker._undetected_since = 0.0  # noqa: SLF001
    tracker._current = "absent"  # noqa: SLF001
    fired = scheduler.maybe_trigger(b"jpeg", "floor_suspect", now=0.0)
    assert fired is True
    held_jobs[0]()  # run the held job now, synchronously

    # By the time the answer is collected, the person is confirmed standing
    # and the suspicion has cleared -- the trigger condition no longer holds.
    tracker._floor_suspect_since = None  # noqa: SLF001

    _publish_frame(bus)
    run_once(
        bus,
        backend,
        ZoneMap(),
        tracker,
        now_fn=lambda: 0.1,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert bus._streams.get("person", {}) is not None
    events = [PersonState.model_validate_json(e.data) for e in bus._streams.get("person", [])]  # noqa: SLF001
    assert all(e.state != "on_floor" for e in events)
    assert scheduler.positive_streak_count == 0


def test_on_floor_from_vision_clears_when_pose_confirms_upright_again():
    tracker = StateTracker(thresholds=ClassifyThresholds(), confirm_frames=1)
    confirmed = tracker.confirm_floor(now=0.0, confidence=0.9, source="vision")
    assert confirmed == ("on_floor", 0.9)
    assert tracker.snapshot()[0] == "on_floor"

    result = tracker.update(standing_pose(), "other", now=1.0)
    assert result is not None
    assert result[0] == "standing"
    assert tracker.snapshot()[0] == "standing"


def test_confirm_floor_does_not_republish_when_already_on_floor():
    tracker = StateTracker(thresholds=ClassifyThresholds())
    first = tracker.confirm_floor(now=0.0, confidence=0.9)
    second = tracker.confirm_floor(now=1.0, confidence=0.9)

    assert first == ("on_floor", 0.9)
    assert second is None


def test_run_once_floor_check_never_blocks_frame_loop_without_a_completed_answer():
    """Even with a check in flight (held, never run), `run_once` must
    return promptly -- the frame loop never blocks on a floor check."""
    bus = _bus()
    backend = ScriptedBackend([None])
    tracker = _tracker_with_floor_suspect(now=0.0)

    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.9)])
    held_jobs = []
    scheduler = FloorCheckScheduler(client=client, submit=held_jobs.append)
    trigger = FloorCheckTrigger()

    _publish_frame(bus)
    event = run_once(
        bus,
        backend,
        BED_ZONE,
        tracker,
        now_fn=lambda: 0.0,
        floor_scheduler=scheduler,
        floor_trigger=trigger,
    )

    assert event is None  # tracker still reports absent; nothing new to publish
    assert len(held_jobs) == 1  # the check was scheduled, not run inline
    assert len(client.calls) == 0
