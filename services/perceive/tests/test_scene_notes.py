"""Tests for `perceive.scene_notes`: `SceneNoteCache` and `SessionPhaseTracker`.

Everything runs with `synchronous_submit`, so there are no real threads and
no sleeping anywhere in this file -- the whole point of the injected
executor hook (per the issue #9 brief: "a test that depends on thread
timing is a flaky test").
"""

from nc_shared.bus import FakeBus
from nc_shared.events import SessionState

from perceive.scene_notes import (
    SCENE_NOTE_MAX_AGE_SECONDS,
    SceneNoteCache,
    SessionPhaseTracker,
    read_session_phase,
    synchronous_submit,
)
from perceive.vision import FakeVisionClient


def _cache(responses, interval_seconds: float = 60.0) -> tuple[SceneNoteCache, FakeVisionClient]:
    client = FakeVisionClient(responses)
    cache = SceneNoteCache(
        vision_client=client, interval_seconds=interval_seconds, submit=synchronous_submit
    )
    return cache, client


def test_state_change_fires_exactly_one_request_and_note_becomes_available():
    cache, client = _cache(["standing near the door."])

    fired = cache.maybe_request(b"jpeg", "standing", now=0.0)

    assert fired is True
    assert len(client.calls) == 1
    assert cache.current(now=0.0) == "standing near the door."


def test_second_frame_in_flight_does_not_fire_a_second_request():
    # A submit hook that never runs the job simulates "still in flight":
    # nothing completes, so `_in_flight` stays True across the second call.
    client = FakeVisionClient(["a note."])
    held_jobs = []
    cache = SceneNoteCache(vision_client=client, submit=held_jobs.append)

    first = cache.maybe_request(b"jpeg-1", "standing", now=0.0)
    second = cache.maybe_request(b"jpeg-2", "standing", now=0.1)  # same state, no interval elapsed
    third = cache.maybe_request(b"jpeg-3", "walking", now=0.2)  # state changed, but still in flight

    assert first is True
    assert second is False
    assert third is False
    assert len(client.calls) == 0  # the held job never ran
    assert len(held_jobs) == 1  # only one job was ever submitted


def test_engaged_interval_fires_once_elapsed_and_not_before():
    cache, client = _cache(["note one.", "note two."], interval_seconds=60.0)

    assert cache.maybe_request(b"jpeg", "standing", now=0.0, engaged=True) is True
    assert len(client.calls) == 1

    # Same state, engaged, interval not yet elapsed: no new request.
    assert cache.maybe_request(b"jpeg", "standing", now=59.0, engaged=True) is False
    assert len(client.calls) == 1

    # Same state, engaged, interval elapsed: fires.
    assert cache.maybe_request(b"jpeg", "standing", now=60.0, engaged=True) is True
    assert len(client.calls) == 2
    assert cache.current(now=60.0) == "note two."


def test_not_engaged_same_state_never_fires_on_interval_alone():
    cache, client = _cache(["note."])

    assert cache.maybe_request(b"jpeg", "standing", now=0.0, engaged=False) is True
    assert cache.maybe_request(b"jpeg", "standing", now=1000.0, engaged=False) is False
    assert len(client.calls) == 1


def test_failed_call_leaves_current_as_none():
    cache, client = _cache([None])

    cache.maybe_request(b"jpeg", "standing", now=0.0)

    assert len(client.calls) == 1
    assert cache.current(now=0.0) is None


def test_raising_vision_client_is_swallowed_and_current_stays_none():
    class RaisingClient:
        def describe(self, jpeg):
            raise RuntimeError("model process died")

    cache = SceneNoteCache(vision_client=RaisingClient(), submit=synchronous_submit)

    # Must not raise.
    fired = cache.maybe_request(b"jpeg", "standing", now=0.0)

    assert fired is True
    assert cache.current(now=0.0) is None


def test_stale_note_is_not_attached():
    cache, _client = _cache(["a note."])
    cache.maybe_request(b"jpeg", "standing", now=0.0)

    assert cache.current(now=SCENE_NOTE_MAX_AGE_SECONDS) == "a note."
    assert cache.current(now=SCENE_NOTE_MAX_AGE_SECONDS + 0.01) is None


def test_session_phase_tracker_defaults_to_idle_and_not_engaged():
    tracker = SessionPhaseTracker()
    assert tracker.phase == "IDLE"
    assert tracker.engaged is False


def test_read_session_phase_updates_tracker_from_latest_session_state():
    bus = FakeBus()
    tracker = SessionPhaseTracker()

    bus.publish(
        SessionState(source="agent", phase="OBSERVING", goal="return_to_bed", strategy_index=0)
    )
    bus.publish(
        SessionState(source="agent", phase="ENGAGED", goal="return_to_bed", strategy_index=1)
    )

    read_session_phase(bus, tracker)

    assert tracker.phase == "ENGAGED"
    assert tracker.engaged is True


def test_read_session_phase_does_nothing_when_stream_is_empty():
    bus = FakeBus()
    tracker = SessionPhaseTracker()

    read_session_phase(bus, tracker)

    assert tracker.phase == "IDLE"


class _GroupCountingBus(FakeBus):
    """A `FakeBus` that counts how often a consumer group is created."""

    def __init__(self) -> None:
        super().__init__()
        self.ensure_group_calls = 0

    def ensure_group(self, stream: str, group: str) -> None:
        self.ensure_group_calls += 1
        super().ensure_group(stream, group)


def test_the_session_group_is_created_once_not_once_per_loop_iteration():
    """`run()` calls `read_session_phase` on every iteration of a loop that
    spins roughly twenty times a second. Creating the consumer group each
    time would mean a redundant round-trip to Redis all night long.
    """
    bus = _GroupCountingBus()
    tracker = SessionPhaseTracker()

    for _ in range(50):
        read_session_phase(bus, tracker)

    assert bus.ensure_group_calls == 1


def test_a_phase_published_after_the_group_exists_is_still_seen():
    """Guarding group creation must not cost us later reads."""
    bus = _GroupCountingBus()
    tracker = SessionPhaseTracker()
    read_session_phase(bus, tracker)
    assert tracker.phase == "IDLE"

    bus.publish(
        SessionState(source="agent", phase="ENGAGED", goal="return_to_bed", strategy_index=0)
    )
    read_session_phase(bus, tracker)

    assert tracker.phase == "ENGAGED"
    assert tracker.engaged is True
    assert bus.ensure_group_calls == 1
