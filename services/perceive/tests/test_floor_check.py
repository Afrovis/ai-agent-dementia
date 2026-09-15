"""Tests for `perceive.floor_check`: `FloorCheckTrigger`, `FloorCheckScheduler`,
`FakeFloorCheckClient`, and `OllamaFloorCheckClient`'s fail-safe behaviour
with a stubbed `httpx`.

`FloorCheckScheduler` tests all run with `synchronous_submit`, so there are
no real threads and no sleeping anywhere in this file, matching
`test_scene_notes.py`'s reasoning.
"""

import json

from perceive.classify import ClassifyThresholds, StateTracker
from perceive.floor_check import (
    FLOOR_CHECK_PROMPT,
    FakeFloorCheckClient,
    FloorCheckResult,
    FloorCheckScheduler,
    FloorCheckTrigger,
    OllamaFloorCheckClient,
    synchronous_submit,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_body: object = None, raise_on_json=None):
        self.status_code = status_code
        self._json_body = json_body
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._json_body


def _chat_body(person_on_floor: bool, confidence: float) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps({"person_on_floor": person_on_floor, "confidence": confidence}),
        }
    }


# --- OllamaFloorCheckClient --------------------------------------------------


def test_ollama_floor_check_client_parses_a_successful_answer(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200, _chat_body(True, 0.87))

    monkeypatch.setattr("perceive.floor_check.httpx.post", fake_post)

    client = OllamaFloorCheckClient(
        ollama_url="http://host.docker.internal:11434", model="qwen3-vl:8b", timeout_seconds=5.0
    )
    result = client.check(b"\xff\xd8\xff")

    assert result == FloorCheckResult(person_on_floor=True, confidence=0.87)
    assert captured["url"] == "http://host.docker.internal:11434/api/chat"
    assert captured["json"]["model"] == "qwen3-vl:8b"
    assert captured["json"]["options"] == {"temperature": 0}
    assert captured["json"]["think"] is False
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["content"] == FLOOR_CHECK_PROMPT
    assert "images" in captured["json"]["messages"][0]


def test_ollama_floor_check_client_omits_think_when_unset(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse(200, _chat_body(False, 0.1))

    monkeypatch.setattr("perceive.floor_check.httpx.post", fake_post)
    client = OllamaFloorCheckClient(ollama_url="http://x:11434", model="m", think=None)

    assert client.check(b"jpeg") == FloorCheckResult(person_on_floor=False, confidence=0.1)
    assert "think" not in captured["json"]


def test_ollama_floor_check_client_recovers_balanced_json_object(monkeypatch):
    body = {
        "message": {
            "content": 'json\n{{"person_on_floor": true, "confidence": 0.95}',
            "thinking": '{"person_on_floor": false, "confidence": 0.01}',
        }
    }
    monkeypatch.setattr(
        "perceive.floor_check.httpx.post", lambda url, json, timeout: _FakeResponse(200, body)
    )
    client = OllamaFloorCheckClient(ollama_url="http://x:11434", model="m")

    assert client.check(b"jpeg") == FloorCheckResult(person_on_floor=True, confidence=0.95)


def test_ollama_floor_check_client_returns_none_on_transport_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise ConnectionError("no route to host")

    monkeypatch.setattr("perceive.floor_check.httpx.post", fake_post)
    client = OllamaFloorCheckClient(ollama_url="http://x:11434", model="m")

    assert client.check(b"jpeg") is None


def test_ollama_floor_check_client_returns_none_on_non_200(monkeypatch):
    monkeypatch.setattr(
        "perceive.floor_check.httpx.post", lambda url, json, timeout: _FakeResponse(500)
    )
    client = OllamaFloorCheckClient(ollama_url="http://x:11434", model="m")

    assert client.check(b"jpeg") is None


def test_ollama_floor_check_client_returns_none_on_malformed_content(monkeypatch):
    body = {"message": {"content": "not json"}}
    monkeypatch.setattr(
        "perceive.floor_check.httpx.post", lambda url, json, timeout: _FakeResponse(200, body)
    )
    client = OllamaFloorCheckClient(ollama_url="http://x:11434", model="m")

    assert client.check(b"jpeg") is None


def test_ollama_floor_check_client_returns_none_on_missing_fields(monkeypatch):
    body = {"message": {"content": json.dumps({"person_on_floor": True})}}
    monkeypatch.setattr(
        "perceive.floor_check.httpx.post", lambda url, json, timeout: _FakeResponse(200, body)
    )
    client = OllamaFloorCheckClient(ollama_url="http://x:11434", model="m")

    assert client.check(b"jpeg") is None


def test_fake_floor_check_client_returns_canned_responses_in_order_and_records_calls():
    r1 = FloorCheckResult(person_on_floor=True, confidence=0.9)
    client = FakeFloorCheckClient([r1, None])

    assert client.check(b"jpeg-1") == r1
    assert client.check(b"jpeg-2") is None
    assert client.check(b"jpeg-3") is None  # exhausted
    assert client.calls == [b"jpeg-1", b"jpeg-2", b"jpeg-3"]


# --- FloorCheckTrigger --------------------------------------------------------


def _tracker() -> StateTracker:
    return StateTracker(thresholds=ClassifyThresholds())


def test_trigger_fires_on_floor_suspect():
    tracker = _tracker()
    tracker._floor_suspect_since = 5.0  # noqa: SLF001 - simulate an active suspicion

    trigger = FloorCheckTrigger()
    assert trigger.reason(tracker, "absent", "other", now=10.0) == "floor_suspect"


def test_trigger_fires_when_lost_outside_bed_recently():
    tracker = _tracker()
    tracker._last_seen_zone = "other"  # noqa: SLF001
    tracker._undetected_since = 10.0  # noqa: SLF001

    trigger = FloorCheckTrigger()
    assert trigger.reason(tracker, "absent", "other", now=15.0) == "lost_outside_bed"


def test_trigger_does_not_fire_when_lost_more_than_30s_ago():
    tracker = _tracker()
    tracker._last_seen_zone = "other"  # noqa: SLF001
    tracker._undetected_since = 0.0  # noqa: SLF001

    trigger = FloorCheckTrigger()
    assert trigger.reason(tracker, "absent", "other", now=30.01) is None


def test_trigger_does_not_fire_when_last_seen_zone_was_bed():
    tracker = _tracker()
    tracker._last_seen_zone = "bed"  # noqa: SLF001
    tracker._undetected_since = 0.0  # noqa: SLF001

    trigger = FloorCheckTrigger()
    assert trigger.reason(tracker, "absent", "other", now=1.0) is None


def test_trigger_fires_once_low_ratio_persists_three_seconds():
    tracker = _tracker()
    tracker._last_height_ratio = 0.5  # noqa: SLF001

    trigger = FloorCheckTrigger()
    assert trigger.reason(tracker, "sitting_up", "other", now=0.0) is None
    assert trigger.reason(tracker, "sitting_up", "other", now=2.0) is None
    assert trigger.reason(tracker, "sitting_up", "other", now=3.0) == "low_height_ratio"


def test_trigger_resets_low_ratio_timer_when_ratio_recovers():
    tracker = _tracker()
    trigger = FloorCheckTrigger()

    tracker._last_height_ratio = 0.5  # noqa: SLF001
    assert trigger.reason(tracker, "sitting_up", "other", now=0.0) is None

    tracker._last_height_ratio = 0.9  # noqa: SLF001 - ratio recovers before 3s
    assert trigger.reason(tracker, "sitting_up", "other", now=1.0) is None

    tracker._last_height_ratio = 0.5  # noqa: SLF001 - low again, timer restarts
    assert trigger.reason(tracker, "sitting_up", "other", now=1.5) is None
    assert trigger.reason(tracker, "sitting_up", "other", now=4.4) is None
    assert trigger.reason(tracker, "sitting_up", "other", now=4.5) == "low_height_ratio"


def test_trigger_never_fires_in_bed_zone():
    tracker = _tracker()
    tracker._floor_suspect_since = 0.0  # noqa: SLF001
    trigger = FloorCheckTrigger()

    assert trigger.reason(tracker, "in_bed", "bed", now=1.0) is None


def test_trigger_never_fires_in_door_zone():
    tracker = _tracker()
    tracker._floor_suspect_since = 0.0  # noqa: SLF001
    trigger = FloorCheckTrigger()

    assert trigger.reason(tracker, "standing", "door", now=1.0) is None


def test_trigger_never_fires_when_already_on_floor():
    tracker = _tracker()
    tracker._floor_suspect_since = 0.0  # noqa: SLF001
    trigger = FloorCheckTrigger()

    assert trigger.reason(tracker, "on_floor", "other", now=1.0) is None


def test_trigger_does_not_fire_for_standing_or_walking():
    tracker = _tracker()
    trigger = FloorCheckTrigger()

    assert trigger.reason(tracker, "standing", "other", now=1.0) is None
    assert trigger.reason(tracker, "walking", "other", now=1.0) is None


# --- FloorCheckScheduler -------------------------------------------------------


def _scheduler(responses, **kwargs) -> tuple[FloorCheckScheduler, FakeFloorCheckClient]:
    client = FakeFloorCheckClient(responses)
    scheduler = FloorCheckScheduler(client=client, submit=synchronous_submit, **kwargs)
    return scheduler, client


def test_scheduler_fires_and_result_is_available_via_take_result():
    scheduler, client = _scheduler([FloorCheckResult(person_on_floor=True, confidence=0.9)])

    fired = scheduler.maybe_trigger(b"jpeg", "floor_suspect", now=0.0)

    assert fired is True
    assert len(client.calls) == 1
    result = scheduler.take_result()
    assert result is not None
    answer, reason, triggered_at, latency_ms = result
    assert answer == FloorCheckResult(person_on_floor=True, confidence=0.9)
    assert reason == "floor_suspect"
    assert triggered_at == 0.0
    assert latency_ms >= 0.0


def test_scheduler_take_result_consumes_it_once():
    scheduler, _client = _scheduler([FloorCheckResult(person_on_floor=False, confidence=0.1)])
    scheduler.maybe_trigger(b"jpeg", "floor_suspect", now=0.0)

    assert scheduler.take_result() is not None
    assert scheduler.take_result() is None


def test_scheduler_respects_cooldown_between_triggers():
    scheduler, client = _scheduler([None, None], cooldown_seconds=10.0)

    assert scheduler.maybe_trigger(b"jpeg-1", "floor_suspect", now=0.0) is True
    scheduler.take_result()
    assert scheduler.maybe_trigger(b"jpeg-2", "floor_suspect", now=5.0) is False
    assert scheduler.maybe_trigger(b"jpeg-3", "floor_suspect", now=10.0) is True
    assert len(client.calls) == 2


def test_scheduler_defaults_to_requiring_two_positives():
    scheduler, _client = _scheduler([])
    assert scheduler.required_positives == 2


def test_scheduler_note_positive_increments_and_returns_the_streak_count():
    scheduler, _client = _scheduler([])

    assert scheduler.positive_streak_count == 0
    assert scheduler.note_positive() == 1
    assert scheduler.positive_streak_count == 1
    assert scheduler.note_positive() == 2
    assert scheduler.positive_streak_count == 2


def test_scheduler_reset_positive_streak_zeroes_the_count():
    scheduler, _client = _scheduler([])
    scheduler.note_positive()
    scheduler.note_positive()

    scheduler.reset_positive_streak()

    assert scheduler.positive_streak_count == 0


def test_scheduler_skips_the_normal_cooldown_while_a_positive_streak_is_running():
    """A follow-up check, once a first qualifying positive has come in,
    fires as soon as the previous call returns and a newer frame exists --
    it must not wait out the full PERCEIVE_FLOOR_CHECK_COOLDOWN_SECONDS a
    second time."""
    scheduler, client = _scheduler([None, None, None], cooldown_seconds=10.0)

    assert scheduler.maybe_trigger(b"jpeg-1", "floor_suspect", now=0.0) is True
    scheduler.take_result()
    scheduler.note_positive()  # simulate a qualifying first positive answer

    # Nowhere near the 10s cooldown, but the 1s follow-up gap has elapsed.
    assert scheduler.maybe_trigger(b"jpeg-2", "floor_suspect", now=1.2) is True
    assert len(client.calls) == 2


def test_scheduler_still_enforces_the_one_second_follow_up_gap():
    scheduler, client = _scheduler([None, None], cooldown_seconds=10.0)

    scheduler.maybe_trigger(b"jpeg-1", "floor_suspect", now=0.0)
    scheduler.take_result()
    scheduler.note_positive()

    assert scheduler.maybe_trigger(b"jpeg-2", "floor_suspect", now=0.5) is False
    assert len(client.calls) == 1


def test_scheduler_never_fires_a_second_request_while_one_is_in_flight():
    client = FakeFloorCheckClient([FloorCheckResult(person_on_floor=True, confidence=0.9)])
    held_jobs = []
    scheduler = FloorCheckScheduler(client=client, submit=held_jobs.append)

    first = scheduler.maybe_trigger(b"jpeg-1", "floor_suspect", now=0.0)
    second = scheduler.maybe_trigger(b"jpeg-2", "floor_suspect", now=20.0)

    assert first is True
    assert second is False
    assert len(client.calls) == 0  # the held job never ran
    assert len(held_jobs) == 1


def test_scheduler_swallows_a_raising_client():
    class RaisingClient:
        def check(self, jpeg):
            raise RuntimeError("model process died")

    scheduler = FloorCheckScheduler(client=RaisingClient(), submit=synchronous_submit)

    fired = scheduler.maybe_trigger(b"jpeg", "floor_suspect", now=0.0)

    assert fired is True
    result = scheduler.take_result()
    assert result is not None
    answer, _reason, _triggered_at, _latency_ms = result
    assert answer is None


def test_scheduler_take_result_is_none_before_anything_completes():
    scheduler, _client = _scheduler([])
    assert scheduler.take_result() is None
