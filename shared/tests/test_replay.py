"""Tests for nc_shared.replay: recording bus events to JSONL and replaying them.

Uses `nc_shared.bus.FakeBus` throughout, per HANDOFF.md's testing
convention that every service (and this dev tool) must be testable with no
real Redis.
"""

import io
import json

from nc_shared.bus import FakeBus
from nc_shared.events import AudioChunk, Frame, PersonState, Say
from nc_shared.replay import record_once, replay


def test_record_once_writes_one_line_per_event_across_streams():
    bus = FakeBus()
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.9, zone="bed"))
    bus.publish(Say(source="agent", text="hello", strategy="soft_greeting", interruptible=True))
    audio = AudioChunk(source="embodiment", pcm16=b"\x00\x01\x02\x03\xff", sample_rate=16000)
    bus.publish(audio)

    out = io.StringIO()
    written = record_once(bus, out)

    assert written == 3
    lines = [json.loads(line) for line in out.getvalue().splitlines() if line]
    assert len(lines) == 3

    by_stream = {line["stream"]: line for line in lines}
    assert set(by_stream) == {"person", "say", "audio_in"}

    audio_line = by_stream["audio_in"]
    assert audio_line["event_type"] == "AudioChunk"
    assert "ts" in audio_line
    assert "recorded_at" in audio_line
    assert audio_line["payload"]["sample_rate"] == 16000
    # The binary field must round-trip byte-for-byte through the JSONL payload.
    reconstructed = AudioChunk.model_validate(audio_line["payload"])
    assert reconstructed.pcm16 == audio.pcm16


def test_record_once_round_trips_frame_jpeg_bytes():
    bus = FakeBus()
    original = Frame(
        source="capture",
        jpeg=b"\xff\xd8\xff\xe0not-really-a-jpeg\x00\x01",
        width=640,
        height=480,
        source_kind="usb",
    )
    bus.publish(original)

    out = io.StringIO()
    record_once(bus, out)

    line = json.loads(out.getvalue().splitlines()[0])
    assert line["stream"] == "frames"
    assert line["event_type"] == "Frame"
    reconstructed = Frame.model_validate(line["payload"])
    assert reconstructed.jpeg == original.jpeg
    assert reconstructed.width == 640
    assert reconstructed.height == 480


def test_record_once_acks_so_a_second_pass_finds_nothing_new():
    bus = FakeBus()
    bus.publish(PersonState(source="perceive", state="in_bed", confidence=0.5, zone="bed"))

    first_pass = record_once(bus, io.StringIO())
    second_pass = record_once(bus, io.StringIO())

    assert first_pass == 1
    assert second_pass == 0


class _FakeClock:
    """Records requested sleep durations instead of actually sleeping."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def _write_jsonl(path, lines):
    with open(path, "w") as handle:
        for line in lines:
            handle.write(json.dumps(line))
            handle.write("\n")


def test_replay_reconstructs_events_onto_correct_streams(tmp_path):
    fixture = tmp_path / "night.jsonl"
    _write_jsonl(
        fixture,
        [
            {
                "stream": "person",
                "event_type": "PersonState",
                "ts": "2026-01-01T00:00:00+00:00",
                "recorded_at": 1.0,
                "payload": PersonState(
                    source="perceive", state="standing", confidence=0.8, zone="bed"
                ).model_dump(mode="json"),
            },
            {
                "stream": "audio_in",
                "event_type": "AudioChunk",
                "ts": "2026-01-01T00:00:02+00:00",
                "recorded_at": 3.0,
                "payload": AudioChunk(
                    source="embodiment", pcm16=b"\x10\x20\x30", sample_rate=16000
                ).model_dump(mode="json"),
            },
        ],
    )

    bus = FakeBus()
    clock = _FakeClock()
    published = replay(bus, str(fixture), speed=1.0, sleep_fn=clock)

    assert published == 2
    bus.ensure_group("person", "readback")
    bus.ensure_group("audio_in", "readback")
    person_read = bus.read("person", "readback", "c")
    audio_read = bus.read("audio_in", "readback", "c")

    assert len(person_read) == 1
    _, person_event = person_read[0]
    assert isinstance(person_event, PersonState)
    assert person_event.state == "standing"
    assert person_event.zone == "bed"

    assert len(audio_read) == 1
    _, audio_event = audio_read[0]
    assert isinstance(audio_event, AudioChunk)
    assert audio_event.pcm16 == b"\x10\x20\x30"


def test_replay_sleeps_proportionally_to_original_gap_scaled_by_speed(tmp_path):
    fixture = tmp_path / "night.jsonl"
    _write_jsonl(
        fixture,
        [
            {
                "stream": "person",
                "event_type": "PersonState",
                "ts": "2026-01-01T00:00:00+00:00",
                "recorded_at": 0.0,
                "payload": PersonState(
                    source="perceive", state="in_bed", confidence=0.9, zone="bed"
                ).model_dump(mode="json"),
            },
            {
                "stream": "person",
                "event_type": "PersonState",
                "ts": "2026-01-01T00:00:10+00:00",
                "recorded_at": 10.0,
                "payload": PersonState(
                    source="perceive", state="sitting_up", confidence=0.9, zone="bed"
                ).model_dump(mode="json"),
            },
            {
                "stream": "person",
                "event_type": "PersonState",
                "ts": "2026-01-01T00:00:30+00:00",
                "recorded_at": 30.0,
                "payload": PersonState(
                    source="perceive", state="standing", confidence=0.9, zone="bed"
                ).model_dump(mode="json"),
            },
        ],
    )

    bus = FakeBus()
    clock = _FakeClock()
    replay(bus, str(fixture), speed=5.0, sleep_fn=clock)

    # No sleep before the first event; gaps of 10s and 20s scaled by speed=5.
    assert clock.sleeps == [10 / 5, 20 / 5]


def test_replay_with_speed_zero_or_none_never_sleeps(tmp_path):
    fixture = tmp_path / "night.jsonl"
    _write_jsonl(
        fixture,
        [
            {
                "stream": "person",
                "event_type": "PersonState",
                "ts": "2026-01-01T00:00:00+00:00",
                "recorded_at": 0.0,
                "payload": PersonState(
                    source="perceive", state="in_bed", confidence=0.9, zone="bed"
                ).model_dump(mode="json"),
            },
            {
                "stream": "person",
                "event_type": "PersonState",
                "ts": "2026-01-01T00:05:00+00:00",
                "recorded_at": 300.0,
                "payload": PersonState(
                    source="perceive", state="standing", confidence=0.9, zone="bed"
                ).model_dump(mode="json"),
            },
        ],
    )

    for speed in (0, None):
        bus = FakeBus()
        clock = _FakeClock()
        published = replay(bus, str(fixture), speed=speed, sleep_fn=clock)
        assert published == 2
        assert clock.sleeps == []


def test_replay_applies_maxlen_to_capped_streams(tmp_path):
    from nc_shared.replay import CAPPED_MAXLEN

    fixture_count = CAPPED_MAXLEN["frames"] + 5
    lines = [
        {
            "stream": "frames",
            "event_type": "Frame",
            "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
            "recorded_at": float(i),
            "payload": Frame(
                source="capture", jpeg=b"x", width=1, height=1, source_kind="usb"
            ).model_dump(mode="json"),
        }
        for i in range(fixture_count)
    ]
    fixture = tmp_path / "frames.jsonl"
    _write_jsonl(fixture, lines)

    bus = FakeBus()
    published = replay(bus, str(fixture), speed=None, sleep_fn=lambda _: None)

    assert published == fixture_count
    assert len(bus._streams["frames"]) == CAPPED_MAXLEN["frames"]


class _XRangeClient:
    """Minimal stand-in for the redis client: stream name -> [(id, fields)]."""

    def __init__(self, streams):
        self._streams = streams

    def xrange(self, stream):
        return self._streams.get(stream, [])


def _fields(event):
    return {
        b"event_type": type(event).__name__.encode(),
        b"data": event.model_dump_json().encode(),
    }


def test_export_history_merges_streams_in_ts_order_and_skips_media():
    from datetime import UTC, datetime

    from nc_shared.replay import export_history

    def at(second):
        return datetime(2026, 9, 22, 22, 0, second, tzinfo=UTC)

    old = PersonState(source="perceive", ts=at(0), state="in_bed", confidence=0.9, zone="bed")
    earlier = PersonState(source="perceive", ts=at(1), state="standing", confidence=0.9, zone="bed")
    frame = Frame(
        source="capture", ts=at(2), jpeg=b"\xff\xd8", width=1, height=1, source_kind="usb"
    )
    later = Say(source="agent", ts=at(3), text="hi", strategy="soft_greeting", interruptible=True)
    client = _XRangeClient(
        {
            "say": [("3-0", _fields(later))],
            "person": [("0-0", _fields(old)), ("1-0", _fields(earlier))],
            "frames": [("2-0", _fields(frame))],
        }
    )

    out = io.StringIO()
    count = export_history(client, out, since=at(1), now_fn=lambda: 0.0)

    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert count == 2
    assert [line["event_type"] for line in lines] == ["PersonState", "Say"]
    assert lines[0]["payload"]["state"] == "standing"
    assert export_history(client, io.StringIO(), include_media=True) == 4
