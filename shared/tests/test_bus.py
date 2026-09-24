"""Tests for nc_shared.bus.FakeBus: publish/read/ack and pending redelivery.

`Bus` (real Redis) is not exercised here: HANDOFF.md's testing convention
requires every service to be testable with no Redis, and these tests are
meant to run with no live Redis instance.
"""

from nc_shared.bus import FakeBus
from nc_shared.events import Frame, Health, Notify, PersonState


def test_publish_read_ack_cycle():
    bus = FakeBus()
    bus.ensure_group("person", "perceive-group")

    event = PersonState(source="perceive", state="standing", confidence=0.9, zone="bed")
    msg_id = bus.publish(event)
    assert isinstance(msg_id, str)

    read = bus.read("person", "perceive-group", "consumer-1")
    assert len(read) == 1
    got_id, got_event = read[0]
    assert got_id == msg_id
    assert isinstance(got_event, PersonState)
    assert got_event.state == "standing"
    assert got_event.zone == "bed"


def test_unacked_message_stays_pending():
    bus = FakeBus()
    bus.ensure_group("health", "store-group")

    bus.publish(Health(source="agent", service="agent", ok=True, detail="fine"))
    read = bus.read("health", "store-group", "consumer-1")
    (msg_id, _) = read[0]

    assert msg_id in bus.pending("health", "store-group")

    bus.ack("health", "store-group", msg_id)
    assert msg_id not in bus.pending("health", "store-group")


def test_read_does_not_redeliver_already_read_messages():
    bus = FakeBus()
    bus.ensure_group("notify", "notify-group")
    bus.publish(
        Notify(source="agent", level="attention", title="t", body="b", repeat_until_ack=False)
    )

    first = bus.read("notify", "notify-group", "consumer-1")
    second = bus.read("notify", "notify-group", "consumer-1")

    assert len(first) == 1
    assert len(second) == 0
    # The message from the first read is still pending because it was never acked.
    assert len(bus.pending("notify", "notify-group")) == 1


def test_independent_consumer_groups_each_see_all_messages():
    bus = FakeBus()
    bus.ensure_group("person", "group-a")
    bus.ensure_group("person", "group-b")

    bus.publish(PersonState(source="perceive", state="in_bed", confidence=0.99, zone="bed"))

    read_a = bus.read("person", "group-a", "consumer-a")
    read_b = bus.read("person", "group-b", "consumer-b")

    assert len(read_a) == 1
    assert len(read_b) == 1


def test_maxlen_caps_stream_length():
    bus = FakeBus()
    bus.ensure_group("frames", "capture-group")
    for _ in range(5):
        bus.publish(
            Frame(source="capture", jpeg=b"abc", width=1, height=1, source_kind="usb"),
            maxlen=2,
        )

    assert len(bus._streams["frames"]) == 2


def test_consumer_keeps_reading_after_a_capped_stream_is_trimmed():
    """A consumer of a capped stream must not go deaf once trimming starts.

    `frames`, `frames_raw`, and `audio_in` are capped at `MAXLEN ~ 50`, and
    `capture` consumes `frames_raw` for the whole night. Trimming shifts
    every entry down the list, so a positional read cursor has to be rebased
    or the consumer stops seeing anything the moment the stream first fills.
    """
    bus = FakeBus()
    bus.ensure_group("frames", "capture-group")

    read_total = 0
    for _ in range(120):
        bus.publish(
            Frame(source="capture", jpeg=b"abc", width=1, height=1, source_kind="usb"),
            maxlen=10,
        )
        for msg_id, _ in bus.read("frames", "capture-group", "consumer-1"):
            bus.ack("frames", "capture-group", msg_id)
            read_total += 1

    assert read_total == 120


def test_trimming_drops_the_oldest_unread_entries_only():
    """A consumer slower than `maxlen` loses the oldest entries, not all of them."""
    bus = FakeBus()
    bus.ensure_group("frames", "capture-group")

    for _ in range(30):
        bus.publish(
            Frame(source="capture", jpeg=b"abc", width=1, height=1, source_kind="usb"),
            maxlen=10,
        )

    read = bus.read("frames", "capture-group", "consumer-1", count=100)
    assert len(read) == 10
    assert [msg_id for msg_id, _ in read] == [entry.msg_id for entry in bus._streams["frames"]]


def test_fake_read_many_returns_only_streams_with_new_messages():
    bus = FakeBus()
    for stream in ("person", "notify", "health"):
        bus.ensure_group(stream, "g")
    bus.publish(PersonState(source="perceive", state="in_bed", confidence=0.9, zone="bed"))
    bus.publish(Health(source="agent", service="agent", ok=True, detail="ok"))

    batch = bus.read_many(["person", "notify", "health"], "g", "c")

    assert set(batch) == {"person", "health"}
    assert isinstance(batch["person"][0][1], PersonState)
    assert bus.read_many(["person", "notify", "health"], "g", "c") == {}


def test_read_many_makes_one_xreadgroup_call_over_all_streams():
    from nc_shared.bus import Bus

    event = Health(source="agent", service="agent", ok=True, detail="ok")

    class Client:
        calls = []

        def xreadgroup(self, group, consumer, streams, count, block):
            self.calls.append((group, consumer, streams, count, block))
            fields = {b"event_type": b"Health", b"data": event.model_dump_json().encode()}
            return [[b"health", [(b"1-0", fields)]]]

    client = Client()
    batch = Bus(client).read_many(["say", "health"], "g", "c", count=5, block_ms=100)

    assert client.calls == [("g", "c", {"say": ">", "health": ">"}, 5, 100)]
    assert batch == {"health": [("1-0", event)]}
