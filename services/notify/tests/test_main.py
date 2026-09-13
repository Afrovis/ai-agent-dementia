"""Tests for the `notify` consume loop, using `FakeBus` and `LoggingBackend`.

No real Redis, no real HTTP, no real sleep: the critical-repeat behaviour
is driven by an injected `clock` callable so tests can fast-forward
"time" by simply returning a later value.
"""

from nc_shared.bus import FakeBus
from nc_shared.events import Ack, Notify

from notify.backends import DryRunBackend, LoggingBackend, NtfyBackend
from notify.main import (
    ACK_STREAM,
    CRITICAL_REPEAT_INTERVAL_S,
    NOTIFY_STREAM,
    NotifyState,
    consume_once,
    make_backend,
)


def _fake_clock(start: float = 0.0):
    """Return a mutable clock: `clock()` reads, `clock.set(t)` fast-forwards."""

    box = {"t": start}

    def clock() -> float:
        return box["t"]

    def set_time(t: float) -> None:
        box["t"] = t

    clock.set = set_time  # type: ignore[attr-defined]
    return clock


def test_info_notification_is_sent_once_and_not_tracked():
    bus = FakeBus()
    backend = LoggingBackend()
    state = NotifyState()
    bus.publish(Notify(source="agent", level="info", title="t", body="b", repeat_until_ack=False))

    sent = consume_once(bus, backend, state, clock=_fake_clock())

    assert sent == 1
    assert backend.sent == [("info", "t", "b")]
    assert state.outstanding == {}


def test_attention_notification_is_sent_once_and_not_tracked():
    bus = FakeBus()
    backend = LoggingBackend()
    state = NotifyState()
    bus.publish(
        Notify(source="agent", level="attention", title="t", body="b", repeat_until_ack=False)
    )

    sent = consume_once(bus, backend, state, clock=_fake_clock())

    assert sent == 1
    assert state.outstanding == {}


def test_critical_notification_is_tracked_for_repeat():
    bus = FakeBus()
    backend = LoggingBackend()
    state = NotifyState()
    bus.publish(
        Notify(source="agent", level="critical", title="Fall", body="b", repeat_until_ack=True)
    )
    clock = _fake_clock(1000.0)

    sent = consume_once(bus, backend, state, clock=clock)

    assert sent == 1
    assert len(state.outstanding) == 1
    (notify_id, item) = next(iter(state.outstanding.items()))
    assert item.next_due_at == 1000.0 + CRITICAL_REPEAT_INTERVAL_S


def test_critical_notification_resends_once_due_and_not_before():
    bus = FakeBus()
    backend = LoggingBackend()
    state = NotifyState()
    bus.publish(
        Notify(source="agent", level="critical", title="Fall", body="b", repeat_until_ack=True)
    )
    clock = _fake_clock(0.0)

    consume_once(bus, backend, state, clock=clock)  # initial send
    assert len(backend.sent) == 1

    # Not due yet: 30s < 60s interval.
    clock.set(30.0)
    consume_once(bus, backend, state, clock=clock)
    assert len(backend.sent) == 1

    # Due: 60s have passed.
    clock.set(60.0)
    consume_once(bus, backend, state, clock=clock)
    assert len(backend.sent) == 2
    assert backend.sent[1] == ("critical", "Fall", "b")

    # Still tracked, next due 60s later.
    (notify_id, item) = next(iter(state.outstanding.items()))
    assert item.next_due_at == 120.0


def test_critical_notification_stops_repeating_once_acknowledged():
    bus = FakeBus()
    backend = LoggingBackend()
    state = NotifyState()
    bus.ensure_group(NOTIFY_STREAM, "notify")
    msg_id = bus.publish(
        Notify(source="agent", level="critical", title="Fall", body="b", repeat_until_ack=True)
    )
    clock = _fake_clock(0.0)

    consume_once(bus, backend, state, clock=clock)
    assert len(state.outstanding) == 1
    assert msg_id in state.outstanding

    bus.ensure_group(ACK_STREAM, "notify")
    bus.publish(Ack(source="dashboard", notify_id=msg_id))

    clock.set(60.0)
    sent_count = consume_once(bus, backend, state, clock=clock)

    assert state.outstanding == {}
    # Only the ack-processing pass ran; no repeat send happened.
    assert sent_count == 0
    assert len(backend.sent) == 1


def test_ack_helper_removes_outstanding_entry_directly():
    from notify.main import ack

    state = NotifyState()
    bus = FakeBus()
    backend = LoggingBackend()
    bus.publish(
        Notify(source="agent", level="critical", title="t", body="b", repeat_until_ack=True)
    )
    consume_once(bus, backend, state, clock=_fake_clock())
    (notify_id,) = state.outstanding.keys()

    assert ack(state, notify_id) is True
    assert notify_id not in state.outstanding
    assert ack(state, notify_id) is False


def test_dry_run_overrides_configured_ntfy_delivery():
    backend = make_backend({"DRY_RUN": "true", "NTFY_URL": "https://ntfy.sh/must-not-be-contacted"})

    assert isinstance(backend, DryRunBackend)


def test_delivery_uses_ntfy_when_dry_run_is_off():
    backend = make_backend({"DRY_RUN": "false", "NTFY_URL": "https://ntfy.sh/test"})

    assert isinstance(backend, NtfyBackend)


def test_invalid_dry_run_value_fails_before_delivery_backend_is_built():
    try:
        make_backend({"DRY_RUN": "treu", "NTFY_URL": "https://ntfy.sh/must-not-be-contacted"})
    except ValueError as exc:
        assert "DRY_RUN" in str(exc)
    else:
        raise AssertionError("invalid DRY_RUN value was accepted")
