"""Tests for the `dashboard` placeholder's Health emission, using FakeBus (no Redis)."""

from nc_shared.bus import FakeBus
from nc_shared.events import Health

from dashboard.main import emit_placeholder_health


def test_emit_placeholder_health_publishes_one_event():
    bus = FakeBus()
    bus.ensure_group("health", "test-group")

    emit_placeholder_health(bus)

    read = bus.read("health", "test-group", "consumer-1")
    assert len(read) == 1
    _, event = read[0]
    assert isinstance(event, Health)
    assert event.service == "dashboard"
    assert event.ok is False
    assert event.detail == "placeholder"
