from nc_shared.bus import FakeBus
from nc_shared.events import LightCommand, Notify

from light.backends import DisabledBackend
from light.main import LightState, consume_once, make_backend


class FailingBackend:
    def set_state(self, on: bool) -> bool:
        return False


def test_feature_flag_defaults_to_disabled_backend():
    assert isinstance(make_backend({}), DisabledBackend)


def test_enabled_shelly_requires_device_url():
    try:
        make_backend({"LIGHT_ENABLED": "true"})
    except ValueError as exc:
        assert "LIGHT_DEVICE_URL" in str(exc)
    else:
        raise AssertionError("missing device URL was accepted")


def test_consume_once_applies_and_acks_command():
    bus = FakeBus()
    backend = DisabledBackend()
    bus.publish(
        LightCommand(
            source="agent",
            light="hallway",
            state="on",
            reason="restroom_goal_started",
        )
    )
    state = LightState()
    assert consume_once(bus, backend, state) == 1
    assert backend.states == [True]
    assert state.ok


def test_failure_notifies_caregiver_once_until_recovery():
    bus = FakeBus()
    for desired in ("on", "off"):
        bus.publish(LightCommand(source="agent", light="hallway", state=desired, reason="test"))
    state = LightState()
    assert consume_once(bus, FailingBackend(), state) == 2
    bus.ensure_group("notify", "test")
    notifications = [
        event
        for _msg_id, event in bus.read("notify", "test", "test-1")
        if isinstance(event, Notify)
    ]
    assert len(notifications) == 1
    assert notifications[0].level == "attention"
    assert not state.ok
