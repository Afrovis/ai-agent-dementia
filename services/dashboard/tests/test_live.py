from __future__ import annotations

from nc_shared.bus import FakeBus
from nc_shared.events import Ack, Notify, Say, SessionState, Utterance

from dashboard.live import LIVE_GROUP, LiveState, consume_live_once


def test_consumes_live_text_state_and_acks_every_message():
    bus = FakeBus()
    bus.publish(
        SessionState(
            source="agent",
            session_id="session-1",
            phase="ENGAGED",
            goal="return_to_bed",
            strategy_index=2,
        )
    )
    bus.publish(
        Utterance(
            source="listen",
            session_id="session-1",
            text="Where am I?",
            confidence=0.9,
            duration_s=1.2,
        )
    )
    bus.publish(
        Say(
            source="agent",
            session_id="session-1",
            text="You are safe at home.",
            strategy="orient_time_place",
            interruptible=True,
        )
    )
    state = LiveState()

    assert consume_live_once(bus, state) == 3
    snapshot = state.snapshot()
    assert snapshot.session.phase == "ENGAGED"
    assert snapshot.last_heard.text == "Where am I?"
    assert snapshot.last_said.text == "You are safe at home."
    assert bus.pending("session", LIVE_GROUP) == []
    assert bus.pending("speech_in", LIVE_GROUP) == []
    assert bus.pending("say", LIVE_GROUP) == []


def test_repeating_alert_is_keyed_by_message_id_and_removed_by_ack():
    bus = FakeBus()
    notify_id = bus.publish(
        Notify(
            source="agent",
            session_id="session-1",
            level="critical",
            title="Person may have fallen",
            body="Check the room now.",
            repeat_until_ack=True,
        )
    )
    state = LiveState()
    consume_live_once(bus, state)

    assert state.snapshot().alerts[0].notify_id == notify_id

    bus.publish(Ack(source="dashboard", session_id="session-1", notify_id=notify_id))
    consume_live_once(bus, state)

    assert state.snapshot().alerts == ()


def test_non_repeating_notification_does_not_stay_in_live_alerts():
    bus = FakeBus()
    bus.publish(
        Notify(
            source="agent",
            level="attention",
            title="Session escalated",
            body="Please review.",
            repeat_until_ack=False,
        )
    )
    state = LiveState()

    consume_live_once(bus, state)

    assert state.snapshot().alerts == ()
