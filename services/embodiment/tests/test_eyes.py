"""Hardware-free tests for the bedside eyes state."""

import pytest
from nc_shared.events import (
    Ack,
    Gaze,
    Notify,
    PersonState,
    SessionState,
    Show,
    SpeechStarted,
    Utterance,
)

from embodiment.eyes import DOZE_SECONDS, EyesState


def person(state):
    return PersonState(source="perceive", state=state, confidence=1, zone="other")


def show(face):
    return Show(source="agent", face=face, headline="", body="", brightness=0.5)


def session(phase):
    return SessionState(source="agent", phase=phase, goal="test", strategy_index=0)


def notify():
    return Notify(
        source="agent", level="attention", title="test", body="test", repeat_until_ack=True
    )


@pytest.mark.parametrize(
    ("posture", "expression"),
    [
        ("in_bed", "sleepy"),
        ("sitting_up", "sleepy"),
        ("standing", "open"),
        ("walking", "open"),
        ("on_floor", "open"),
        ("absent", "open"),
    ],
)
def test_posture_expression(posture, expression):
    eyes = EyesState()
    assert eyes.state == {
        "type": "eyes",
        "expression": "open",
        "alert": False,
        "gaze": {"target": "none", "x": None, "y": None},
    }
    eyes.consume(person(posture))
    assert eyes.state["expression"] == expression


def test_speech_started_times_out_on_tick_and_utterance_ends_listening():
    eyes = EyesState(clock=lambda: 0)
    eyes.consume(person("in_bed"), now=0)
    assert eyes.consume(SpeechStarted(source="listen"), now=1)["expression"] == "listening"
    eyes.consume(SpeechStarted(source="listen"), now=10)
    assert eyes.tick(24) is None
    assert eyes.tick(25)["expression"] == "sleepy"
    eyes.consume(SpeechStarted(source="listen"), now=30)
    assert (
        eyes.consume(Utterance(source="listen", text="hi", confidence=1, duration_s=1), now=31)[
            "expression"
        ]
        == "sleepy"
    )
    assert eyes.tick(31 + DOZE_SECONDS - 0.1) is None
    assert eyes.tick(31 + DOZE_SECONDS)["expression"] == "sleeping"


def test_show_override_and_posture_interaction():
    eyes = EyesState(clock=lambda: 0)
    eyes.consume(person("sitting_up"))
    assert eyes.consume(show("speaking"))["expression"] == "speaking"
    eyes.consume(person("in_bed"))
    assert eyes.state["expression"] == "speaking"
    assert eyes.consume(show("listening"))["expression"] == "listening"
    eyes.consume(SpeechStarted(source="listen"), now=0)
    eyes.consume(Utterance(source="listen", text="hi", confidence=1, duration_s=1), now=1)
    assert eyes.state["expression"] == "listening"
    assert eyes.consume(show("asleep"))["expression"] == "sleepy"
    assert eyes.consume(show("awake")) is None


def test_show_override_expires_after_fifteen_seconds():
    eyes = EyesState(clock=lambda: 0)
    eyes.consume(person("in_bed"), now=0)
    assert eyes.consume(show("listening"), now=100)["expression"] == "listening"
    assert eyes.tick(114) is None
    assert eyes.tick(115)["expression"] == "sleepy"
    assert eyes.tick(115 + DOZE_SECONDS)["expression"] == "sleeping"


@pytest.mark.parametrize("posture", ["in_bed", "sitting_up"])
def test_resting_eyes_doze_then_sleep(posture):
    eyes = EyesState(clock=lambda: 0)
    assert eyes.consume(person(posture), now=10)["expression"] == "sleepy"
    assert eyes.tick(10 + DOZE_SECONDS - 0.1) is None
    assert eyes.tick(10 + DOZE_SECONDS)["expression"] == "sleeping"
    assert eyes.consume(person("standing"), now=100)["expression"] == "open"


def test_lying_down_keeps_dozing_and_sitting_up_restarts_it():
    eyes = EyesState(clock=lambda: 0)
    eyes.consume(person("sitting_up"), now=0)
    assert eyes.consume(person("in_bed"), now=10) is None
    assert eyes.tick(DOZE_SECONDS)["expression"] == "sleeping"
    assert eyes.consume(person("in_bed"), now=60) is None
    assert eyes.consume(person("sitting_up"), now=70)["expression"] == "sleepy"
    assert eyes.consume(person("sitting_up"), now=80) is None
    assert eyes.tick(70 + DOZE_SECONDS)["expression"] == "sleeping"


def test_show_speaking_after_speech_started_takes_over():
    eyes = EyesState(clock=lambda: 0)
    eyes.consume(SpeechStarted(source="listen"), now=0)
    assert eyes.consume(show("speaking"), now=1)["expression"] == "speaking"
    assert eyes.tick(15) is None


def test_gaze_is_latest_event_as_is():
    eyes = EyesState()
    assert eyes.consume(Gaze(source="perceive", target="face", x=0.2, y=0.7))["gaze"] == {
        "target": "face",
        "x": 0.2,
        "y": 0.7,
    }
    assert eyes.consume(Gaze(source="perceive", target="none"))["gaze"] == {
        "target": "none",
        "x": None,
        "y": None,
    }


def test_alert_ack_matching_id_stays_off_until_next_escalation():
    eyes = EyesState(clock=lambda: 0)
    assert eyes.consume(session("ESCALATED"), now=20)["alert"] is True
    eyes.consume(notify(), msg_id="123-0", now=21)
    assert eyes.consume(Ack(source="dashboard", notify_id="other"), now=22) is None
    assert eyes.state["alert"] is True
    assert eyes.consume(Ack(source="dashboard", notify_id="123-0"), now=23)["alert"] is False
    eyes.consume(notify(), msg_id="124-0", now=24)
    eyes.consume(session("ESCALATED"), now=25)
    assert eyes.state["alert"] is False
    eyes.consume(session("ENGAGED"), now=26)
    assert eyes.consume(session("ESCALATED"), now=27)["alert"] is True
    assert eyes.consume(session("IDLE"), now=28)["alert"] is False


def test_notify_in_ten_seconds_before_entry_can_be_acknowledged():
    eyes = EyesState(clock=lambda: 0)
    eyes.consume(notify(), msg_id="old", now=9)
    eyes.consume(notify(), msg_id="recent", now=10)
    eyes.consume(session("ESCALATED"), now=20)
    assert eyes.consume(Ack(source="dashboard", notify_id="old"), now=20) is None
    assert eyes.consume(Ack(source="dashboard", notify_id="recent"), now=20)["alert"] is False
