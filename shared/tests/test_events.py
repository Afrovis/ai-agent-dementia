"""Tests for nc_shared.events: instantiation, JSON round-trip, validation."""

import pytest
from pydantic import ValidationError

from nc_shared.events import (
    Ack,
    AudioChunk,
    CloudCall,
    Frame,
    GoalChanged,
    Health,
    LightCommand,
    Notify,
    PersonState,
    Say,
    SessionState,
    Show,
    SpeechStarted,
    Utterance,
)


def _roundtrip(event):
    """Serialise `event` to JSON and back, returning the reconstructed copy."""
    cls = type(event)
    return cls.model_validate_json(event.model_dump_json())


def test_frame_roundtrip():
    event = Frame(
        source="capture",
        jpeg=b"\xff\xd8\xff\xe0not-really-a-jpeg",
        width=640,
        height=480,
        source_width=1920,
        source_height=1080,
        source_kind="browser",
    )
    copy = _roundtrip(event)
    assert copy.jpeg == event.jpeg
    assert copy.width == 640
    assert copy.height == 480
    assert copy.source_width == 1920
    assert copy.source_height == 1080
    assert copy.source_kind == "browser"
    assert copy.session_id is None
    assert copy.ts == event.ts


def test_person_state_roundtrip():
    event = PersonState(
        source="perceive",
        session_id="sess-1",
        state="sitting_up",
        confidence=0.87,
        zone="bed",
        scene_note="stirring",
    )
    copy = _roundtrip(event)
    assert copy.state == "sitting_up"
    assert copy.confidence == pytest.approx(0.87)
    assert copy.zone == "bed"
    assert copy.scene_note == "stirring"
    assert copy.session_id == "sess-1"


def test_person_state_rejects_invalid_state():
    with pytest.raises(ValidationError):
        PersonState(
            source="perceive",
            state="dancing",
            confidence=0.5,
            zone="bed",
        )


def test_person_state_rejects_invalid_zone():
    with pytest.raises(ValidationError):
        PersonState(
            source="perceive",
            state="in_bed",
            confidence=0.5,
            zone="kitchen",
        )


def test_utterance_roundtrip():
    event = Utterance(source="listen", text="I need the toilet", confidence=0.92, duration_s=1.4)
    copy = _roundtrip(event)
    assert copy.text == "I need the toilet"
    assert copy.confidence == pytest.approx(0.92)
    assert copy.duration_s == pytest.approx(1.4)


def test_speech_started_roundtrip():
    event = SpeechStarted(source="listen", session_id="sess-1")
    copy = _roundtrip(event)
    assert copy.source == "listen"
    assert copy.session_id == "sess-1"


def test_session_state_roundtrip():
    event = SessionState(source="agent", phase="ENGAGED", goal="return_to_bed", strategy_index=2)
    copy = _roundtrip(event)
    assert copy.phase == "ENGAGED"
    assert copy.goal == "return_to_bed"
    assert copy.strategy_index == 2


def test_session_state_rejects_invalid_phase():
    with pytest.raises(ValidationError):
        SessionState(source="agent", phase="SLEEPING", goal="return_to_bed", strategy_index=0)


def test_goal_changed_roundtrip():
    event = GoalChanged(
        source="agent",
        from_goal="return_to_bed",
        to_goal="restroom",
        reason="mentioned toilet",
    )
    copy = _roundtrip(event)
    assert copy.from_goal == "return_to_bed"
    assert copy.to_goal == "restroom"
    assert copy.reason == "mentioned toilet"


def test_cloud_call_is_text_only_and_roundtrips_exact_payload():
    event = CloudCall(
        source="agent",
        session_id="sess-1",
        task="interpret",
        model="claude-opus-5",
        payload={"task": "classify", "input": {"utterance": "I need help"}},
    )
    assert _roundtrip(event).payload == event.payload

    with pytest.raises(ValidationError):
        CloudCall(
            source="agent",
            task="plan",
            model="claude-opus-5",
            payload={"audio": b"not allowed"},
        )


def test_light_command_roundtrip():
    event = LightCommand(
        source="agent",
        session_id="sess-1",
        light="hallway",
        state="on",
        reason="restroom_goal_started",
    )
    copy = _roundtrip(event)
    assert copy.light == "hallway"
    assert copy.state == "on"
    assert copy.reason == "restroom_goal_started"


def test_say_roundtrip():
    event = Say(
        source="agent",
        text="Let's head back to bed.",
        strategy="guided_return",
        interruptible=True,
    )
    copy = _roundtrip(event)
    assert copy.text == "Let's head back to bed."
    assert copy.strategy == "guided_return"
    assert copy.interruptible is True
    assert copy.clip_id is None


def test_say_roundtrip_with_caregiver_clip_id():
    event = Say(
        source="agent",
        text="Here is a familiar voice for you.",
        strategy="familiar_voice",
        interruptible=True,
        clip_id="toms-message",
    )

    assert _roundtrip(event).clip_id == "toms-message"


def test_show_roundtrip():
    event = Show(
        source="agent",
        face="speaking",
        headline="It is night",
        body="Time to rest",
        photo_id=None,
        brightness=0.3,
    )
    copy = _roundtrip(event)
    assert copy.face == "speaking"
    assert copy.headline == "It is night"
    assert copy.body == "Time to rest"
    assert copy.photo_id is None
    assert copy.brightness == pytest.approx(0.3)


def test_show_rejects_invalid_face():
    with pytest.raises(ValidationError):
        Show(source="agent", face="grinning", headline="h", body="b", brightness=0.5)


def test_notify_roundtrip():
    event = Notify(
        source="agent",
        level="critical",
        title="Fell",
        body="On floor for 2 minutes",
        repeat_until_ack=True,
    )
    copy = _roundtrip(event)
    assert copy.level == "critical"
    assert copy.repeat_until_ack is True


def test_notify_rejects_invalid_level():
    with pytest.raises(ValidationError):
        Notify(source="agent", level="urgent", title="t", body="b", repeat_until_ack=False)


def test_ack_roundtrip():
    event = Ack(source="dashboard", notify_id="notif-123")
    copy = _roundtrip(event)
    assert copy.notify_id == "notif-123"


def test_audio_chunk_roundtrip():
    event = AudioChunk(source="embodiment", pcm16=b"\x00\x01\x02\x03", sample_rate=16000)
    copy = _roundtrip(event)
    assert copy.pcm16 == event.pcm16
    assert copy.sample_rate == 16000


def test_audio_chunk_rejects_invalid_sample_rate():
    with pytest.raises(ValidationError):
        AudioChunk(source="embodiment", pcm16=b"\x00\x01", sample_rate=44100)


def test_health_roundtrip():
    event = Health(source="agent", service="agent", ok=True, detail="fine")
    copy = _roundtrip(event)
    assert copy.service == "agent"
    assert copy.ok is True
    assert copy.detail == "fine"


def test_base_fields_default_session_id_none_and_ts_set():
    event = Health(source="agent", service="agent", ok=True, detail="fine")
    assert event.session_id is None
    assert event.ts is not None
