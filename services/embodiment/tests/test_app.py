"""Tests for the `embodiment` FastAPI app, using `FakeBus` (no Redis, no real TLS)."""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nc_shared.bus import FakeBus
from nc_shared.events import (
    Activity,
    AudioChunk,
    BedZoneStatus,
    CalibrateBed,
    DebugControl,
    GoalChanged,
    Health,
    PersonState,
    PoseDebug,
    RawFrame,
    ResetSession,
    Say,
    SessionState,
    Show,
    SpeechStarted,
    Utterance,
)
from nc_shared.replay import CAPPED_MAXLEN

from embodiment.app import (
    DEMO_PHOTO_DIR,
    ClientSocket,
    ConnectionManager,
    broadcast_loop,
    create_app,
    handle_media_message,
    publish_audio_chunk,
    publish_frame,
    publish_playback,
    resolve_photo,
    resolve_voice_clip,
)


class FakeSpeech:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.synthesized: list[str] = []
        self.pre_rendered: tuple[str, ...] = ()

    def synthesize(self, text: str) -> str:
        self.synthesized.append(text)
        audio_id = "a" * 64
        (self.directory / f"{audio_id}.wav").write_bytes(b"RIFFfake-wave")
        return audio_id

    def pre_render(self, phrases) -> int:
        self.pre_rendered = tuple(phrases)
        return len(self.pre_rendered)

    def resolve(self, audio_id: str) -> Path | None:
        path = self.directory / f"{audio_id}.wav"
        return path if path.is_file() else None


def test_index_serves_face_page():
    bus = FakeBus()
    app = create_app(bus)
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'id="face"' in response.text


def test_page_and_static_files_are_revalidated():
    bus = FakeBus()
    app = create_app(bus)
    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/static/script.js")

    assert page.headers["cache-control"] == "no-cache"
    assert script.headers["cache-control"] == "no-cache"


def test_static_css_and_js_are_served():
    bus = FakeBus()
    app = create_app(bus)
    with TestClient(app) as client:
        css = client.get("/static/style.css")
        js = client.get("/static/script.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]


def test_photo_route_serves_an_uploaded_photo(tmp_path):
    (tmp_path / "family_photo.jpg").write_bytes(b"\xff\xd8\xff-not-really-a-jpeg")
    bus = FakeBus()
    app = create_app(bus, photo_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/photos/family_photo")

    assert response.status_code == 200
    assert response.content == b"\xff\xd8\xff-not-really-a-jpeg"


def test_photo_route_404s_when_the_caregiver_uploaded_nothing(tmp_path):
    bus = FakeBus()
    app = create_app(bus, photo_dir=tmp_path)
    with TestClient(app) as client:
        response = client.get("/photos/family_photo")

    assert response.status_code == 404


def test_resolve_photo_finds_each_supported_extension(tmp_path):
    for index, extension in enumerate((".jpg", ".jpeg", ".png", ".webp")):
        photo_id = f"photo{index}"
        (tmp_path / f"{photo_id}{extension}").write_bytes(b"x")
        assert resolve_photo(tmp_path, photo_id) == tmp_path / f"{photo_id}{extension}"


def test_resolve_photo_rejects_ids_that_would_escape_the_photo_dir(tmp_path):
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"private")
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()

    assert resolve_photo(photo_dir, "../secret") is None
    assert resolve_photo(photo_dir, "/etc/passwd") is None
    assert resolve_photo(photo_dir, "..") is None


def test_bundled_demo_photos_resolve_on_a_fresh_checkout(tmp_path):
    # The fake agent ships these ids, so `docker compose up` must not 404.
    for photo_id in ("demo_room", "demo_family"):
        resolved = resolve_photo(tmp_path, photo_id)
        assert resolved is not None, photo_id
        assert resolved.parent == DEMO_PHOTO_DIR


def test_uploaded_photo_takes_precedence_over_a_demo_photo(tmp_path):
    (tmp_path / "demo_room.jpg").write_bytes(b"the caregiver's own photo")

    assert resolve_photo(tmp_path, "demo_room") == tmp_path / "demo_room.jpg"


def test_demo_photos_can_be_disabled(tmp_path):
    assert resolve_photo(tmp_path, "demo_room", demo_dir=None) is None


def test_photo_route_serves_the_demo_photos_the_fake_agent_publishes(tmp_path):
    # The ids are duplicated from the fake agent rather than imported: the
    # two services are separate packages, so embodiment's tests cannot
    # import `agent`. `test_photo_ids_are_demo_ids` on the agent side pins
    # the other half of this contract.
    bus = FakeBus()
    app = create_app(bus, photo_dir=tmp_path)
    with TestClient(app) as client:
        for photo_id in ("demo_room", "demo_family"):
            assert client.get(f"/photos/{photo_id}").status_code == 200, photo_id


def test_photo_route_rejects_a_traversal_id(tmp_path):
    (tmp_path / "secret.jpg").write_bytes(b"private")
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    bus = FakeBus()
    app = create_app(bus, photo_dir=photo_dir)
    with TestClient(app) as client:
        response = client.get("/photos/..%2Fsecret")

    assert response.status_code == 404
    assert b"private" not in response.content


def test_broadcast_loop_forwards_show_event_to_connected_clients():
    bus = FakeBus()
    manager = ConnectionManager()
    bus.publish(
        Show(
            source="agent",
            face="awake",
            headline="It is night",
            body="Let's rest.",
            photo_id=None,
            brightness=0.5,
        )
    )

    asyncio.run(broadcast_loop(bus, manager, max_iterations=1))

    assert manager.last_show is not None
    assert manager.last_show["type"] == "show"
    assert manager.last_show["face"] == "awake"
    assert manager.last_show["headline"] == "It is night"


def test_debug_messages_forward_and_late_join_caches_person_and_session():
    bus = FakeBus()
    manager = ConnectionManager()
    bus.publish(PersonState(source="perceive", state="standing", confidence=0.8, zone="other"))
    bus.publish(
        SessionState(source="agent", phase="ENGAGED", goal="return_to_bed", strategy_index=2)
    )
    bus.publish(
        GoalChanged(source="agent", from_goal="root", to_goal="return_to_bed", reason="test")
    )
    bus.publish(Utterance(source="listen", text="hello", confidence=0.9, duration_s=0.5))
    bus.publish(
        PoseDebug(
            source="perceive", landmarks={}, bbox=None, confidence=0, detected=False, latency_ms=2
        ),
        maxlen=50,
    )
    bus.publish(
        Activity(source="agent", service="agent", kind="interpret", phase="start"), maxlen=200
    )

    class Socket:
        def __init__(self):
            self.messages = []

        async def send_text(self, raw):
            import json

            self.messages.append(json.loads(raw))

    live = Socket()
    manager._connections.append(live)
    asyncio.run(broadcast_loop(bus, manager, max_iterations=1))
    assert {msg["type"] for msg in live.messages} == {
        "person",
        "session",
        "utterance",
        "pose",
        "activity",
    }
    assert next(msg for msg in live.messages if msg["type"] == "pose")["latency_ms"] == 2
    late = Socket()
    asyncio.run(manager.send_current_state(late))
    assert [msg["type"] for msg in late.messages] == [
        "debug_config",
        "person",
        "session",
        "eyes",
        "config",
    ]


def receive_initial_eyes_and_config(websocket):
    assert websocket.receive_json()["type"] == "debug_config"
    assert websocket.receive_json() == {
        "type": "eyes",
        "expression": "open",
        "alert": False,
        "gaze": {"target": "none", "x": None, "y": None},
    }
    assert websocket.receive_json() == {
        "type": "config",
        "night_start": "20:00",
        "night_end": "07:00",
        "clock_24h": False,
    }


def test_new_websocket_replays_latest_eyes_and_config():
    bus = FakeBus()
    app = create_app(bus, night_start="21:30", night_end="06:15", clock_24h=True)
    app.state.manager.last_eyes = {
        "type": "eyes",
        "expression": "sleepy",
        "alert": True,
        "gaze": {"target": "face", "x": 0.25, "y": 0.4},
    }
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "debug_config"
        assert websocket.receive_json() == app.state.manager.last_eyes
        assert websocket.receive_json() == {
            "type": "config",
            "night_start": "21:30",
            "night_end": "06:15",
            "clock_24h": True,
        }


def test_debug_stream_forwards_applied_state_and_bed_zone_only():
    bus = FakeBus()
    manager = ConnectionManager()
    bus.publish(DebugControl(source="embodiment", time_offset_hours=9), maxlen=100)
    bus.publish(CalibrateBed(source="embodiment"), maxlen=100)
    bus.publish(ResetSession(source="embodiment"), maxlen=100)
    bus.publish(DebugControl(source="agent", time_offset_hours=3, force_in_bed=True), maxlen=100)
    bus.publish(
        BedZoneStatus(
            source="perceive", has_bed=True, polygon=[(0.1, 0.2), (0.8, 0.2), (0.8, 0.9)]
        ),
        maxlen=100,
    )

    class Socket:
        def __init__(self):
            self.messages = []

        async def send_text(self, raw):
            self.messages.append(json.loads(raw))

    live = Socket()
    manager._connections.append(live)  # noqa: SLF001
    asyncio.run(broadcast_loop(bus, manager, max_iterations=1))
    assert [item["type"] for item in live.messages] == ["debug_state", "bed_zone"]
    assert live.messages[0] == {
        "type": "debug_state",
        "time_offset_hours": 3.0,
        "force_in_bed": True,
    }
    assert live.messages[1]["polygon"] == [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9]] or live.messages[1][
        "polygon"
    ] == [(0.1, 0.2), (0.8, 0.2), (0.8, 0.9)]
    assert bus.pending("debug", "embodiment") == []
    late = Socket()
    asyncio.run(manager.send_current_state(late))
    assert [item["type"] for item in late.messages] == [
        "debug_config",
        "eyes",
        "config",
        "debug_state",
        "bed_zone",
    ]


def test_debug_controls_disabled_ignores_requests(monkeypatch):
    monkeypatch.delenv("EMBODIMENT_DEBUG_CONTROLS", raising=False)
    bus = FakeBus()
    with TestClient(create_app(bus)) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {"type": "debug_config", "enabled": False}
        websocket.send_json({"type": "debug_control", "time_offset_hours": 2, "force_in_bed": True})
        websocket.send_json({"type": "calibrate_bed"})
        websocket.send_json({"type": "reset_session"})
        websocket.send_json(client_hello())
        websocket.receive_json()
    assert bus._streams.get("debug", []) == []  # noqa: SLF001


def test_debug_controls_enabled_publishes_validated_requests(monkeypatch):
    monkeypatch.setenv("EMBODIMENT_DEBUG_CONTROLS", "yes")
    bus = FakeBus()
    with TestClient(create_app(bus)) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {"type": "debug_config", "enabled": True}
        websocket.send_json(
            {"type": "debug_control", "time_offset_hours": 100, "force_in_bed": True}
        )
        websocket.send_json(
            {"type": "debug_control", "time_offset_hours": -100, "force_in_bed": False}
        )
        websocket.send_json(
            {"type": "debug_control", "time_offset_hours": "nan", "force_in_bed": True}
        )
        websocket.send_json(
            {"type": "debug_control", "time_offset_hours": True, "force_in_bed": True}
        )
        websocket.send_json({"type": "calibrate_bed"})
        websocket.send_json({"type": "reset_session"})
        websocket.send_json(client_hello())
        websocket.receive_json()
    events = [
        {"DebugControl": DebugControl, "CalibrateBed": CalibrateBed, "ResetSession": ResetSession}[
            entry.event_type
        ].model_validate_json(entry.data)
        for entry in bus._streams["debug"]  # noqa: SLF001
    ]
    assert [type(event) for event in events] == [
        DebugControl,
        DebugControl,
        CalibrateBed,
        ResetSession,
    ]
    assert [event.time_offset_hours for event in events[:2]] == [23, -23]
    assert [event.force_in_bed for event in events[:2]] == [True, False]
    assert all(event.source == "embodiment" for event in events)


def test_websocket_receives_show_event_delivered_via_bus():
    bus = FakeBus()
    app = create_app(bus)
    show = Show(
        source="agent",
        face="listening",
        headline="Hello",
        body="I'm here.",
        photo_id=None,
        brightness=0.4,
    )
    bus.publish(show)

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "debug_config"
        # The app's own startup task runs the broadcast loop unbounded in the
        # background; give it a moment to pick up the pre-published event.
        message = websocket.receive_json()

    assert message["type"] == "show"
    assert message["face"] == "listening"
    assert message["headline"] == "Hello"


def test_websocket_delivers_say_event_published_after_connect():
    bus = FakeBus()
    app = create_app(bus)

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        receive_initial_eyes_and_config(websocket)
        bus.publish(
            Say(source="agent", text="It is night.", strategy="soft_greeting", interruptible=True)
        )
        message = websocket.receive_json()

    assert message["type"] == "say"
    assert message["text"] == "It is night."


def test_websocket_synthesizes_say_and_delivers_same_origin_audio_url(tmp_path):
    bus = FakeBus()
    speech = FakeSpeech(tmp_path)
    app = create_app(bus, speech=speech)

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        receive_initial_eyes_and_config(websocket)
        bus.publish(
            Say(
                source="agent",
                session_id="session-1",
                text="Rest now.",
                strategy="soft_greeting",
                interruptible=True,
            )
        )
        start = websocket.receive_json()
        end = websocket.receive_json()
        message = websocket.receive_json()

    assert speech.synthesized == ["Rest now."]
    assert (start["type"], start["kind"], start["phase"]) == ("activity", "tts", "start")
    assert (end["type"], end["kind"], end["phase"], end["ok"]) == ("activity", "tts", "end", True)
    assert end["duration_ms"] >= 0
    assert message["audio_url"] == f"/speech/{'a' * 64}.wav"
    assert message["session_id"] == "session-1"


def test_websocket_uses_caregiver_clip_without_calling_piper(tmp_path):
    clip_id = "family-message"
    (tmp_path / f"{clip_id}.wav").write_bytes(b"RIFFfamily-wave")
    bus = FakeBus()
    speech = FakeSpeech(tmp_path)
    app = create_app(bus, speech=speech, voice_clip_dir=tmp_path)

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        receive_initial_eyes_and_config(websocket)
        bus.publish(
            Say(
                source="agent",
                text="Here is a familiar voice for you.",
                strategy="familiar_voice",
                interruptible=True,
                clip_id=clip_id,
            )
        )
        message = websocket.receive_json()

    assert message["audio_url"] == f"/voice/{clip_id}.wav"
    assert speech.synthesized == []


def test_websocket_missing_caregiver_clip_stays_text_only_without_piper(tmp_path):
    bus = FakeBus()
    speech = FakeSpeech(tmp_path)
    app = create_app(bus, speech=speech, voice_clip_dir=tmp_path)

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        receive_initial_eyes_and_config(websocket)
        bus.publish(
            Say(
                source="agent",
                text="Here is a familiar voice for you.",
                strategy="familiar_voice",
                interruptible=True,
                clip_id="missing",
            )
        )
        message = websocket.receive_json()

    assert "audio_url" not in message
    assert speech.synthesized == []


def test_websocket_delivers_early_speech_signal_for_barge_in():
    bus = FakeBus()
    app = create_app(bus)

    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        receive_initial_eyes_and_config(websocket)
        bus.publish(SpeechStarted(source="listen", session_id="session-1"))
        assert websocket.receive_json()["expression"] == "listening"
        message = websocket.receive_json()

    assert message == {"type": "speech_started", "session_id": "session-1"}


def test_browser_stops_only_interruptible_speech_on_early_voice_signal():
    script = (Path(__file__).parents[1] / "embodiment/static/script.js").read_text()

    assert 'msg.type === "speech_started"' in script
    assert "currentSpeechInterruptible" in script
    assert "currentSpeech.pause()" in script
    assert "echoCancellation: true" in script


def test_playback_websocket_logs_and_publishes_activity(caplog):
    bus = FakeBus()
    app = create_app(bus)
    with caplog.at_level(logging.INFO, logger="embodiment"):
        with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "playback",
                    "phase": "requested",
                    "strategy": "soft_greeting",
                    "session_id": "session-1",
                    "audio_id": "abc123",
                    "latency_ms": 17,
                }
            )
            websocket.send_json(
                {
                    "type": "playback",
                    "phase": "failed",
                    "strategy": "soft_greeting",
                    "session_id": "session-1",
                    "audio_id": "abc123",
                    "latency_ms": 23,
                    "error_name": "NotAllowedError",
                    "error_message": "User gesture required",
                }
            )
            websocket.close()

    entries = bus._streams["activity"]  # noqa: SLF001
    assert len(entries) == 2
    started, failed = (Activity.model_validate_json(entry.data) for entry in entries)
    assert (started.kind, started.phase, started.ok, started.duration_ms) == (
        "playback",
        "start",
        True,
        17,
    )
    assert (failed.kind, failed.phase, failed.ok, failed.detail) == (
        "playback",
        "end",
        False,
        "NotAllowedError",
    )
    assert failed.session_id == "session-1"
    logs = [
        json.loads(record.message) for record in caplog.records if record.message.startswith("{")
    ]
    playback_logs = [item for item in logs if item.get("event_type") == "playback"]
    assert [item["playback_phase"] for item in playback_logs] == ["requested", "failed"]
    assert playback_logs[1]["audio_id"] == "abc123"
    assert playback_logs[1]["error_message"] == "User gesture required"


def test_playback_barge_in_maps_to_failed_end_activity():
    bus = FakeBus()
    event = publish_playback(
        bus,
        {
            "type": "playback",
            "phase": "interrupted",
            "strategy": "orient_time_place",
            "session_id": "session-1",
            "audio_id": "abc123",
            "latency_ms": 2800,
            "detail": "barge-in",
        },
    )
    assert event is not None
    assert (event.kind, event.phase, event.ok, event.detail) == (
        "playback",
        "end",
        False,
        "barge-in",
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "playback", "phase": "bogus", "latency_ms": 1},
        {"type": "playback", "phase": ["ended"], "latency_ms": 1},
        {"type": "playback", "phase": "ended", "latency_ms": -1},
        {"type": "playback", "phase": "ended", "latency_ms": 1, "strategy": "x" * 129},
        {"type": "playback", "phase": "failed", "latency_ms": 1, "error_message": "x" * 161},
    ],
)
def test_playback_websocket_ignores_invalid_reports(bad):
    bus = FakeBus()
    app = create_app(bus)
    with TestClient(app) as client, client.websocket_connect("/ws") as websocket:
        websocket.send_json(bad)
        websocket.send_text("x" * 2049)
        websocket.send_json({"type": "playback", "phase": "no_audio", "latency_ms": 2})
        websocket.close()

    entries = bus._streams["activity"]  # noqa: SLF001
    assert len(entries) == 1
    event = Activity.model_validate_json(entries[0].data)
    assert (event.phase, event.ok, event.detail) == ("end", False, "no_audio")


def test_speech_route_serves_cached_wav(tmp_path):
    bus = FakeBus()
    speech = FakeSpeech(tmp_path)
    audio_id = speech.synthesize("Rest now.")
    app = create_app(bus, speech=speech)

    with TestClient(app) as client:
        response = client.get(f"/speech/{audio_id}.wav")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFfake-wave"


def test_speech_route_rejects_unknown_audio_id(tmp_path):
    app = create_app(FakeBus(), speech=FakeSpeech(tmp_path))

    with TestClient(app) as client:
        response = client.get(f"/speech/{'b' * 64}.wav")

    assert response.status_code == 404


def test_voice_route_serves_only_existing_valid_clip_ids(tmp_path):
    (tmp_path / "family-message.wav").write_bytes(b"RIFFfamily-wave")
    app = create_app(FakeBus(), voice_clip_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get("/voice/family-message.wav")
        missing = client.get("/voice/missing.wav")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"RIFFfamily-wave"
    assert missing.status_code == 404


@pytest.mark.parametrize(
    "url",
    [
        "/voice/..%2Fsecret.wav",
        "/voice/%2E%2E%5Csecret.wav",
        "/voice/Upper.wav",
    ],
)
def test_voice_route_rejects_traversal_and_invalid_ids(tmp_path, url):
    (tmp_path / "secret.wav").write_bytes(b"private")
    app = create_app(FakeBus(), voice_clip_dir=tmp_path)

    with TestClient(app) as client:
        response = client.get(url)

    assert response.status_code == 404
    assert b"private" not in response.content


def test_resolve_voice_clip_rejects_unsafe_ids(tmp_path):
    (tmp_path / "safe.wav").write_bytes(b"wave")

    assert resolve_voice_clip(tmp_path, "safe") == tmp_path / "safe.wav"
    assert resolve_voice_clip(tmp_path, "../safe") is None
    assert resolve_voice_clip(tmp_path, r"..\safe") is None


def test_app_prerenders_fixed_phrases_during_startup(tmp_path):
    speech = FakeSpeech(tmp_path)
    app = create_app(
        FakeBus(),
        speech=speech,
        prerender_phrases=("Hello Jean.", "Someone is coming to help."),
    )

    with TestClient(app):
        pass

    assert speech.pre_rendered == ("Hello Jean.", "Someone is coming to help.")


def test_publish_frame_publishes_one_frame_event_with_decoded_jpeg():
    bus = FakeBus()
    jpeg_bytes = b"\xff\xd8\xff\xd9fake-jpeg-bytes"

    event = publish_frame(
        bus,
        {
            "type": "frame",
            "jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
            "width": 320,
            "height": 240,
            "source_width": 1920,
            "source_height": 1080,
        },
        session_id="sess-1",
    )

    assert isinstance(event, RawFrame)
    assert event.jpeg == jpeg_bytes
    assert event.width == 320
    assert event.height == 240
    assert event.source_width == 1920
    assert event.source_height == 1080
    assert event.source_kind == "browser"
    assert event.source == "embodiment"
    assert event.session_id == "sess-1"

    entries = bus._streams["frames_raw"]  # noqa: SLF001 - inspecting FakeBus internals for the test
    assert len(entries) == 1
    assert RawFrame.model_validate_json(entries[0].data).jpeg == jpeg_bytes


def test_publish_frame_uses_capped_stream_maxlen():
    bus = FakeBus()
    for i in range(CAPPED_MAXLEN["frames_raw"] + 5):
        publish_frame(
            bus,
            {
                "type": "frame",
                "jpeg_b64": base64.b64encode(f"frame-{i}".encode()).decode("ascii"),
                "width": 1,
                "height": 1,
                "source_width": 1,
                "source_height": 1,
            },
        )

    assert len(bus._streams["frames_raw"]) == CAPPED_MAXLEN["frames_raw"]  # noqa: SLF001


def test_browser_letterboxes_16_by_9_and_emits_both_dimension_pairs():
    script = (Path(__file__).parents[1] / "embodiment/static/script.js").read_text()

    source_width, source_height = 1920, 1080
    target_width, target_height = 640, 480
    scale = min(target_width / source_width, target_height / source_height)
    rendered_width = round(source_width * scale)
    rendered_height = round(source_height * scale)
    assert (rendered_width, rendered_height) == (640, 360)
    assert ((target_width - rendered_width) // 2, (target_height - rendered_height) // 2) == (
        0,
        60,
    )
    assert (
        "const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight)" in script
    )
    assert "ctx.fillRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT)" in script
    assert "destination.width" in script
    assert "destination.height" in script
    assert "source_width: sourceWidth" in script
    assert "source_height: sourceHeight" in script


def test_publish_audio_chunk_publishes_one_audio_chunk_event_with_decoded_pcm16():
    bus = FakeBus()
    pcm16_bytes = b"\x00\x01\x02\x03"

    event = publish_audio_chunk(
        bus,
        {
            "type": "audio",
            "pcm16_b64": base64.b64encode(pcm16_bytes).decode("ascii"),
            "sample_rate": 16000,
        },
        session_id="sess-2",
    )

    assert isinstance(event, AudioChunk)
    assert event.pcm16 == pcm16_bytes
    assert event.sample_rate == 16000
    assert event.source == "embodiment"
    assert event.session_id == "sess-2"

    entries = bus._streams["audio_in"]  # noqa: SLF001
    assert len(entries) == 1


def test_publish_audio_chunk_uses_capped_stream_maxlen():
    bus = FakeBus()
    for i in range(CAPPED_MAXLEN["audio_in"] + 5):
        publish_audio_chunk(
            bus,
            {
                "type": "audio",
                "pcm16_b64": base64.b64encode(f"chunk-{i}".encode()).decode("ascii"),
                "sample_rate": 16000,
            },
        )

    assert len(bus._streams["audio_in"]) == CAPPED_MAXLEN["audio_in"]  # noqa: SLF001


def test_handle_media_message_ignores_unknown_type_without_raising():
    bus = FakeBus()

    asyncio.run(handle_media_message(bus, {"type": "bogus"}))

    assert bus._streams == {}  # noqa: SLF001 - nothing should have been published


def test_handle_media_message_ignores_malformed_frame_without_raising():
    bus = FakeBus()

    asyncio.run(handle_media_message(bus, {"type": "frame", "jpeg_b64": "not base64!!"}))

    assert "frames_raw" not in bus._streams


def test_media_websocket_publishes_frame_event():
    bus = FakeBus()
    app = create_app(bus)
    jpeg_bytes = b"jpeg-bytes-over-the-wire"

    with TestClient(app) as client, client.websocket_connect("/media") as websocket:
        websocket.send_json(
            {
                "type": "frame",
                "jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
                "width": 160,
                "height": 120,
                "source_width": 1280,
                "source_height": 720,
            }
        )
        websocket.close()

    entries = bus._streams.get("frames_raw", [])  # noqa: SLF001
    assert len(entries) == 1
    event = RawFrame.model_validate_json(entries[0].data)
    assert event.jpeg == jpeg_bytes
    assert event.width == 160
    assert event.height == 120
    assert event.source_width == 1280
    assert event.source_height == 720
    assert event.source_kind == "browser"


def test_media_websocket_publishes_audio_chunk_event():
    bus = FakeBus()
    app = create_app(bus)
    pcm16_bytes = b"\x10\x20\x30\x40"

    with TestClient(app) as client, client.websocket_connect("/media") as websocket:
        websocket.send_json(
            {
                "type": "audio",
                "pcm16_b64": base64.b64encode(pcm16_bytes).decode("ascii"),
                "sample_rate": 16000,
            }
        )
        websocket.close()

    entries = bus._streams.get("audio_in", [])  # noqa: SLF001
    assert len(entries) == 1
    event = AudioChunk.model_validate_json(entries[0].data)
    assert event.pcm16 == pcm16_bytes
    assert event.sample_rate == 16000


def test_media_websocket_survives_malformed_message():
    bus = FakeBus()
    app = create_app(bus)

    with TestClient(app) as client, client.websocket_connect("/media") as websocket:
        websocket.send_text("not even json")
        websocket.send_json({"type": "unknown_thing"})
        websocket.send_json({"type": "frame", "jpeg_b64": "!!not base64!!"})
        # The connection should still be alive after three bad messages: a
        # well-formed one right after should still get published normally.
        websocket.send_json(
            {
                "type": "audio",
                "pcm16_b64": base64.b64encode(b"ok").decode("ascii"),
                "sample_rate": 16000,
            }
        )
        websocket.close()

    assert "frames_raw" not in bus._streams
    assert len(bus._streams.get("audio_in", [])) == 1


def client_hello(page_id="page123456", **changes):
    return {
        "type": "hello",
        "page_id": page_id,
        "device_id": "device123456",
        "user_agent": "Mozilla/5.0 Safari/605.1",
        "platform": "MacIntel",
        "screen": {"width": 1440, "height": 900},
        "visibility": "visible",
        "audio_unlocked": False,
        "page_load_time": "2026-09-22T12:00:00Z",
        **changes,
    }


def test_hello_registers_both_sockets_and_disconnect_logs_duration(caplog):
    bus = FakeBus()
    app = create_app(bus)
    with caplog.at_level(logging.INFO, logger="embodiment"):
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws, client.websocket_connect("/media") as media:
                receive_initial_eyes_and_config(ws)
                ws.send_json(client_hello())
                assert ws.receive_json()["pages"][0]["page_id"] == "page1234"
                media.send_json(client_hello())
                media.close(code=1000)
                ws.close(code=1000)
    logs = [
        json.loads(record.message) for record in caplog.records if record.message.startswith("{")
    ]
    hellos = [item for item in logs if item.get("event_type") == "client hello"]
    closes = [item for item in logs if item.get("event_type") == "client disconnected"]
    assert {item["channel"] for item in hellos} == {"ws", "media"}
    assert {item["channel"] for item in closes} == {"ws", "media"}
    assert all(item["duration_s"] >= 0 and item["close_code"] == 1000 for item in closes)
    activities = [Activity.model_validate_json(entry.data) for entry in bus._streams["activity"]]  # noqa: SLF001
    assert any(item.kind == "client" and item.phase == "start" for item in activities)
    assert any(item.kind == "client" and item.phase == "end" for item in activities)
    assert Health.model_validate_json(bus._streams["health"][0].data).detail == "1 page connected"  # noqa: SLF001


def test_two_pages_get_count_and_health_update():
    bus = FakeBus()
    app = create_app(bus)

    def clients_message(socket, count):
        for _ in range(6):
            message = socket.receive_json()
            if message["type"] == "clients" and len(message["pages"]) == count:
                return message
        raise AssertionError("clients message was not sent")

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
            first.send_json(client_hello(page_id="firstpage"))
            assert clients_message(first, 1)["pages"][0]["page_id"] == "firstpag"
            second.send_json(client_hello(page_id="secondpage"))
            assert len(clients_message(second, 2)["pages"]) == 2
            first.close()
            second.close()
    details = [Health.model_validate_json(entry.data).detail for entry in bus._streams["health"]]  # noqa: SLF001
    assert "2 pages connected" in details


def test_broadcast_logs_recipients_and_send_failures(caplog):
    manager = ConnectionManager()

    class Socket:
        def __init__(self, fail=False):
            self.fail = fail
            self.messages = []

        async def send_text(self, raw):
            if self.fail:
                raise RuntimeError("closed")
            self.messages.append(json.loads(raw))

    live, failed = Socket(), Socket(True)
    manager._connections.extend([live, failed])
    manager._clients[live] = ClientSocket(  # noqa: SLF001
        "ws",
        "10.0.0.1",
        time.monotonic(),
        page_id="page123456",
        visibility="visible",
        audio_unlocked=True,
    )
    manager._clients[failed] = ClientSocket(  # noqa: SLF001
        "ws",
        "10.0.0.2",
        time.monotonic(),
        page_id="page654321",
        visibility="hidden",
        audio_unlocked=False,
    )
    with caplog.at_level(logging.INFO, logger="embodiment"):
        asyncio.run(manager.broadcast({"type": "say", "strategy": "soft_greeting"}))
    log = next(
        json.loads(record.message)
        for record in caplog.records
        if '"event_type": "broadcast"' in record.message
    )
    assert log["type"] == "say" and len(log["recipients"]) == 1
    assert log["recipients"] == [
        {"page_id": "page123456", "visibility": "visible", "audio_unlocked": True}
    ]
    assert log["failures"] == ["page654321"]
    assert failed not in manager._connections


def test_stale_and_bad_hello_are_ignored(caplog):
    bus = FakeBus()
    app = create_app(bus)
    manager = app.state.manager
    with caplog.at_level(logging.INFO, logger="embodiment"):
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            receive_initial_eyes_and_config(ws)
            ws.send_json(client_hello(page_id="x" * 65))
            ws.send_json(client_hello(user_agent="x" * 300))
            ws.send_text(json.dumps(client_hello()) + " " * 2100)
            ws.send_json(client_hello())
            assert ws.receive_json()["pages"][0]["page_id"] == "page1234"
            socket = next(iter(manager._clients))  # noqa: SLF001
            state = manager._clients[socket]  # noqa: SLF001
            client.portal.call(manager.mark_stale, state.last_heartbeat + 31)
            assert state.stale
            ws.close()
    hellos = [
        record for record in caplog.records if '"event_type": "client hello"' in record.message
    ]
    assert len(hellos) == 1
    assert any('"event_type": "client stale"' in record.message for record in caplog.records)
