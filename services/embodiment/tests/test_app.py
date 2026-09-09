"""Tests for the `embodiment` FastAPI app, using `FakeBus` (no Redis, no real TLS)."""

import asyncio
import base64

from fastapi.testclient import TestClient
from nc_shared.bus import FakeBus
from nc_shared.events import AudioChunk, RawFrame, Say, Show
from nc_shared.replay import CAPPED_MAXLEN

from embodiment.app import (
    DEMO_PHOTO_DIR,
    ConnectionManager,
    broadcast_loop,
    create_app,
    handle_media_message,
    publish_audio_chunk,
    publish_frame,
    resolve_photo,
)


def test_index_serves_face_page():
    bus = FakeBus()
    app = create_app(bus)
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'id="face"' in response.text


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
        bus.publish(
            Say(source="agent", text="It is night.", strategy="soft_greeting", interruptible=True)
        )
        message = websocket.receive_json()

    assert message["type"] == "say"
    assert message["text"] == "It is night."


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
        },
        session_id="sess-1",
    )

    assert isinstance(event, RawFrame)
    assert event.jpeg == jpeg_bytes
    assert event.width == 320
    assert event.height == 240
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
            },
        )

    assert len(bus._streams["frames_raw"]) == CAPPED_MAXLEN["frames_raw"]  # noqa: SLF001


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
            }
        )
        websocket.close()

    entries = bus._streams.get("frames_raw", [])  # noqa: SLF001
    assert len(entries) == 1
    event = RawFrame.model_validate_json(entries[0].data)
    assert event.jpeg == jpeg_bytes
    assert event.width == 160
    assert event.height == 120
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
