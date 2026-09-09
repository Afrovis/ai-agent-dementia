"""Tests for the `embodiment` FastAPI app, using `FakeBus` (no Redis, no real TLS)."""

import asyncio

from fastapi.testclient import TestClient
from nc_shared.bus import FakeBus
from nc_shared.events import Say, Show

from embodiment.app import ConnectionManager, broadcast_loop, create_app


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
