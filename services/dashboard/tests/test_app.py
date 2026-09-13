"""Tests for the `dashboard` FastAPI app, using `FakeBus` and `TestClient`
(no Redis, no camera, no browser). Test JPEGs are generated in-process with
Pillow; none are committed as fixtures."""

from __future__ import annotations

import io
import json
import wave

import pytest
from fastapi.testclient import TestClient
from nc_shared.bus import FakeBus
from nc_shared.events import Frame
from PIL import Image

from dashboard.app import LatestFrame, consume_frame_once, create_app
from dashboard.zones_store import load_existing_zones


def _jpeg_bytes(color=(10, 20, 30), size=(16, 12)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _publish_frame(bus: FakeBus, jpeg: bytes, width=16, height=12) -> None:
    bus.publish(
        Frame(source="capture", jpeg=jpeg, width=width, height=height, source_kind="browser")
    )


VALID_ZONES = {
    "bed": [[0.0, 0.2], [0.4, 0.2], [0.4, 0.8], [0.0, 0.8]],
    "door": [[0.85, 0.0], [1.0, 0.0], [1.0, 1.0]],
    "bathroom_path": [],
}


# --- fail-closed with no password ------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/zones"),
        ("GET", "/zones/frame.jpg"),
        ("GET", "/zones/current"),
        ("GET", "/profile"),
        ("GET", "/strategies"),
        ("GET", "/media"),
        ("GET", "/static/style.css"),
        ("GET", "/static/script.js"),
        ("POST", "/zones"),
    ],
)
def test_every_route_503s_with_no_password_configured(method, path):
    bus = FakeBus()
    app = create_app(bus, password=None)
    with TestClient(app) as client:
        response = client.request(method, path, data={"zones_json": "{}"})

    assert response.status_code == 503
    assert "DASHBOARD_PASSWORD" in response.text


def test_every_route_503s_with_an_empty_string_password():
    bus = FakeBus()
    app = create_app(bus, password="")
    with TestClient(app) as client:
        response = client.get("/zones")

    assert response.status_code == 503
    assert "DASHBOARD_PASSWORD" in response.text


def test_503_holds_even_with_credentials_supplied():
    bus = FakeBus()
    app = create_app(bus, password=None)
    with TestClient(app) as client:
        response = client.get("/zones", auth=("anyone", "anything"))

    assert response.status_code == 503


# --- auth with a password configured ---------------------------------------


def test_request_without_credentials_is_rejected():
    bus = FakeBus()
    app = create_app(bus, password="secret123")
    with TestClient(app) as client:
        response = client.get("/zones")

    assert response.status_code == 401


def test_request_with_wrong_password_is_rejected():
    bus = FakeBus()
    app = create_app(bus, password="secret123")
    with TestClient(app) as client:
        response = client.get("/zones", auth=("caregiver", "wrong"))

    assert response.status_code == 401


def test_request_with_correct_password_succeeds():
    bus = FakeBus()
    app = create_app(bus, password="secret123")
    with TestClient(app) as client:
        response = client.get("/zones", auth=("caregiver", "secret123"))

    assert response.status_code == 200
    assert "Zones" in response.text


def test_any_username_is_accepted_with_the_right_password():
    bus = FakeBus()
    app = create_app(bus, password="secret123")
    with TestClient(app) as client:
        response = client.get("/zones", auth=("whoever", "secret123"))

    assert response.status_code == 200


# --- frame preview -----------------------------------------------------


def test_frame_jpg_503s_before_any_frame_has_arrived():
    bus = FakeBus()
    app = create_app(bus, password="secret123")
    with TestClient(app) as client:
        response = client.get("/zones/frame.jpg", auth=("c", "secret123"))

    assert response.status_code == 503
    assert "camera" in response.text.lower() or "frame" in response.text.lower()


def test_frame_jpg_serves_the_most_recent_frame_bytes():
    bus = FakeBus()
    jpeg = _jpeg_bytes()
    _publish_frame(bus, jpeg)
    app = create_app(bus, password="secret123")

    # Drive the single-iteration consumer directly rather than racing the
    # background task's timing, same reasoning as
    # embodiment.app.broadcast_loop's direct-call tests.
    consume_frame_once(bus, app.state.latest_frame)

    with TestClient(app) as client:
        response = client.get("/zones/frame.jpg", auth=("c", "secret123"))

    assert response.status_code == 200
    assert response.content == jpeg
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"


def test_frame_jpg_keeps_only_the_most_recent_frame():
    bus = FakeBus()
    first = _jpeg_bytes(color=(1, 1, 1))
    second = _jpeg_bytes(color=(2, 2, 2))
    _publish_frame(bus, first)
    _publish_frame(bus, second)
    app = create_app(bus, password="secret123")

    consume_frame_once(bus, app.state.latest_frame, count=10)

    with TestClient(app) as client:
        response = client.get("/zones/frame.jpg", auth=("c", "secret123"))

    assert response.content == second


def test_consume_frame_once_acks_every_message_read():
    bus = FakeBus()
    bus.ensure_group("frames", "dashboard")
    _publish_frame(bus, _jpeg_bytes())
    latest = LatestFrame()

    count = consume_frame_once(bus, latest)

    assert count == 1
    assert bus.pending("frames", "dashboard") == []


# --- saving zones --------------------------------------------------------


def test_post_zones_saves_a_valid_payload(tmp_path):
    bus = FakeBus()
    zones_path = tmp_path / "zones.yaml"
    app = create_app(bus, password="secret123", zones_path=zones_path)

    with TestClient(app) as client:
        response = client.post(
            "/zones",
            data={"zones_json": json.dumps(VALID_ZONES)},
            auth=("c", "secret123"),
        )

    assert response.status_code == 200
    assert "restart" in response.text.lower()
    assert zones_path.exists()

    saved = load_existing_zones(zones_path)
    assert saved["bed"] == VALID_ZONES["bed"]
    assert saved["door"] == VALID_ZONES["door"]

    # No leftover temp files in the target directory.
    assert list(tmp_path.iterdir()) == [zones_path]


def test_post_zones_rejects_too_few_points(tmp_path):
    bus = FakeBus()
    zones_path = tmp_path / "zones.yaml"
    app = create_app(bus, password="secret123", zones_path=zones_path)
    bad = {"bed": [[0.1, 0.1], [0.2, 0.2]], "door": [], "bathroom_path": []}

    with TestClient(app) as client:
        response = client.post(
            "/zones", data={"zones_json": json.dumps(bad)}, auth=("c", "secret123")
        )

    assert response.status_code == 400
    assert any("at least" in msg for msg in response.json()["detail"])
    assert not zones_path.exists()


def test_post_zones_rejects_coordinate_out_of_range(tmp_path):
    bus = FakeBus()
    zones_path = tmp_path / "zones.yaml"
    app = create_app(bus, password="secret123", zones_path=zones_path)
    bad = {"bed": [[0.1, 0.1], [1.2, 0.2], [0.2, 0.9]], "door": [], "bathroom_path": []}

    with TestClient(app) as client:
        response = client.post(
            "/zones", data={"zones_json": json.dumps(bad)}, auth=("c", "secret123")
        )

    assert response.status_code == 400
    assert any("out of range" in msg for msg in response.json()["detail"])
    assert not zones_path.exists()


def test_post_zones_rejects_non_numeric_coordinate(tmp_path):
    bus = FakeBus()
    zones_path = tmp_path / "zones.yaml"
    app = create_app(bus, password="secret123", zones_path=zones_path)
    bad = {"bed": [[0.1, 0.1], ["nope", 0.2], [0.2, 0.9]], "door": [], "bathroom_path": []}

    with TestClient(app) as client:
        response = client.post(
            "/zones", data={"zones_json": json.dumps(bad)}, auth=("c", "secret123")
        )

    assert response.status_code == 400
    assert any("numbers" in msg for msg in response.json()["detail"])
    assert not zones_path.exists()


def test_post_zones_rejects_malformed_json():
    bus = FakeBus()
    app = create_app(bus, password="secret123")

    with TestClient(app) as client:
        response = client.post("/zones", data={"zones_json": "{not json"}, auth=("c", "secret123"))

    assert response.status_code == 400


def test_post_zones_requires_auth_too():
    bus = FakeBus()
    app = create_app(bus, password="secret123")

    with TestClient(app) as client:
        response = client.post("/zones", data={"zones_json": json.dumps(VALID_ZONES)})

    assert response.status_code == 401


# --- existing zones for pre-populating the editor --------------------------


def test_zones_current_reflects_a_saved_file(tmp_path):
    bus = FakeBus()
    zones_path = tmp_path / "zones.yaml"
    app = create_app(bus, password="secret123", zones_path=zones_path)

    with TestClient(app) as client:
        client.post("/zones", data={"zones_json": json.dumps(VALID_ZONES)}, auth=("c", "secret123"))
        response = client.get("/zones/current", auth=("c", "secret123"))

    assert response.status_code == 200
    body = response.json()
    assert body["bed"] == VALID_ZONES["bed"]
    assert body["bathroom_path"] == []


def test_a_non_ascii_password_is_refused_loudly_at_startup():
    """HTTP Basic cannot carry a non-ASCII password, so refuse to start.

    FastAPI decodes the Authorization header as ASCII and answers 401 on
    anything else, before any of the dashboard's own code runs. A caregiver
    who picks a password with an accent in it would type the correct
    password and be rejected forever with no clue why, so the service
    refuses to come up at all and names the problem.
    """
    with pytest.raises(ValueError, match="ASCII"):
        create_app(FakeBus(), password="släpp mig in")


def test_an_ascii_password_still_authenticates_normally():
    client = TestClient(create_app(FakeBus(), password="correct horse"))

    assert client.get("/zones", auth=("caregiver", "correct horse")).status_code == 200
    assert client.get("/zones", auth=("caregiver", "wrong horse")).status_code == 401


def test_profile_page_saves_caregiver_fields(tmp_path):
    person_path = tmp_path / "person.yaml"
    app = create_app(FakeBus(), password="secret123", person_path=person_path)
    form = {
        "name": "Jean",
        "preferred_address": "Jeannie",
        "caregiver_name": "Tom",
        "caregiver_relationship": "son",
        "night_themes": "Looks for work",
        "calming_things": "Garden photo",
        "things_to_avoid": "Urgent language",
        "physical_notes": "Uses a walker",
        "restroom_location": "Outside the door",
    }

    with TestClient(app) as client:
        response = client.post("/profile", data=form, auth=("c", "secret123"))

    assert response.status_code == 200
    assert "Profile saved" in response.text
    assert "Jean" in person_path.read_text()


def test_strategy_page_renders_infinite_escalation_dwell_as_until_acknowledged(tmp_path):
    strategies_path = tmp_path / "strategies.yaml"
    strategies_path.write_text(
        """strategies:
  - id: escalate_phone
    order: 10
    enabled: true
    dwell_seconds: .inf
    cooldown_seconds: 0
    intrusiveness: 5
    face: speaking
    brightness: 0.7
    headline: Someone is coming to help
    body: ''
    say: Someone is coming to help.
"""
    )
    app = create_app(FakeBus(), password="secret123", strategies_path=strategies_path)

    with TestClient(app) as client:
        response = client.get("/strategies", auth=("c", "secret123"))

    assert response.status_code == 200
    assert "Until acknowledged" in response.text
    hidden = 'type="hidden" name="escalate_phone__dwell_seconds" value="inf"'
    assert hidden in response.text


def test_media_page_uploads_validated_photo_and_voice_clip(tmp_path):
    photo = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(photo, format="PNG")
    voice = io.BytesIO()
    with wave.open(voice, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 1600)
    app = create_app(
        FakeBus(),
        password="secret123",
        photo_dir=tmp_path / "photos",
        voice_clip_dir=tmp_path / "voice",
    )

    with TestClient(app) as client:
        photo_response = client.post(
            "/media/photo",
            files={"upload": ("Family.png", photo.getvalue(), "image/png")},
            auth=("c", "secret123"),
        )
        voice_response = client.post(
            "/media/voice",
            files={"upload": ("Tom.wav", voice.getvalue(), "audio/wav")},
            auth=("c", "secret123"),
        )

    assert photo_response.status_code == voice_response.status_code == 200
    assert (tmp_path / "photos" / "family.png").exists()
    assert (tmp_path / "voice" / "tom.wav").exists()


def test_media_upload_rejects_spoofed_file_content(tmp_path):
    app = create_app(FakeBus(), password="secret123", photo_dir=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/media/photo",
            files={"upload": ("looks-safe.png", b"not an image", "image/png")},
            auth=("c", "secret123"),
        )

    assert response.status_code == 400
    assert not list(tmp_path.iterdir())


def test_new_settings_routes_require_authentication(tmp_path):
    app = create_app(
        FakeBus(),
        password="secret123",
        person_path=tmp_path / "person.yaml",
        strategies_path=tmp_path / "strategies.yaml",
        photo_dir=tmp_path / "photos",
        voice_clip_dir=tmp_path / "voice",
    )

    with TestClient(app) as client:
        assert client.post("/profile", data={"name": "Jean"}).status_code == 401
        assert client.post("/strategies", data={}).status_code == 401
        photo = {"upload": ("photo.png", b"bad", "image/png")}
        voice = {"upload": ("voice.wav", b"bad", "audio/wav")}
        assert client.post("/media/photo", files=photo).status_code == 401
        assert client.post("/media/voice", files=voice).status_code == 401
