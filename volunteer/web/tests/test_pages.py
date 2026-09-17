from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from volunteer_web.app import create_app
from volunteer_web.config import WebConfig
from volunteer_web.turnstile import StubVerifier


def make_client(tmp_path: Path) -> TestClient:
    app = create_app(
        WebConfig(data_dir=str(tmp_path)), turnstile_verifier=StubVerifier(always_pass=True)
    )
    return TestClient(app)


def test_landing_page_serves(tmp_path: Path) -> None:
    response = make_client(tmp_path).get("/")
    assert response.status_code == 200
    assert "Record a video" in response.text


def test_record_page_serves(tmp_path: Path) -> None:
    response = make_client(tmp_path).get("/record")
    assert response.status_code == 200
    assert "consent-section" in response.text


def test_status_page_serves_for_any_id(tmp_path: Path) -> None:
    response = make_client(tmp_path).get("/s/some-unguessable-id-123456")
    assert response.status_code == 200
    assert "delete-button" in response.text


def test_script_json_is_valid_and_matches_known_states(tmp_path: Path) -> None:
    import json

    response = make_client(tmp_path).get("/script.json")
    assert response.status_code == 200
    steps = response.json()
    known_states = {"in_bed", "sitting_up", "standing", "walking", "on_floor", "absent", "upright"}
    step_ids = [step["step_id"] for step in steps]
    assert len(step_ids) == len(set(step_ids))
    for step in steps:
        assert step["expected_state"] in known_states
        assert isinstance(step["seconds"], int)
    assert json.dumps(steps)  # round-trips cleanly


def test_sample_and_prompts_mounts_404_gracefully_when_empty(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/sample/sample.mp4").status_code == 404
    assert client.get("/prompts/stand_still.wav").status_code == 404
