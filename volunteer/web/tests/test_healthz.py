from __future__ import annotations

from fastapi.testclient import TestClient

from volunteer_web.app import create_app
from volunteer_web.config import WebConfig


def test_healthz_ok() -> None:
    client = TestClient(create_app(WebConfig()))
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_security_headers_present() -> None:
    client = TestClient(create_app(WebConfig()))
    response = client.get("/healthz")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "challenges.cloudflare.com" in response.headers["content-security-policy"]
