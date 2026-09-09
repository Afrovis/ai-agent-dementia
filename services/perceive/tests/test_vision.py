"""Tests for `perceive.vision`: sanitisation, the fake client, and
`OllamaVisionClient`'s fail-safe behaviour with a stubbed `httpx`.

No real Ollama, no network: `OllamaVisionClient` is exercised by
monkeypatching `httpx.post` with hand-built stand-ins, per HANDOFF.md
section 4.
"""

import base64

from perceive.vision import (
    MAX_SCENE_NOTE_LENGTH,
    FakeVisionClient,
    OllamaVisionClient,
    _sanitize_note,
)


class _FakeResponse:
    def __init__(self, status_code: int = 200, json_body: object = None, raise_on_json=None):
        self.status_code = status_code
        self._json_body = json_body
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._json_body


def test_sanitize_note_collapses_whitespace_and_strips_quotes():
    assert _sanitize_note('  "He  is\nsitting  up."  ') == "He is sitting up."


def test_sanitize_note_keeps_only_the_first_sentence():
    assert _sanitize_note("He is pacing. He looks distressed.") == "He is pacing."


def test_sanitize_note_truncates_to_the_documented_maximum():
    long_text = "a" * (MAX_SCENE_NOTE_LENGTH + 50)
    note = _sanitize_note(long_text)
    assert note is not None
    assert len(note) <= MAX_SCENE_NOTE_LENGTH


def test_sanitize_note_returns_none_for_empty_input():
    assert _sanitize_note("   ") is None
    assert _sanitize_note('""') is None


def test_fake_vision_client_returns_canned_responses_in_order_and_records_calls():
    client = FakeVisionClient(["first note.", None, "third note."])

    assert client.describe(b"jpeg-1") == "first note."
    assert client.describe(b"jpeg-2") is None
    assert client.describe(b"jpeg-3") == "third note."
    assert client.describe(b"jpeg-4") is None  # exhausted
    assert client.calls == [b"jpeg-1", b"jpeg-2", b"jpeg-3", b"jpeg-4"]


def test_ollama_vision_client_returns_sanitised_note_on_success(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200, {"response": "She is standing near the door."})

    monkeypatch.setattr("perceive.vision.httpx.post", fake_post)

    client = OllamaVisionClient(
        ollama_url="http://host.docker.internal:11434", model="moondream", timeout_seconds=5.0
    )
    note = client.describe(b"\xff\xd8\xff")

    assert note == "She is standing near the door."
    assert captured["url"] == "http://host.docker.internal:11434/api/generate"
    assert captured["json"]["model"] == "moondream"
    assert captured["json"]["stream"] is False
    assert captured["json"]["images"] == [base64.b64encode(b"\xff\xd8\xff").decode("ascii")]
    assert captured["timeout"] == 5.0


def test_ollama_vision_client_returns_none_on_transport_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise ConnectionError("no route to host")

    monkeypatch.setattr("perceive.vision.httpx.post", fake_post)

    client = OllamaVisionClient(ollama_url="http://x:11434", model="moondream")
    assert client.describe(b"jpeg") is None


def test_ollama_vision_client_returns_none_on_non_200(monkeypatch):
    monkeypatch.setattr("perceive.vision.httpx.post", lambda url, json, timeout: _FakeResponse(500))

    client = OllamaVisionClient(ollama_url="http://x:11434", model="moondream")
    assert client.describe(b"jpeg") is None


def test_ollama_vision_client_returns_none_on_empty_response_field(monkeypatch):
    monkeypatch.setattr(
        "perceive.vision.httpx.post",
        lambda url, json, timeout: _FakeResponse(200, {"response": ""}),
    )

    client = OllamaVisionClient(ollama_url="http://x:11434", model="moondream")
    assert client.describe(b"jpeg") is None


def test_ollama_vision_client_returns_none_on_unparseable_body(monkeypatch):
    monkeypatch.setattr(
        "perceive.vision.httpx.post",
        lambda url, json, timeout: _FakeResponse(200, raise_on_json=ValueError("bad json")),
    )

    client = OllamaVisionClient(ollama_url="http://x:11434", model="moondream")
    assert client.describe(b"jpeg") is None


def test_ollama_vision_client_never_raises_on_unexpected_exception(monkeypatch):
    def fake_post(url, json, timeout):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr("perceive.vision.httpx.post", fake_post)

    client = OllamaVisionClient(ollama_url="http://x:11434", model="moondream")
    # Must not raise.
    assert client.describe(b"jpeg") is None
