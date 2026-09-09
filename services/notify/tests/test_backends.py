"""Tests for `notify.backends`, using an injected post function (no real network)."""

import httpx

from notify.backends import LoggingBackend, NtfyBackend


class _FakePost:
    """Records calls and returns a canned `httpx.Response`."""

    def __init__(self, status_code: int = 200) -> None:
        self.calls: list[dict] = []
        self.status_code = status_code

    def __call__(self, url, *, content, headers, timeout):
        self.calls.append({"url": url, "content": content, "headers": headers, "timeout": timeout})
        return httpx.Response(status_code=self.status_code)


def test_ntfy_backend_posts_to_topic_url_with_headers():
    post = _FakePost()
    backend = NtfyBackend("https://ntfy.sh/my-topic", post_fn=post)

    ok = backend.send("attention", "Session escalated", "Anna is up.")

    assert ok is True
    assert len(post.calls) == 1
    call = post.calls[0]
    assert call["url"] == "https://ntfy.sh/my-topic"
    assert call["content"] == b"Anna is up."
    assert call["headers"]["Title"] == "Session escalated"
    assert call["headers"]["Priority"] == "default"


def test_ntfy_backend_priority_mapping_per_level():
    post = _FakePost()
    backend = NtfyBackend("https://ntfy.sh/my-topic", post_fn=post)

    backend.send("info", "t", "b")
    backend.send("attention", "t", "b")
    backend.send("critical", "t", "b")

    priorities = [call["headers"]["Priority"] for call in post.calls]
    assert priorities == ["low", "default", "urgent"]


def test_ntfy_backend_returns_false_on_error_status():
    post = _FakePost(status_code=500)
    backend = NtfyBackend("https://ntfy.sh/my-topic", post_fn=post)

    assert backend.send("critical", "t", "b") is False


def test_ntfy_backend_returns_false_on_http_error():
    def raising_post(*args, **kwargs):
        raise httpx.ConnectError("boom")

    backend = NtfyBackend("https://ntfy.sh/my-topic", post_fn=raising_post)

    assert backend.send("critical", "t", "b") is False


def test_logging_backend_records_sends_without_network():
    backend = LoggingBackend()

    ok = backend.send("info", "title", "body")

    assert ok is True
    assert backend.sent == [("info", "title", "body")]
