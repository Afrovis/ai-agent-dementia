"""Pluggable notification backends for the `notify` service.

HANDOFF.md section 2: "Notifications: ntfy by default. Backend interface
allows Pushover and Telegram later." PLAN.md section 9 lists the same.  A
backend only needs to know how to deliver one notification; it is
deliberately decoupled from `nc_shared.events.Notify` and the bus so a new
backend (Pushover, Telegram, ...) can be added without touching the event
schema or the consume loop in `notify.main`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import httpx

logger = logging.getLogger("notify.backends")

# ntfy priority names, see https://docs.ntfy.sh/publish/#message-priority.
# Mapped from the three `Notify.level` values in nc_shared.events.
NTFY_PRIORITY_BY_LEVEL: dict[str, str] = {
    "info": "low",
    "attention": "default",
    "critical": "urgent",
}


class NotifyBackend(Protocol):
    """Interface every notification backend implements.

    Fields mirror `nc_shared.events.Notify` (`level`, `title`, `body`)
    rather than the event class itself, so backends stay free of any
    bus/event dependency.
    """

    def send(self, level: str, title: str, body: str) -> bool:
        """Attempt to deliver one notification. Return True on success."""
        ...


class NtfyBackend:
    """Default backend. POSTs to an ntfy topic URL (self-hosted or ntfy.sh).

    The HTTP call goes through `post_fn` so tests can inject a stub and
    never touch the network; the default `post_fn` opens a short-lived
    `httpx.Client` per call, which is fine at notify's low request rate.
    """

    def __init__(
        self,
        url: str,
        post_fn: Callable[..., httpx.Response] | None = None,
        timeout: float = 10.0,
    ) -> None:
        """`url` is the full ntfy topic URL, e.g. `https://ntfy.sh/my-topic`."""
        self.url = url
        self.timeout = timeout
        self._post_fn = post_fn or self._default_post

    def _default_post(
        self, url: str, *, content: bytes, headers: dict[str, str], timeout: float
    ) -> httpx.Response:
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, content=content, headers=headers)

    def send(self, level: str, title: str, body: str) -> bool:
        """POST `body` to the ntfy topic with a `Title` header and mapped `Priority`."""
        headers = {
            "Title": title,
            "Priority": NTFY_PRIORITY_BY_LEVEL.get(level, "default"),
        }
        try:
            response = self._post_fn(
                self.url, content=body.encode("utf-8"), headers=headers, timeout=self.timeout
            )
        except httpx.HTTPError:
            logger.exception("ntfy request failed: url=%s", self.url)
            return False
        ok = 200 <= response.status_code < 300
        if not ok:
            logger.warning("ntfy responded with status %s", response.status_code)
        return ok


class LoggingBackend:
    """Fallback backend for local dev and tests: logs instead of calling out.

    Useful when `NTFY_URL` is not configured, or in tests that want a real
    `NotifyBackend` without any HTTP mocking. `sent` records every call for
    assertions.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, level: str, title: str, body: str) -> bool:
        logger.info("notification (%s): %s - %s", level, title, body)
        self.sent.append((level, title, body))
        return True


class DryRunBackend:
    """A deliberate no-delivery backend for supervised volunteer trials.

    Unlike ``LoggingBackend``, this backend does not write notification titles
    or bodies to stdout. The original ``Notify`` event is still retained by the
    store and visible in the authenticated dashboard timeline, which is the
    review surface for issue #27.
    """

    def __init__(self) -> None:
        self.suppressed_count = 0

    def send(self, level: str, title: str, body: str) -> bool:
        logger.info("dry run suppressed outbound notification: level=%s", level)
        self.suppressed_count += 1
        return True
