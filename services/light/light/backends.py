"""Hardware adapters for the hallway path light.

The first supported device is a Shelly Gen2+ switch using its LAN-only RPC
endpoint. The interface is intentionally tiny so another local smart plug can
be added without changing the event or consume loop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("light.backends")


class LightBackend(Protocol):
    def set_state(self, on: bool) -> bool:
        """Set an absolute state and return whether the device accepted it."""
        ...


class DisabledBackend:
    """Feature-flag-off backend used by default and in hardware-free tests."""

    def __init__(self) -> None:
        self.states: list[bool] = []

    def set_state(self, on: bool) -> bool:
        self.states.append(on)
        logger.info("path light disabled; requested state=%s", "on" if on else "off")
        return True


class UnavailableBackend:
    """Permanent failure backend used to keep invalid enabled config unhealthy."""

    def set_state(self, on: bool) -> bool:
        return False


class ShellyBackend:
    """Shelly Gen2+ `Switch.Set` adapter over the local network."""

    def __init__(
        self,
        device_url: str,
        *,
        switch_id: int = 0,
        timeout: float = 3.0,
        open_fn: Callable[..., object] | None = None,
    ) -> None:
        self.url = f"{device_url.rstrip('/')}/rpc/Switch.Set"
        self.switch_id = switch_id
        self.timeout = timeout
        self._open_fn = open_fn or urlopen

    def set_state(self, on: bool) -> bool:
        try:
            request = Request(
                self.url,
                data=json.dumps({"id": self.switch_id, "on": on}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = self._open_fn(request, timeout=self.timeout)
        except (HTTPError, URLError, TimeoutError, OSError):
            logger.exception("smart-plug request failed")
            return False
        status = getattr(response, "status", 0)
        close = getattr(response, "close", None)
        if callable(close):
            close()
        if 200 <= status < 300:
            return True
        logger.warning("smart plug responded with status %s", status)
        return False
