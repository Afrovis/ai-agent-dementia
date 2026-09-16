"""Cloudflare Turnstile server-side verification (HANDOFF.md section 5.5).

`TurnstileVerifier` is the real client; tests use `StubVerifier` so the V5
API can be exercised without reaching Cloudflare (CLAUDE.md: no service may
require a live network to run its tests).
"""

from __future__ import annotations

from typing import Protocol

import httpx

VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


class TurnstileVerifierProtocol(Protocol):
    async def verify(self, token: str) -> bool: ...


class TurnstileVerifier:
    """Calls Cloudflare's siteverify endpoint. `remoteip` is never sent
    (HANDOFF.md 5.5: "Turnstile siteverify is called without remoteip")."""

    def __init__(self, secret_key: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._secret_key = secret_key
        self._client = client

    async def verify(self, token: str) -> bool:
        if not token:
            return False
        client = self._client or httpx.AsyncClient()
        try:
            response = await client.post(
                VERIFY_URL,
                data={"secret": self._secret_key, "response": token},
                timeout=5.0,
            )
            response.raise_for_status()
            return bool(response.json().get("success", False))
        except httpx.HTTPError:
            return False
        finally:
            if self._client is None:
                await client.aclose()


class StubVerifier:
    """Test double: verifies whatever `always_pass` says, regardless of token."""

    def __init__(self, always_pass: bool = True) -> None:
        self.always_pass = always_pass
        self.calls: list[str] = []

    async def verify(self, token: str) -> bool:
        self.calls.append(token)
        return self.always_pass
