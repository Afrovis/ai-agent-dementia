from __future__ import annotations

import pytest

from volunteer_web.turnstile import StubVerifier


@pytest.mark.asyncio
async def test_stub_verifier_records_calls_and_returns_configured_result() -> None:
    verifier = StubVerifier(always_pass=False)
    assert await verifier.verify("token") is False
    assert verifier.calls == ["token"]
