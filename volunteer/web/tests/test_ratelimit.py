from __future__ import annotations

from volunteer_web.ratelimit import RateLimiter


def test_allows_up_to_the_limit_then_blocks() -> None:
    limiter = RateLimiter(max_per_window=2, window_seconds=3600)
    assert limiter.allow("1.2.3.4", now=0.0) is True
    assert limiter.allow("1.2.3.4", now=1.0) is True
    assert limiter.allow("1.2.3.4", now=2.0) is False


def test_separate_keys_are_independent() -> None:
    limiter = RateLimiter(max_per_window=1, window_seconds=3600)
    assert limiter.allow("1.2.3.4", now=0.0) is True
    assert limiter.allow("5.6.7.8", now=0.0) is True


def test_window_expires() -> None:
    limiter = RateLimiter(max_per_window=1, window_seconds=10)
    assert limiter.allow("1.2.3.4", now=0.0) is True
    assert limiter.allow("1.2.3.4", now=5.0) is False
    assert limiter.allow("1.2.3.4", now=11.0) is True
