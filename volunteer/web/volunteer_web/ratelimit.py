"""In-memory per-IP rate limiting (HANDOFF.md section 5.5: 5/hour, in memory only)."""

from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_per_window: int, window_seconds: float = 3600.0) -> None:
        self._max = max_per_window
        self._window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        cutoff = now - self._window
        hits = [t for t in self._hits[key] if t > cutoff]
        allowed = len(hits) < self._max
        if allowed:
            hits.append(now)
        self._hits[key] = hits
        return allowed
