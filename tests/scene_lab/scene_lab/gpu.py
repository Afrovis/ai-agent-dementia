"""Is the host GPU free? Checked before triage work that runs on the Mac mini.

Triage quick tests (pytest, replays) run on the same 16 GB host as Ollama.
Measured timings only mean something on a quiet machine, so each triage
worker waits here until the GPU is idle, one worker at a time. This module
only reads: it never unloads a model, stops a stack or signals Ollama
(AGENTS.md, "Operational gotchas").
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from urllib.request import urlopen

OLLAMA_URL = os.environ.get("SCENE_LAB_OLLAMA_URL", "http://127.0.0.1:11434")
IDLE_UTILIZATION_PCT = 20
WAIT_S = 300
POLL_S = 15
SAMPLES = 3

_UTILIZATION = re.compile(r'"Device Utilization %"=(\d+)')


def device_utilization() -> int | None:
    """Highest `Device Utilization %` over the host's GPUs; None off macOS."""
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = [int(v) for v in _UTILIZATION.findall(out)]
    return max(values) if values else None


def ollama_models() -> list[dict] | None:
    """Models host Ollama holds in memory; None when it does not answer."""
    try:
        with urlopen(f"{OLLAMA_URL}/api/ps", timeout=3) as response:  # noqa: S310
            models = json.load(response).get("models", [])
    except (OSError, ValueError):
        return None
    return [{"name": m.get("name"), "size_vram": m.get("size_vram")} for m in models]


def snapshot(samples: int = SAMPLES, sleep=time.sleep) -> dict:
    """GPU utilisation (the highest of a few samples a second apart) and Ollama's models."""
    readings = []
    for i in range(samples):
        if i:
            sleep(1)
        readings.append(device_utilization())
    known = [r for r in readings if r is not None]
    return {
        "device_utilization_pct": max(known) if known else None,
        "ollama_models": ollama_models(),
    }


def wait_idle(
    timeout_s: float = WAIT_S,
    poll_s: float = POLL_S,
    *,
    snap=snapshot,
    sleep=time.sleep,
    clock=time.monotonic,
) -> dict:
    """Wait until GPU utilisation is at most IDLE_UTILIZATION_PCT, up to `timeout_s`.

    Returns the last snapshot with `idle`, `waited_s` and, when still busy,
    `busy`. The caller goes ahead either way and marks the work as contended.
    """
    started = clock()
    while True:
        state = snap()
        util = state.get("device_utilization_pct")
        waited = round(clock() - started, 1)
        if util is None or util <= IDLE_UTILIZATION_PCT:
            return {**state, "idle": True, "waited_s": waited}
        if waited >= timeout_s:
            return {
                **state,
                "idle": False,
                "waited_s": waited,
                "busy": f"GPU at {util}% after waiting {waited:g} s",
            }
        sleep(poll_s)
