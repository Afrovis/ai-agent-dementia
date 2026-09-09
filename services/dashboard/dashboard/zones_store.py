"""Reading, validating, and writing the caregiver-drawn zones config
(issue #10).

The schema and path-resolution order here are a deliberate mirror of
`perceive.zones.load_zones`: `bed`, `door`, `bathroom_path`, each a list of
`[x, y]` points normalised to 0.0-1.0 (origin top-left), resolved from an
explicit path, then `ZONES_PATH`, then `config/zones.yaml`, falling back to
`config/zones.example.yaml` for *reading* (never for writing) when the
primary file does not exist yet.

`dashboard` does not import `perceive`'s code to do this -- "nothing calls
another service directly" (HANDOFF.md section 1) applies to imports as much
as to network calls, and the two services are built, deployed, and tested
independently. Instead this module reimplements the small amount of yaml
handling `perceive/zones.py` needs, and
`services/dashboard/tests/test_zones_store.py` imports the real
`perceive.zones.load_zones` to prove a file this module writes is read back
unchanged -- that test is the thing keeping the two in sync, not code
sharing.

`perceive` loads `zones.yaml` once at startup (`perceive.main.run`) and
never rereads it, so a successful save here does not take effect until
`perceive` is restarted; `dashboard.app` tells the caregiver that after
every save.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

Point = list[float]
Polygon = list[Point]

ZONE_NAMES: tuple[str, ...] = ("bed", "door", "bathroom_path")

DEFAULT_ZONES_DIR = Path("config")
DEFAULT_ZONES_FILENAME = "zones.yaml"
DEFAULT_ZONES_EXAMPLE_FILENAME = "zones.example.yaml"

# A caregiver drawing a zone by hand with a mouse or finger will never need
# more than a few dozen points; this just keeps a malformed or hostile
# payload from writing an absurdly large config file.
MAX_POINTS = 200
MIN_POINTS = 3

# Keeps `zones.yaml` short and diffable across edits, while still being far
# more precise than a caregiver's mouse or touch input.
DECIMAL_PLACES = 4


def _primary_path(path: str | Path | None, env: dict[str, str] | None) -> Path:
    """Resolve the primary `zones.yaml` path: explicit `path`, then
    `ZONES_PATH`, then `config/zones.yaml` -- the same order
    `perceive.zones.load_zones` uses, minus its final fallback to
    `zones.example.yaml`, which only applies when reading."""
    env = os.environ if env is None else env
    if path is not None:
        return Path(path)
    return Path(env.get("ZONES_PATH", DEFAULT_ZONES_DIR / DEFAULT_ZONES_FILENAME))


def load_existing_zones(
    path: str | Path | None = None, *, env: dict[str, str] | None = None
) -> dict[str, Polygon]:
    """Load the zones currently in effect, for the editor to pre-populate.

    Falls back to `zones.example.yaml` next to the primary path if it does
    not exist yet, same as `perceive`, so the editor shows the same zones
    `perceive` is actually using. Never raises: a missing, unreadable, or
    malformed file (or zone) degrades to an empty polygon for that zone, so
    a caregiver can always open the editor and start drawing.
    """
    primary = _primary_path(path, env)
    candidate = primary if primary.exists() else primary.parent / DEFAULT_ZONES_EXAMPLE_FILENAME

    empty: dict[str, Polygon] = {name: [] for name in ZONE_NAMES}
    if not candidate.exists():
        return empty

    try:
        with candidate.open() as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return empty

    if not isinstance(raw, dict):
        return empty

    result: dict[str, Polygon] = {}
    for name in ZONE_NAMES:
        points = raw.get(name) or []
        try:
            result[name] = [[float(p[0]), float(p[1])] for p in points]
        except (TypeError, ValueError, IndexError):
            result[name] = []
    return result


def validate_zones(payload: Any) -> tuple[dict[str, Polygon] | None, list[str]]:
    """Validate a caregiver-submitted zones payload server-side.

    Never trusts the browser: checks the payload is a mapping of the three
    known zone names to lists of `[x, y]` points, each zone either empty
    (legal -- "this zone is not configured", per `perceive.zones`) or at
    least `MIN_POINTS` and at most `MAX_POINTS` points, each coordinate a
    finite number within 0.0-1.0. Returns `(zones, [])` on success, where
    coordinates are rounded to `DECIMAL_PLACES`, or `(None, errors)` with
    one human-readable message per problem found -- collecting every
    problem rather than stopping at the first, so a caregiver fixing a
    rejected drawing does not have to resubmit once per mistake.
    """
    errors: list[str] = []

    if not isinstance(payload, dict):
        return None, ["zones payload must be a JSON object"]

    unknown = sorted(set(payload) - set(ZONE_NAMES))
    if unknown:
        errors.append(f"unknown zone name(s): {', '.join(unknown)}")

    result: dict[str, Polygon] = {}
    for name in ZONE_NAMES:
        raw_points = payload.get(name) or []
        if not isinstance(raw_points, list):
            errors.append(f"{name}: must be a list of points")
            continue
        if len(raw_points) == 0:
            result[name] = []  # legal: zone not configured
            continue
        if len(raw_points) < MIN_POINTS:
            errors.append(f"{name}: needs at least {MIN_POINTS} points, got {len(raw_points)}")
            continue
        if len(raw_points) > MAX_POINTS:
            errors.append(f"{name}: at most {MAX_POINTS} points allowed, got {len(raw_points)}")
            continue

        parsed: Polygon = []
        zone_ok = True
        for index, point in enumerate(raw_points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                errors.append(f"{name}: point {index} must be an [x, y] pair")
                zone_ok = False
                continue
            x_raw, y_raw = point
            try:
                x = float(x_raw)
                y = float(y_raw)
            except (TypeError, ValueError):
                errors.append(f"{name}: point {index} coordinates must be numbers")
                zone_ok = False
                continue
            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                errors.append(f"{name}: point {index} out of range 0.0-1.0")
                zone_ok = False
                continue
            parsed.append([round(x, DECIMAL_PLACES), round(y, DECIMAL_PLACES)])
        if zone_ok:
            result[name] = parsed

    if errors:
        return None, errors
    return result, []


def save_zones(
    zones: dict[str, Polygon], path: str | Path | None = None, *, env: dict[str, str] | None = None
) -> Path:
    """Write `zones` to the primary zones path, atomically.

    Writes to a temp file in the same directory first, then `os.replace`s
    it into place, so a crash or a concurrent read never sees a
    half-written file -- `perceive` reads this file at startup, and a torn
    write it happens to catch would be exactly the silent failure
    HANDOFF.md rule 4 forbids. The temp file is always cleaned up, on
    success (it no longer exists once renamed) or on failure (removed in
    the `except` below).
    """
    target = _primary_path(path, env)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{DEFAULT_ZONES_FILENAME}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(zones, handle, default_flow_style=None, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target
