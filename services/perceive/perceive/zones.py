"""Caregiver-drawn zones for `perceive` (issue #8).

PLAN.md section 6.2: "the bed zone, door zone, and bathroom-direction zone
are drawn once in the dashboard on a daylight frame." The dashboard's Zones
page does not exist yet (it is issue #22+), so for now the caregiver (or a
developer standing in) edits `config/zones.yaml` by hand, in the same
env-then-yaml-then-defaults precedence as every other config in this repo
(HANDOFF.md section 4). `config/zones.example.yaml` documents the schema
and ships checked-in defaults so the service has something sane to fall
back to.

Polygons are stored and matched in *normalised* frame coordinates (0.0-1.0,
origin top-left), not pixels, so a zone drawn once survives a camera
resolution change -- the same reasoning `capture` and `classify` use for
pose landmarks.

Point-in-polygon is a plain-Python ray cast (`_point_in_polygon`): no
shapely, no numpy, per the issue brief, and cheap enough to run once per
frame on CPU.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

SERVICE_NAME = "perceive"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

Point = tuple[float, float]
Polygon = list[Point]

ZoneName = Literal["bed", "door", "bathroom_path", "other"]

DEFAULT_ZONES_DIR = Path("config")
DEFAULT_ZONES_FILENAME = "zones.yaml"
DEFAULT_ZONES_EXAMPLE_FILENAME = "zones.example.yaml"

# Fixed precedence when a point falls inside more than one polygon, most
# safety-relevant first. `bed` wins over `door` because `classify.in_bed`
# is the state HANDOFF.md rule 5 and PLAN.md 6.2 lean on hardest -- a
# blanket defeats pose models, so "in the bed zone" has to be trustworthy
# even when a caregiver's hand-drawn bed and door polygons overlap at the
# edges of a small room. `door` beats `bathroom_path` next because heading
# out of the room is the more specific, more urgent signal of the two; a
# generic path zone is the broadest catch-all and comes last, before
# "other". This order is a safety fallback for imperfectly drawn zones,
# not something callers should rely on as a substitute for drawing them
# not to overlap where it matters.
_PRECEDENCE: tuple[Literal["bed", "door", "bathroom_path"], ...] = (
    "bed",
    "door",
    "bathroom_path",
)


def _point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """Ray-casting point-in-polygon test, pure Python, no dependencies.

    Casts a ray from `(x, y)` in the +x direction and counts how many
    polygon edges it crosses; odd means inside. Standard and boring on
    purpose -- this runs once per frame per configured zone, so simplicity
    and no numpy/shapely dependency matter more than speed here.
    """
    if len(polygon) < 3:
        return False
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if (y1 > y) != (y2 > y):
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


@dataclass(frozen=True)
class ZoneMap:
    """The caregiver-drawn zones, ready for per-frame lookup.

    `polygons` only ever holds `"bed"`, `"door"`, `"bathroom_path"` keys
    that were actually configured with at least one point; an empty
    `ZoneMap` (no keys at all) is the documented degrade-to-"other"
    fallback, not an error state.
    """

    polygons: dict[str, Polygon] = field(default_factory=dict)

    def zone_for_point(self, x: float, y: float) -> ZoneName:
        """Return which zone `(x, y)` (normalised, 0.0-1.0) falls in.

        Applies the fixed precedence documented on `_PRECEDENCE` when
        polygons overlap. Returns `"other"` if the point is in none of
        them, or if no zones are configured at all.
        """
        for name in _PRECEDENCE:
            polygon = self.polygons.get(name)
            if polygon and _point_in_polygon(x, y, polygon):
                return name
        return "other"


def _log_fallback(reason: str, path: str) -> None:
    logger.warning(
        json.dumps(
            {
                "service": SERVICE_NAME,
                "message": "zones config unavailable, all points classify as 'other'",
                "reason": reason,
                "path": path,
            }
        )
    )


def load_zones(path: str | Path | None = None, *, env: dict[str, str] | None = None) -> ZoneMap:
    """Load a `ZoneMap` from `config/zones.yaml`, falling back sensibly.

    Resolution order: explicit `path` argument, then `ZONES_PATH` from
    `env` (defaults to `os.environ`), then `config/zones.yaml`. If that
    file does not exist, falls back to `zones.example.yaml` in the same
    directory. If neither exists, or the file cannot be parsed, or it
    parses to nothing usable, returns an empty `ZoneMap` (everything
    classifies as `"other"`) and logs a warning rather than raising --
    a zone-less system still has to report person states (HANDOFF.md rule
    4: fail loud to the caregiver via the log, fail quiet -- i.e. keep
    running -- for the person being watched).
    """
    env = os.environ if env is None else env
    primary = (
        Path(path)
        if path is not None
        else Path(env.get("ZONES_PATH", DEFAULT_ZONES_DIR / DEFAULT_ZONES_FILENAME))
    )
    candidate = primary if primary.exists() else primary.parent / DEFAULT_ZONES_EXAMPLE_FILENAME

    if not candidate.exists():
        _log_fallback("no zones.yaml or zones.example.yaml found", str(candidate))
        return ZoneMap()

    try:
        with candidate.open() as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        _log_fallback(f"failed to parse {candidate}: {exc}", str(candidate))
        return ZoneMap()

    if not isinstance(raw, dict):
        _log_fallback(
            f"{candidate} did not contain a mapping of zone names to polygons", str(candidate)
        )
        return ZoneMap()

    polygons: dict[str, Polygon] = {}
    for name in _PRECEDENCE:
        points = raw.get(name)
        if not points:
            continue
        try:
            polygons[name] = [(float(p[0]), float(p[1])) for p in points]
        except (TypeError, ValueError, IndexError):
            _log_fallback(f"malformed polygon for zone {name!r} in {candidate}", str(candidate))

    if not polygons:
        _log_fallback(f"{candidate} defined no usable zones", str(candidate))

    return ZoneMap(polygons=polygons)
