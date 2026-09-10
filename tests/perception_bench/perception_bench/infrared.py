"""Tier 3: real infrared clips from the actual room. The real bench.

Currently empty: no infrared recordings of consenting volunteers exist yet
(HANDOFF.md section 8; this issue's brief). This module defines the
fixture format whoever records them will follow -- see `RECORDING.md` next
to this file for the recording protocol itself -- and loads a manifest of
clips if one exists. When none does, `run_tier3` reports that plainly and
returns a result with `accuracy=None` and `latency=None`, never raising,
never fabricating a number.

Manifest format
----------------

A YAML file (default `tests/perception_bench/fixtures/ir_manifest.yaml`,
override with `--ir-manifest`) with this shape:

```yaml
zones:
  bed: [[0.0, 0.2], [0.35, 0.2], [0.35, 0.85], [0.0, 0.85]]
  door: [[0.85, 0.0], [1.0, 0.0], [1.0, 1.0], [0.85, 1.0]]
  bathroom_path: [[0.35, 0.85], [0.85, 0.85], [0.85, 1.0], [0.35, 1.0]]

clips:
  - path: clips/2026-01-10_sit_up_to_walk.mp4   # relative to the manifest's own directory
    frame_interval_s: 0.5                        # 1 / recording fps
    timeline:
      - {from_s: 0.0, to_s: 8.0, state: in_bed}
      - {from_s: 8.0, to_s: 11.0, state: sitting_up}
      - {from_s: 11.0, to_s: 14.0, state: standing}
      - {from_s: 14.0, to_s: 22.0, state: walking}
      - {from_s: 22.0, to_s: 30.0, state: absent}      # left the frame via the door
```

Fields:

- `zones`: the same polygon format as `config/zones.yaml` (see
  `perceive.zones`), for *this* clip's room and camera placement. A
  manifest may define zones once at the top level (used by every clip that
  does not override it) and/or per clip under a `zones` key nested inside
  that clip's mapping, for a manifest spanning more than one camera
  position.
- `clips[].path`: relative to the manifest file's own directory, so the
  manifest and its clips can be moved together without editing paths.
  Never a path under `data/` committed to git -- the clips themselves stay
  local (HANDOFF.md rule 2, this issue's hard constraints).
- `clips[].frame_interval_s`: seconds between frames as actually recorded
  or sampled for scoring. Needed to convert the per-frame latency this
  tier can measure (like tier 1) into seconds.
- `clips[].timeline`: a per-interval expected-state timeline, `from_s`
  inclusive to `to_s` exclusive, in the clip's own seconds from zero.
  Intervals should be contiguous and covers the whole clip; a gap is
  treated as "not scored" for those seconds rather than an error. This is
  deliberately coarser than true per-frame annotation (labelling every
  frame of a 20-minute recording by hand is not a reasonable ask) but
  still carries a known transition *time* per boundary, which is what lets
  this tier measure latency, unlike tier 2.

This module reads the manifest and geometry; it does not decode video.
Real video decoding (frame extraction at `frame_interval_s`) needs OpenCV
or similar and is deliberately not implemented until a real manifest and
real clips exist to test it against -- there is nothing to verify that
code against today, and shipping unverified video-decoding code would be
worse than being honest that this tier is a loader plus a documented
format, waiting for data. See `RECORDING.md` for how to produce that data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from perceive.classify import PersonStateName
from perceive.zones import Polygon, ZoneMap

DEFAULT_MANIFEST_PATH = Path("tests/perception_bench/fixtures/ir_manifest.yaml")


@dataclass(frozen=True)
class TimelineInterval:
    """One `[from_s, to_s)` interval of a clip's expected-state timeline."""

    from_s: float
    to_s: float
    state: PersonStateName


@dataclass(frozen=True)
class IrClip:
    """One manifest-listed infrared clip: where to find it, its zones, its
    frame rate, and its expected-state timeline."""

    path: Path
    zones: ZoneMap
    frame_interval_s: float
    timeline: list[TimelineInterval]


def _parse_zones(raw: dict[str, Any]) -> ZoneMap:
    polygons: dict[str, Polygon] = {}
    for name in ("bed", "door", "bathroom_path"):
        points = raw.get(name)
        if points:
            polygons[name] = [(float(p[0]), float(p[1])) for p in points]
    return ZoneMap(polygons=polygons)


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> list[IrClip]:
    """Load `manifest_path` into a list of `IrClip`s. Returns an empty list
    -- not an error -- if the file does not exist: an absent tier-3
    manifest is the expected, documented state of this repository today
    (see the module docstring), not a bench failure.

    Raises `ValueError` for a manifest that exists but is malformed (a
    typo'd state name, a clip missing a required field): unlike "no
    manifest at all", a broken manifest is something the person who wrote
    it should hear about loudly, per HANDOFF.md rule 4's "fail loud to the
    caregiver" -- the equivalent audience here is whoever is building this
    tier.
    """
    if not manifest_path.exists():
        return []

    with manifest_path.open() as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path} must contain a mapping with a 'clips' key")

    default_zones = _parse_zones(raw.get("zones") or {})
    clips_raw = raw.get("clips") or []
    if not isinstance(clips_raw, list):
        raise ValueError(f"{manifest_path}: 'clips' must be a list")

    clips: list[IrClip] = []
    for entry in clips_raw:
        if "path" not in entry:
            raise ValueError(f"{manifest_path}: every clip needs a 'path'")
        if "timeline" not in entry:
            raise ValueError(f"{manifest_path}: clip {entry['path']!r} has no 'timeline'")

        zones = _parse_zones(entry["zones"]) if "zones" in entry else default_zones
        interval = float(entry.get("frame_interval_s", 0.5))
        timeline = [
            TimelineInterval(
                from_s=float(row["from_s"]), to_s=float(row["to_s"]), state=row["state"]
            )
            for row in entry["timeline"]
        ]
        clips.append(
            IrClip(
                path=manifest_path.parent / entry["path"],
                zones=zones,
                frame_interval_s=interval,
                timeline=timeline,
            )
        )
    return clips


@dataclass(frozen=True)
class Tier3Result:
    """What `run_tier3` produces. `accuracy`/`latency` are both `None` when
    the manifest is absent or lists clips whose video files are not
    present on disk -- this bench never fabricates a number for either."""

    clip_count: int
    missing_reason: str | None


def run_tier3(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> Tier3Result:
    """Report tier 3's status. Returns `clip_count=0` and a `missing_reason`
    when there is nothing to score -- currently always, since no infrared
    recordings exist yet. Scoring real clips (decoding video at
    `frame_interval_s`, running a `PoseBackend`, comparing against the
    timeline the same way `synthetic.run_clip` compares against scripted
    ground truth) is intentionally not implemented until there is a real
    manifest and real footage to verify that code against; see the module
    docstring.
    """
    clips = load_manifest(manifest_path)
    if not clips:
        return Tier3Result(
            clip_count=0,
            missing_reason=(
                f"no infrared clips listed in {manifest_path} (or the file does not "
                "exist). Tier 3 is the real bench and milestone 1's definition of "
                "done cannot be met until it is not empty -- see RECORDING.md next "
                "to this file for the recording protocol."
            ),
        )

    missing_files = [str(clip.path) for clip in clips if not clip.path.exists()]
    if missing_files:
        return Tier3Result(
            clip_count=len(clips),
            missing_reason=(
                f"{manifest_path} lists {len(clips)} clip(s) but the video file(s) "
                f"are not present on disk: {', '.join(missing_files)}"
            ),
        )

    return Tier3Result(
        clip_count=len(clips),
        missing_reason=(
            "clip files are present, but this bench does not yet decode video and "
            "score tier 3 (see infrared.py's module docstring) -- report this as a "
            "follow-up once real clips exist, rather than fabricating a result"
        ),
    )
