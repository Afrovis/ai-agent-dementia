"""Tier 2: real daylight RGB footage from the IndoorActionDataset, opt-in.

Source: https://github.com/DaniDeniz/IndoorActionDataset (BSD 3-Clause).
Download it with `fetch_daylight.sh` into `data/perception_bench/daylight/`
(`data/` is gitignored; nothing from the dataset is ever committed). This
module never downloads anything itself and never touches the network: it
only looks for a directory that `fetch_daylight.sh` already populated, and
skips cleanly, with a clear message and exit code 0, if that directory does
not exist.

**What this tier can measure**: per-state accuracy for the four of six
`PersonStateName` states the dataset's classes map onto (see `CLASS_MAP`
below), on real, lit, RGB video of real people, optionally passed through
`perception_bench.degrade.apply_night_degradation` to stress-test a backend
against night-like input.

**What this tier cannot measure, and must not be read as measuring**:

- **Latency.** The dataset has clip-level labels only -- "this clip is
  `walking`" -- with no per-frame annotation and no transition timestamp.
  There is no frame index to measure frames-to-detection from. Only tier 1
  (`perception_bench.synthetic`) and a tier 3 built to the manifest format
  in `perception_bench.infrared` can measure latency.
- **`in_bed`.** The dataset contains no bed and no lying-in-bed class.
  `in_bed` is the state the system spends most of the night in
  (`perceive.classify.StateTracker._holds_bed`'s docstring), and this tier
  says nothing whatsoever about it.
- **Infrared performance.** This is lit RGB video. Passing it through
  `apply_night_degradation` produces something that *looks* more like a
  night IR frame, but see that module's docstring: it is a de-risking
  tool, not evidence. A backend passing this tier under degradation has
  not been shown to work on infrared; it has been shown not to collapse
  under one synthetic approximation of it.

Class-to-state mapping
-----------------------

| Dataset class (best-effort directory name match, see `CLASS_MAP`) | `PersonStateName` |
|---|---|
| falling down                                                      | `on_floor`        |
| lying on the floor                                                | `on_floor`        |
| standing up                                                       | `standing`        |
| walking                                                           | `walking`         |
| no action / empty room                                            | `absent`          |

The exact directory names inside the downloaded archive were not verified
against a live download in this environment (no network access here). This
module matches directory names by keyword (`_CLASS_KEYWORDS`) rather than
an exact string, and logs which directories it found and how each was
classified, specifically so a mismatch is visible and fixable in one place
rather than a silent miscount. If `fetch_daylight.sh` produces different
directory names, adjust `_CLASS_KEYWORDS`, not the calling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from perceive.backends import PoseBackend, build_backend
from perceive.classify import PersonStateName, StateTracker, centroid_of
from perceive.zones import ZoneMap
from PIL import Image

from perception_bench.degrade import apply_night_degradation
from perception_bench.scoring import AccuracyResult, score_predictions

DEFAULT_DATA_DIR = Path("data/perception_bench/daylight")

# Keyword -> PersonStateName. Matched case-insensitively against each
# top-level class directory name found under the dataset root. See the
# module docstring: names are a best-effort guess, not a verified export
# of the real archive layout.
_CLASS_KEYWORDS: tuple[tuple[str, PersonStateName], ...] = (
    ("fall", "on_floor"),
    ("lying", "on_floor"),
    ("lie", "on_floor"),
    ("stand", "standing"),
    ("walk", "walking"),
    ("empty", "absent"),
    ("no_action", "absent"),
    ("noaction", "absent"),
    ("background", "absent"),
)

# No zones in the IndoorActionDataset's room: every frame is scored with an
# empty `ZoneMap`, so every centroid classifies as zone "other" -- correct,
# since `classify_pose`'s `in_bed` branch specifically requires the "bed"
# zone, which does not and cannot exist for this dataset (see the module
# docstring: no bed).
NO_ZONES = ZoneMap()


def classify_directory_name(name: str) -> PersonStateName | None:
    """Map one dataset class directory name onto a `PersonStateName`, or
    `None` if no keyword in `_CLASS_KEYWORDS` matches -- a class this bench
    does not use (e.g. the dataset likely has more classes than our six
    states, such as sitting-down variants we fold into other states, or
    ones with no equivalent at all)."""
    lowered = name.lower()
    for keyword, state in _CLASS_KEYWORDS:
        if keyword in lowered:
            return state
    return None


@dataclass(frozen=True)
class DaylightClip:
    """One clip: an ordered list of frame image paths and the single
    ground-truth state the whole clip is labelled with (clip-level
    labelling, per the dataset -- see the module docstring)."""

    class_dir: str
    ground_truth: PersonStateName
    frame_paths: list[Path]


def discover_clips(data_dir: Path = DEFAULT_DATA_DIR) -> list[DaylightClip]:
    """Scan `data_dir` for class subdirectories this bench recognises, and
    within each, subdirectories of frame images (one subdirectory per
    clip). Returns an empty list -- not an error -- if `data_dir` does not
    exist or contains nothing recognised; `run_tier2` is what turns "empty"
    into the "skip cleanly" message this bench reports.
    """
    if not data_dir.is_dir():
        return []

    clips: list[DaylightClip] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        state = classify_directory_name(class_dir.name)
        if state is None:
            continue
        for clip_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            frame_paths = sorted(
                p for p in clip_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if frame_paths:
                clips.append(
                    DaylightClip(
                        class_dir=class_dir.name,
                        ground_truth=state,
                        frame_paths=frame_paths,
                    )
                )
    return clips


@dataclass(frozen=True)
class Tier2Result:
    """What `run_tier2` produces: either `None` (skipped) or an
    `AccuracyResult`, plus bookkeeping about what was skipped and why."""

    accuracy: AccuracyResult | None
    skipped_reason: str | None
    degraded: bool
    clip_count: int
    pose_backend_name: str


def run_tier2(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    pose_backend_name: str = "mediapipe",
    degrade: bool = False,
    frame_stride: int = 5,
) -> Tier2Result:
    """Score every discovered `DaylightClip` and return a `Tier2Result`.

    Skips cleanly (returns a result with `accuracy=None` and
    `skipped_reason` set) rather than raising when: `data_dir` has no
    recognised clips (most likely, `fetch_daylight.sh` has not been run),
    or `pose_backend_name` needs a library (`mediapipe`/`ultralytics`)
    that is not installed -- both are "no data/no model to test with"
    cases, never bench failures, per this issue's hard constraint that
    missing data is not a failing test.

    `frame_stride` samples every Nth frame rather than every frame: these
    clips can run to hundreds of frames each, and a clip-level label means
    every sampled frame carries the same ground truth, so this is a speed
    trade-off, not an accuracy one -- it does not change what is measured
    (per-state accuracy, not latency, which this tier cannot measure at
    any stride; see the module docstring).
    """
    clips = discover_clips(data_dir)
    if not clips:
        return Tier2Result(
            accuracy=None,
            skipped_reason=(
                f"no daylight clips found under {data_dir}; run "
                "tests/perception_bench/fetch_daylight.sh to download the "
                "IndoorActionDataset, or pass --daylight-data-dir"
            ),
            degraded=degrade,
            clip_count=0,
            pose_backend_name=pose_backend_name,
        )

    try:
        backend = build_backend(pose_backend_name)
    except (RuntimeError, ValueError) as exc:
        return Tier2Result(
            accuracy=None,
            skipped_reason=f"pose backend {pose_backend_name!r} unavailable: {exc}",
            degraded=degrade,
            clip_count=len(clips),
            pose_backend_name=pose_backend_name,
        )

    pairs = _score_clips(clips, backend, degrade=degrade, frame_stride=frame_stride)
    return Tier2Result(
        accuracy=score_predictions(pairs),
        skipped_reason=None,
        degraded=degrade,
        clip_count=len(clips),
        pose_backend_name=pose_backend_name,
    )


def _score_clips(
    clips: list[DaylightClip],
    backend: PoseBackend,
    *,
    degrade: bool,
    frame_stride: int,
) -> list[tuple[str, str]]:
    """Run `backend` over every sampled frame of every clip through a fresh
    `StateTracker` per clip (a new tracker per clip, not one shared across
    all of them, since consecutive clips are unrelated footage and sharing
    hysteresis state across a cut would misattribute it), returning
    `(ground_truth, predicted)` pairs for `scoring.score_predictions`.
    """
    pairs: list[tuple[str, str]] = []
    for clip in clips:
        tracker = StateTracker()
        for index, path in enumerate(clip.frame_paths):
            if index % frame_stride != 0:
                continue
            jpeg = _load_frame_jpeg(path, degrade=degrade)
            pose = backend.detect(jpeg)
            zone = NO_ZONES.zone_for_point(*centroid_of(pose)) if pose is not None else "other"
            tracker.update(pose, zone, float(index))
            snapshot = tracker.snapshot()
            predicted = snapshot[0] if snapshot is not None else "absent"
            pairs.append((clip.ground_truth, predicted))
    return pairs


def _load_frame_jpeg(path: Path, *, degrade: bool) -> bytes:
    """Load one frame from disk, optionally night-degraded, as JPEG bytes
    (what every `PoseBackend.detect` expects, matching `Frame.jpeg` on the
    real bus)."""
    import io

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if degrade:
            rgb = apply_night_degradation(rgb)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG")
        return buffer.getvalue()
