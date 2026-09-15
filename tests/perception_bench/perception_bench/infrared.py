"""Tier 3: real infrared clips from the actual room, scored through
`tools/video_eval` (see `docs/VIDEO_EVAL.md`, step A8). The real bench.

Manifest format
----------------

A YAML file (default `tests/perception_bench/fixtures/ir_manifest.yaml`,
override with `--ir-manifest`) that lives at the data root -- the directory
that also holds `clips/`, the same root `tools/video_eval` reads and writes
(`VIDEO_EVAL_DATA`, default `../data-ai-agent-dementia`):

```yaml
clips:
  - clip_id: 2026-09-13_bedroom-sample-01
    confirmed_by: Name        # copied from reference.yaml at confirm time
    confirmed_at: 2026-09-15
```

Each `clip_id` resolves to `<manifest dir>/clips/<clip_id>/`, the same
layout `video_eval.paths.EvalPaths` uses. Zones, frames, predictions and the
confirmed reference timeline all live under that clip directory -- this
manifest only lists *which* clips tier 3 should score; it carries no zones,
no timeline and no video path of its own (those belonged to the old,
never-populated manifest format this replaces).

`video_eval reconcile --confirm` appends/updates a clip's entry in this file
automatically (see `video_eval.manifest`); nothing else needs to write it by
hand. `confirmed_by`/`confirmed_at` here are informational, copied at
confirm time -- the thing that actually gates scoring is a fresh read of
that clip's own `labels/reference.yaml` (see `run_tier3` below), since the
manifest entry could in principle go stale.

An absent manifest is "skip cleanly" (exit 0, a plain `missing_reason`,
never a fabricated number) -- currently always true, since no infrared
recordings of consenting volunteers exist yet; see `RECORDING.md` next to
this file for the recording protocol. A manifest that exists but is
malformed (not a mapping, `clips` not a list, an entry with no `clip_id`)
raises `ValueError` instead: unlike "no manifest at all", a broken manifest
is something whoever is building this tier should hear about loudly
(HANDOFF.md rule 4's "fail loud").

Scoring itself is delegated entirely to `video_eval.score.score_clip`
(reusing its up-to-date/force and report-writing behaviour) and, optionally,
`video_eval.predict.predict_clip`. `video_eval` is a large, model-dependent
package (MediaPipe/YOLOv8, `capture`, `perceive`) that `perception_bench`
must keep working without -- see `run_tier3`'s lazy import.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from perception_bench.scoring import ACCURACY_TARGET, LATENCY_TARGET_S

DEFAULT_MANIFEST_PATH = Path("tests/perception_bench/fixtures/ir_manifest.yaml")

# A prediction tag is `f"{backend}-{variant}-{sha[:8]}{'-g' if gated else ''}"`
# (see `video_eval.predict.prediction_tag`) -- an 8-hex-digit short SHA with
# an optional trailing `-g`. Stripping that suffix pools tags produced by the
# same backend/variant across clips scored at different commits.
_TAG_SHA_SUFFIX_RE = re.compile(r"-[0-9a-f]{7,40}(-g)?$")


def tag_family(tag: str) -> str:
    """`tag` with its trailing git-sha component (and optional `-g`) removed."""
    return _TAG_SHA_SUFFIX_RE.sub("", tag)


@dataclass(frozen=True)
class ManifestClip:
    """One manifest-listed clip: its `video_eval` clip id, plus the
    confirmer/date recorded at `reconcile --confirm` time."""

    clip_id: str
    confirmed_by: str | None
    confirmed_at: str | None


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> list[ManifestClip]:
    """Load `manifest_path` into a list of `ManifestClip`s. Returns an empty
    list -- not an error -- if the file does not exist (see module
    docstring). Raises `ValueError` for a manifest that exists but is
    malformed.
    """
    if not manifest_path.exists():
        return []

    with manifest_path.open() as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path} must contain a mapping with a 'clips' key")

    clips_raw = raw.get("clips") or []
    if not isinstance(clips_raw, list):
        raise ValueError(f"{manifest_path}: 'clips' must be a list")

    clips: list[ManifestClip] = []
    for entry in clips_raw:
        if not isinstance(entry, dict) or not entry.get("clip_id"):
            raise ValueError(f"{manifest_path}: every clip needs a 'clip_id'")
        clips.append(
            ManifestClip(
                clip_id=str(entry["clip_id"]),
                confirmed_by=entry.get("confirmed_by"),
                confirmed_at=entry.get("confirmed_at"),
            )
        )
    return clips


@dataclass(frozen=True)
class ClipTagResult:
    """One manifest clip scored against one prediction tag, read straight
    out of `reports/<tag>.json` (see `video_eval.score.score_clip`)."""

    clip_id: str
    tag: str
    frame_count: int
    overall_accuracy: float | None
    standing_recall: float | None
    on_floor_recall: float | None
    on_floor_latency_s: float | None
    """Max on_floor detection delay in seconds (`gates.on_floor_delay.value_s`
    in the report), `None` when no on_floor event was in the reference."""
    gates: dict[str, Any]
    """Exactly `report["gates"]` as `score.py` wrote it."""
    confusion: dict[str, dict[str, int]] = field(repr=False)
    """`report["frames"]["all"]["exact"]["confusion"]` -- the same frame view
    `score.py` uses for its gates. Kept for tag-family pooling; not itself
    part of the per-clip report table."""


@dataclass(frozen=True)
class PooledResult:
    """One tag family's confusion counts summed across every clip scored
    under it, and the recall/latency gates recomputed from that pooled
    confusion (using `perception_bench.scoring`'s target constants)."""

    tag_family: str
    clip_ids: list[str]
    frame_count: int
    overall_accuracy: float | None
    standing_recall: float | None
    on_floor_recall: float | None
    on_floor_latency_s: float | None
    """Max on_floor latency across the pooled clips' own max latencies."""
    gates: dict[str, Any]


@dataclass(frozen=True)
class Tier3Result:
    """What `run_tier3` produces. `missing_reason` is set (and everything
    else empty) whenever there is nothing to score -- an absent manifest, no
    `video_eval` installed, or every listed clip skipped -- so this bench
    never fabricates a number for either."""

    clip_count: int
    missing_reason: str | None
    clip_results: list[ClipTagResult] = field(default_factory=list)
    pooled_results: list[PooledResult] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    """`(clip_id, reason)` for every manifest clip that was not scored."""


def _recall_from_confusion(confusion: dict[str, dict[str, int]], state: str) -> float | None:
    row = confusion.get(state)
    if not row:
        return None
    support = sum(row.values())
    if support == 0:
        return None
    return row.get(state, 0) / support


def _recall_gate(recall: float | None) -> dict[str, Any]:
    return {
        "measured": recall is not None,
        "met": None if recall is None else recall >= ACCURACY_TARGET,
        "value": recall,
        "target": ACCURACY_TARGET,
    }


def _latency_gate(latency: float | None, *, measured: bool) -> dict[str, Any]:
    return {
        "measured": measured,
        "met": None if not measured else (latency is not None and latency <= LATENCY_TARGET_S),
        "value_s": latency,
        "target_s": LATENCY_TARGET_S,
    }


def _clip_result_from_report(clip_id: str, tag: str, report: dict[str, Any]) -> ClipTagResult:
    exact = report["frames"]["all"]["exact"]
    per_state = exact["per_state"]
    gates = report["gates"]
    return ClipTagResult(
        clip_id=clip_id,
        tag=tag,
        frame_count=exact["frame_count"],
        overall_accuracy=exact["overall_accuracy"],
        standing_recall=per_state.get("standing", {}).get("recall"),
        on_floor_recall=per_state.get("on_floor", {}).get("recall"),
        on_floor_latency_s=gates["on_floor_delay"]["value_s"],
        gates=gates,
        confusion=exact["confusion"],
    )


def _pool_by_tag_family(clip_results: list[ClipTagResult]) -> list[PooledResult]:
    from perception_bench.scoring import STATE_NAMES

    groups: dict[str, list[ClipTagResult]] = {}
    for result in clip_results:
        groups.setdefault(tag_family(result.tag), []).append(result)

    pooled: list[PooledResult] = []
    for family in sorted(groups):
        rows = groups[family]
        confusion: dict[str, dict[str, int]] = {}
        for row in rows:
            for actual, predicted_counts in row.confusion.items():
                bucket = confusion.setdefault(actual, {})
                for predicted, count in predicted_counts.items():
                    bucket[predicted] = bucket.get(predicted, 0) + count

        frame_count = sum(row.frame_count for row in rows)
        overall_accuracy = None
        if frame_count:
            correct = sum(confusion.get(state, {}).get(state, 0) for state in STATE_NAMES)
            overall_accuracy = correct / frame_count

        standing_recall = _recall_from_confusion(confusion, "standing")
        on_floor_recall = _recall_from_confusion(confusion, "on_floor")

        latencies = [row.on_floor_latency_s for row in rows if row.on_floor_latency_s is not None]
        on_floor_latency = max(latencies) if latencies else None
        latency_measured = any(row.gates["on_floor_delay"]["measured"] for row in rows)

        pooled.append(
            PooledResult(
                tag_family=family,
                clip_ids=[row.clip_id for row in rows],
                frame_count=frame_count,
                overall_accuracy=overall_accuracy,
                standing_recall=standing_recall,
                on_floor_recall=on_floor_recall,
                on_floor_latency_s=on_floor_latency,
                gates={
                    "standing_recall": _recall_gate(standing_recall),
                    "on_floor_recall": _recall_gate(on_floor_recall),
                    "on_floor_delay": _latency_gate(on_floor_latency, measured=latency_measured),
                },
            )
        )
    return pooled


def run_tier3(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    tag_glob: str | None = None,
    backend: str | None = None,
    variant: str = "squash",
    rescore: bool = False,
) -> Tier3Result:
    """Score every confirmed clip in `manifest_path` with `video_eval`.

    - No manifest, or an empty `clips` list: clean skip (see module
      docstring).
    - `video_eval` not installed: clean skip naming the install command --
      `perception_bench` must keep working without it (see module
      docstring).
    - Per clip: skipped with a reason if `labels/reference.yaml` is missing
      or not confirmed (no `confirmed_by`), or if it has no prediction files
      after `tag_glob` filtering. Otherwise scored per matching tag with
      `video_eval.score.score_clip` (optionally preceded by
      `video_eval.predict.predict_clip` when `backend` is given), then
      pooled per tag family (see `tag_family`).
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

    try:
        from video_eval.paths import EvalPaths
        from video_eval.score import score_clip
    except ImportError:
        return Tier3Result(
            clip_count=len(clips),
            missing_reason=(
                f"{manifest_path} lists {len(clips)} clip(s) but tools/video_eval is "
                "not installed in this environment -- install it (e.g. `pip install "
                "-e tools/video_eval`) to score tier 3's confirmed clips"
            ),
        )

    data_root = manifest_path.parent
    clip_results: list[ClipTagResult] = []
    skipped: list[tuple[str, str]] = []

    for clip in clips:
        paths = EvalPaths.for_clip(clip.clip_id, data_root)
        reference_path = paths.labels / "reference.yaml"
        if not reference_path.exists():
            skipped.append((clip.clip_id, "labels/reference.yaml is missing"))
            continue
        reference_raw = yaml.safe_load(reference_path.read_text(encoding="utf-8")) or {}
        if not reference_raw.get("confirmed_by"):
            skipped.append(
                (clip.clip_id, "labels/reference.yaml is not confirmed (no confirmed_by)")
            )
            continue

        if backend is not None:
            from video_eval.predict import predict_clip

            predicted = predict_clip(
                clip.clip_id,
                root=data_root,
                backend_name=backend,
                variant=variant,
                force=rescore,
            )
            tags = [predicted["tag"]]
        else:
            tags = [path.stem for path in sorted(paths.predictions.glob("*.jsonl"))]

        if tag_glob:
            tags = [tag for tag in tags if fnmatch.fnmatch(tag, tag_glob)]

        if not tags:
            skipped.append(
                (clip.clip_id, "no prediction files found under predictions/ (after filtering)")
            )
            continue

        for tag in tags:
            score_clip(clip.clip_id, root=data_root, tag=tag, force=rescore)
            report_path = paths.reports / f"{tag}.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            clip_results.append(_clip_result_from_report(clip.clip_id, tag, report))

    return Tier3Result(
        clip_count=len(clips),
        missing_reason=None,
        clip_results=clip_results,
        pooled_results=_pool_by_tag_family(clip_results),
        skipped=skipped,
    )
