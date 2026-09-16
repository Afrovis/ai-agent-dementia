"""Find phantom objects by tracking every raw detection candidate, not just
the one `KnownPhantoms.select()` picked each frame.

`phantom_jump_finder.py` looks at the already-selected single box per frame
and infers a phantom indirectly, from teleports in that one trajectory. This
script instead greedily IoU-matches *every* candidate box across consecutive
frames into short per-object tracks, the way a minimal multi-object tracker
would, using raw-candidate cache files in the `detection_cache.py` schema
(`{"frame_index", "t_s", "dets": [{"bbox", "conf", ...}, ...]}`).

With exactly one person in frame, the signature is direct rather than
threshold-tuned: the real person's track has high center-position variance
(people move) and a moderate lifespan; a phantom object's track has many
frames, near-zero center variance, and often a confidence band that
overlaps the person's -- which is exactly the ambiguity that fools
`KnownPhantoms.select()` into a wrong pick. A track need not span the whole
video: it only has to survive gaps up to `--max-gap` frames, so a phantom
that is only visible above the raw confidence floor some of the time still
accumulates one track instead of many fragments.

Usage:
    python tools/video_eval/scripts/phantom_track_finder.py \
        --cache ../data-ai-agent-dementia/clips/<clip1>/cache/<model>-<variant>.jsonl \
        --cache ../data-ai-agent-dementia/clips/<clip2>/cache/<model>-<variant>.jsonl \
        --out /tmp/phantom_tracks.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b
    inter_x_min = max(ax_min, bx_min)
    inter_y_min = max(ay_min, by_min)
    inter_x_max = min(ax_max, bx_max)
    inter_y_max = min(ay_max, by_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
    area_b = max(0.0, bx_max - bx_min) * max(0.0, by_max - by_min)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def center(box: Box) -> tuple[float, float]:
    x_min, y_min, x_max, y_max = box
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _median_box(boxes: list[Box]) -> Box:
    return (
        _median([b[0] for b in boxes]),
        _median([b[1] for b in boxes]),
        _median([b[2] for b in boxes]),
        _median([b[3] for b in boxes]),
    )


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


@dataclass
class Track:
    clip: str
    frames: list[int] = field(default_factory=list)
    t_s: list[float] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    last_frame: int = -1

    def gap_since(self, frame_index: int) -> int:
        return frame_index - self.last_frame - 1

    def add(self, frame_index: int, t_s: float, box: Box, conf: float) -> None:
        self.frames.append(frame_index)
        self.t_s.append(t_s)
        self.boxes.append(box)
        self.confidences.append(conf)
        self.last_frame = frame_index

    @property
    def span_frames(self) -> int:
        return self.frames[-1] - self.frames[0] + 1 if self.frames else 0

    @property
    def center_stdev(self) -> float:
        """Spread of the track's box centers. Near zero means the box barely
        moved despite persisting -- the phantom signature. A real, moving
        person's track has this well above zero."""
        cx = [center(b)[0] for b in self.boxes]
        cy = [center(b)[1] for b in self.boxes]
        return (_stdev(cx) ** 2 + _stdev(cy) ** 2) ** 0.5

    def summary(self) -> dict:
        return {
            "clip": self.clip,
            "box": list(_median_box(self.boxes)),
            "n_frames": len(self.frames),
            "span_frames": self.span_frames,
            "first_frame": self.frames[0],
            "last_frame": self.frames[-1],
            "first_t_s": round(self.t_s[0], 1),
            "last_t_s": round(self.t_s[-1], 1),
            "center_stdev": round(self.center_stdev, 4),
            "mean_confidence": round(sum(self.confidences) / len(self.confidences), 3),
            "min_confidence": round(min(self.confidences), 3),
            "max_confidence": round(max(self.confidences), 3),
        }


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_tracks(
    rows: list[dict],
    clip: str,
    *,
    match_iou: float,
    max_gap: int,
    conf_floor: float,
) -> list[Track]:
    """Greedy per-frame IoU matching: each detection extends the open track
    whose most recent box it overlaps best (>= match_iou) if that track's gap
    since its last hit is <= max_gap, else it starts a new track. This is
    deliberately the simplest tracker that works -- no Kalman filter, no
    Hungarian assignment -- because with one room and typically 1-3 boxes per
    frame, greedy nearest-IoU is enough to separate a moving person from a
    static object."""
    open_tracks: list[Track] = []
    finished: list[Track] = []

    for row in rows:
        frame_index = row["frame_index"]
        t_s = row.get("t_s", 0.0)
        dets = [d for d in row.get("dets", []) if d["conf"] >= conf_floor]

        used = set()
        for track in open_tracks:
            if track.gap_since(frame_index) > max_gap:
                continue
            best_i, best_iou = None, 0.0
            for i, det in enumerate(dets):
                if i in used:
                    continue
                score = iou(tuple(det["bbox"]), track.boxes[-1])
                if score > best_iou:
                    best_i, best_iou = i, score
            if best_i is not None and best_iou >= match_iou:
                det = dets[best_i]
                track.add(frame_index, t_s, tuple(det["bbox"]), det["conf"])
                used.add(best_i)

        stale = [t for t in open_tracks if t.gap_since(frame_index) > max_gap]
        for t in stale:
            open_tracks.remove(t)
            finished.append(t)

        for i, det in enumerate(dets):
            if i in used:
                continue
            new_track = Track(clip=clip)
            new_track.add(frame_index, t_s, tuple(det["bbox"]), det["conf"])
            open_tracks.append(new_track)

    finished.extend(open_tracks)
    return finished


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", action="append", required=True, dest="caches")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--max-gap",
        type=int,
        default=3,
        help="Frames a track may go undetected and still be extended on re-appearance",
    )
    parser.add_argument("--conf-floor", type=float, default=0.0)
    parser.add_argument("--min-frames", type=int, default=8, help="Drop tracks shorter than this")
    parser.add_argument(
        "--max-center-stdev",
        type=float,
        default=0.03,
        help="Tracks with center spread above this look like a moving person, not a phantom",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    all_tracks: list[Track] = []
    for cache_path in args.caches:
        path = Path(cache_path)
        rows = read_jsonl(path)
        clip = path.parent.parent.name  # .../<clip>/cache/<file>.jsonl
        tracks = build_tracks(
            rows,
            clip,
            match_iou=args.match_iou,
            max_gap=args.max_gap,
            conf_floor=args.conf_floor,
        )
        print(f"{clip} ({path.name}): {len(rows)} rows -> {len(tracks)} tracks")
        all_tracks.extend(tracks)

    candidates = [
        t
        for t in all_tracks
        if len(t.frames) >= args.min_frames and t.center_stdev <= args.max_center_stdev
    ]
    candidates.sort(key=lambda t: (len(t.frames), -t.center_stdev), reverse=True)

    print(
        f"\n{len(candidates)} phantom-track candidates "
        f"(>= {args.min_frames} frames, center_stdev <= {args.max_center_stdev}):\n"
    )
    for t in candidates:
        s = t.summary()
        print(
            f"  clip={s['clip']} box={[round(v, 3) for v in s['box']]} "
            f"n_frames={s['n_frames']} span={s['span_frames']} "
            f"t_s=[{s['first_t_s']}, {s['last_t_s']}] "
            f"center_stdev={s['center_stdev']} "
            f"conf=[{s['min_confidence']}, {s['mean_confidence']}, {s['max_confidence']}]"
        )

    # For context: the longest moving (non-phantom-shaped) tracks, so the
    # real person's numbers are visible for comparison.
    moving = [t for t in all_tracks if len(t.frames) >= args.min_frames and t.center_stdev > args.max_center_stdev]
    moving.sort(key=lambda t: len(t.frames), reverse=True)
    print(f"\nFor comparison, {len(moving)} longer tracks that DO move (likely the real person):")
    for t in moving[:5]:
        s = t.summary()
        print(
            f"  clip={s['clip']} box={[round(v, 3) for v in s['box']]} "
            f"n_frames={s['n_frames']} center_stdev={s['center_stdev']} "
            f"conf=[{s['min_confidence']}, {s['mean_confidence']}, {s['max_confidence']}]"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as handle:
            yaml.safe_dump(
                {"phantom_track_candidates": [t.summary() for t in candidates]},
                handle,
                sort_keys=False,
            )
        print(f"\nWrote {len(candidates)} candidates to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
