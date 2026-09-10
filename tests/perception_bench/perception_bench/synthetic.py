"""Tier 1: synthetic scripted clips, generated in-process, no downloads.

Every fixture here is a stick figure whose pose landmarks are known exactly
because this module places them -- the same landmark set `perceive.backends`
normalises every real backend onto (`LANDMARK_NAMES`). A `perceive.backends
.ScriptedBackend` reports those exact landmarks back out, so `run_clip`
drives the real `perceive.classify.StateTracker` and `classify_pose`
end-to-end, without a camera, Redis, or any model weights.

Each frame also renders a small greyscale stick-figure JPEG via Pillow
(`ScriptedFrame.jpeg`), because a person debugging a fixture should be able
to look at it. That image is *not* fed to any pose detector: a hand-drawn
stick figure looks nothing like a real body to MediaPipe or YOLO, so
scoring a real backend against it would measure "can this backend detect
cartoon stick figures", not anything about the `perceive` pipeline. The
known landmarks drive `ScriptedBackend` directly instead; the image is
documentation.

This is the only tier that can measure latency (see `scoring.LatencyResult`
and `run_clip` below): because every frame's ground-truth state is chosen
by this module, the exact frame index a transition "happens" on is known,
which no recorded footage -- daylight or infrared -- can offer without
frame-by-frame hand-labelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from perceive.backends import Landmark, PoseResult, ScriptedBackend
from perceive.classify import ClassifyThresholds, PersonStateName, StateTracker, centroid_of
from perceive.zones import ZoneMap
from PIL import Image, ImageDraw

from perception_bench.scoring import LatencyResult

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# A small room, roughly matching `config/zones.example.yaml`'s shape: bed
# against the left wall, door on the right, bathroom path along the bottom
# right. Self-contained rather than reading the real config file, so this
# tier never depends on `config/` existing or being unmodified.
ROOM_ZONES = ZoneMap(
    polygons={
        "bed": [(0.0, 0.2), (0.35, 0.2), (0.35, 0.85), (0.0, 0.85)],
        "door": [(0.85, 0.0), (1.0, 0.0), (1.0, 1.0), (0.85, 1.0)],
        "bathroom_path": [(0.35, 0.85), (0.85, 0.85), (0.85, 1.0), (0.35, 1.0)],
    }
)


def _pose(points: dict[str, tuple[float, float]], confidence: float = 0.9) -> PoseResult:
    landmarks = {name: Landmark(x=x, y=y, visibility=0.9) for name, (x, y) in points.items()}
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return PoseResult(landmarks=landmarks, bbox=bbox, confidence=confidence)


def in_bed_pose(x: float = 0.15) -> PoseResult:
    """Lying flat, torso horizontal, inside the bed zone."""
    return _pose(
        {
            "nose": (x - 0.10, 0.50),
            "left_shoulder": (x - 0.07, 0.48),
            "right_shoulder": (x - 0.07, 0.52),
            "left_hip": (x + 0.15, 0.50),
            "right_hip": (x + 0.15, 0.54),
            "left_knee": (x + 0.20, 0.50),
            "right_knee": (x + 0.20, 0.54),
            "left_ankle": (x + 0.23, 0.50),
            "right_ankle": (x + 0.23, 0.54),
        }
    )


def sitting_up_pose(x: float = 0.20) -> PoseResult:
    """Upright torso, bent legs -- ankles not clearly below the hips."""
    return _pose(
        {
            "nose": (x, 0.30),
            "left_shoulder": (x - 0.02, 0.35),
            "right_shoulder": (x + 0.02, 0.35),
            "left_hip": (x - 0.01, 0.55),
            "right_hip": (x + 0.01, 0.55),
            "left_knee": (x - 0.01, 0.60),
            "right_knee": (x + 0.01, 0.60),
            "left_ankle": (x - 0.01, 0.58),
            "right_ankle": (x + 0.01, 0.58),
        }
    )


def standing_pose(x: float = 0.50) -> PoseResult:
    """Upright torso, ankles well below the hips, large vertical extent."""
    return _pose(
        {
            "nose": (x, 0.10),
            "left_shoulder": (x - 0.02, 0.20),
            "right_shoulder": (x + 0.02, 0.20),
            "left_hip": (x - 0.01, 0.50),
            "right_hip": (x + 0.01, 0.50),
            "left_knee": (x - 0.01, 0.75),
            "right_knee": (x + 0.01, 0.75),
            "left_ankle": (x - 0.01, 0.95),
            "right_ankle": (x + 0.01, 0.95),
        }
    )


def on_floor_pose(x: float = 0.55) -> PoseResult:
    """Lying flat, low in the frame, outside the bed zone."""
    return _pose(
        {
            "nose": (x - 0.12, 0.90),
            "left_shoulder": (x - 0.08, 0.88),
            "right_shoulder": (x - 0.08, 0.92),
            "left_hip": (x + 0.10, 0.90),
            "right_hip": (x + 0.10, 0.94),
            "left_knee": (x + 0.16, 0.90),
            "right_knee": (x + 0.16, 0.94),
            "left_ankle": (x + 0.20, 0.90),
            "right_ankle": (x + 0.20, 0.94),
        }
    )


def _render_stick_figure(pose: PoseResult | None, label: str) -> bytes:
    """Draw `pose`'s landmarks and connecting bones on a small dark canvas,
    for a human reviewing fixtures. Returns JPEG bytes; see the module
    docstring for why this is never fed to a detector."""
    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), color=(15, 15, 20))
    draw = ImageDraw.Draw(image)

    if pose is not None:
        points = {
            name: (lm.x * FRAME_WIDTH, lm.y * FRAME_HEIGHT) for name, lm in pose.landmarks.items()
        }
        bones = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]
        for a, b in bones:
            if a in points and b in points:
                draw.line([points[a], points[b]], fill=(200, 200, 210), width=2)
        for x, y in points.values():
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(230, 230, 60))

    draw.text((4, 4), label, fill=(120, 200, 255))

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@dataclass(frozen=True)
class ScriptedFrame:
    """One frame of a `ScriptedClip`: the ground truth a volunteer would
    have been asked to act out, and the exact pose that "acting" produces."""

    ground_truth: PersonStateName
    pose: PoseResult | None
    jpeg: bytes = field(repr=False)


@dataclass(frozen=True)
class ScriptedClip:
    """A scripted sequence of `ScriptedFrame`s at a fixed frame interval,
    the synthetic equivalent of `capture`'s `Frame` stream at `CAPTURE_FPS`.
    """

    name: str
    frames: list[ScriptedFrame]
    frame_interval_s: float
    zones: ZoneMap = ROOM_ZONES


def _frames(
    *labelled: tuple[PersonStateName, PoseResult | None], repeat: int = 1
) -> list[ScriptedFrame]:
    out: list[ScriptedFrame] = []
    for ground_truth, pose in labelled:
        jpeg = _render_stick_figure(pose, ground_truth)
        out.extend(ScriptedFrame(ground_truth, pose, jpeg) for _ in range(repeat))
    return out


def build_full_night_clip(frame_interval_s: float = 0.5) -> ScriptedClip:
    """The primary tier-1 fixture: one continuous scripted night covering
    every `PersonStateName` and the transitions PLAN.md section 12 calls
    out (sit up, stand, walk to the door, lie on the floor), plus leaving
    the room. `frame_interval_s` defaults to `CAPTURE_FPS=2.0`'s interval
    (see `.env.example`), so latency numbers this clip produces are
    directly comparable to the real pipeline's frame rate.
    """
    # Frame counts roughly match RECORDING.md's suggested real-world dwell
    # times per action (converted at this clip's frame rate), not the
    # smallest number that "works": a state held for only a couple of
    # `StateTracker.confirm_frames` in a fixture makes its own hysteresis
    # delay dominate the recall number, which would be measuring the
    # fixture's brevity, not the pipeline. Real wake-ups hold each posture
    # for several seconds, so the fixture does too.
    frames: list[ScriptedFrame] = []

    # Settled in bed for a while -- most of a real night, per
    # `perceive.classify.StateTracker._holds_bed`'s docstring.
    frames += _frames(("in_bed", in_bed_pose()), repeat=20)  # ~10s

    # Wakes and sits up.
    frames += _frames(("sitting_up", sitting_up_pose()), repeat=16)  # ~8s

    # Stands, still near the bed.
    frames += _frames(("standing", standing_pose(x=0.30)), repeat=16)  # ~8s

    # Walks toward the door: centroid steps right each frame until
    # `StateTracker`'s displacement threshold is crossed and hysteresis
    # confirms `walking`.
    walk_xs = [0.30 + 0.05 * i for i in range(1, 13)]  # 0.35 .. 0.90, ~6s
    for x in walk_xs:
        frames += _frames(("walking", standing_pose(x=x)))

    # Falls: on_floor bypasses hysteresis entirely (issue #8/HANDOFF rule 5),
    # so this is the transition the <2s latency target matters most for.
    frames += _frames(("on_floor", on_floor_pose(x=0.60)), repeat=20)  # ~10s

    # Nobody gets up off the floor unassisted in this script; the person
    # leaves the frame instead (e.g. helped up and walked out) -- pose
    # detection loses them entirely, which is `absent`, not `in_bed`,
    # since the last confirmed state was not `in_bed`
    # (`StateTracker._holds_bed` only holds a *previous* `in_bed`).
    frames += _frames(("absent", None), repeat=16)  # ~8s

    return ScriptedClip(name="full_night", frames=frames, frame_interval_s=frame_interval_s)


@dataclass(frozen=True)
class ClipRun:
    """The result of driving one `ScriptedClip` through a `StateTracker`."""

    clip_name: str
    pairs: list[tuple[str, str]]
    """`(ground_truth, predicted)` for every frame, predicted being
    whatever the tracker currently believes (its last confirmed state),
    the same read a caregiver's dashboard would show at that instant."""
    latency: LatencyResult
    undetected_transitions: list[str]
    """Ground-truth transitions the tracker never confirmed before the clip
    ended -- a real failure, not a "not measured" case, surfaced separately
    from `latency.samples` since `LatencyResult` only holds numbers."""


def run_clip(clip: ScriptedClip, thresholds: ClassifyThresholds = ClassifyThresholds()) -> ClipRun:
    """Drive every frame of `clip` through a fresh `StateTracker` (built
    with `thresholds`, matching `perceive.main.build_tracker`'s defaults
    unless the caller overrides them) via a `ScriptedBackend`, and score
    the result.

    This bypasses `perceive.main.run_once` deliberately: `run_once` also
    wires in the Redis bus and `nc_shared` event models, neither of which
    this bench needs or wants as a dependency (see the package docstring).
    The three lines this function drives -- `backend.detect`,
    `zones.zone_for_point`, `tracker.update` -- are the entire pipeline
    `run_once` runs per frame; nothing else in `perceive.main` touches
    classification.
    """
    backend = ScriptedBackend([frame.pose for frame in clip.frames])
    tracker = StateTracker(thresholds=thresholds)

    pairs: list[tuple[str, str]] = []
    latency_samples: list[tuple[str, float]] = []
    undetected: list[str] = []

    transition_target: str | None = None
    transition_start_frame = 0
    prev_ground_truth: str | None = None
    now = 0.0

    for index, frame in enumerate(clip.frames):
        pose = backend.detect(b"")  # ScriptedBackend ignores the argument
        zone = clip.zones.zone_for_point(*centroid_of(pose)) if pose is not None else "other"
        tracker.update(pose, zone, now)

        snapshot = tracker.snapshot()
        predicted = snapshot[0] if snapshot is not None else "absent"
        pairs.append((frame.ground_truth, predicted))

        if frame.ground_truth != prev_ground_truth:
            if transition_target is not None:
                undetected.append(f"{prev_ground_truth} -> {transition_target}")
            transition_target = frame.ground_truth
            transition_start_frame = index
            prev_ground_truth = frame.ground_truth

        if transition_target is not None and predicted == transition_target:
            latency = (index - transition_start_frame) * clip.frame_interval_s
            latency_samples.append((f"-> {transition_target}", latency))
            transition_target = None

        now += clip.frame_interval_s

    if transition_target is not None:
        undetected.append(f"{prev_ground_truth} -> {transition_target}")

    return ClipRun(
        clip_name=clip.name,
        pairs=pairs,
        latency=LatencyResult(samples=latency_samples),
        undetected_transitions=undetected,
    )


def default_clips() -> list[ScriptedClip]:
    """Every tier-1 fixture the bench runs by default."""
    return [build_full_night_clip()]


def run_tier1(clips: list[ScriptedClip] | None = None) -> list[ClipRun]:
    """Run every clip in `clips` (`default_clips()` if omitted) and return
    their `ClipRun`s."""
    clips = clips if clips is not None else default_clips()
    return [run_clip(clip) for clip in clips]
