"""Pose-detection backends for `perceive` (issue #8).

Issue #8 asks to "compare MediaPipe Pose vs YOLOv8-pose on IR captures."
No IR captures of consenting volunteers exist yet (HANDOFF.md section 8
anticipates exactly this and says to build against webcam fixtures and mark
the IR evaluation a follow-up -- see issue #11). So the comparison is made
a config flag instead of a one-off script: every backend implements the
same `PoseBackend` protocol and normalises its output onto the same
canonical landmark set (`LANDMARK_NAMES`), so swapping `MediaPipeBackend`
for `YoloPoseBackend` is a `PERCEIVE_POSE_BACKEND` env change, not a code
change, whenever real IR fixtures land.

Normalisation happens *here*, in each backend, not downstream: coordinates
are 0.0-1.0 with the origin top-left, so `perceive.classify` and
`perceive.zones` never need to know a frame's pixel dimensions or which
backend produced a `PoseResult`.

`MediaPipeBackend` and `YoloPoseBackend` both lazy-import their model
library inside `__init__`, exactly like `capture.sources.OpenCvSource`
does for OpenCV: importing this module must never require `mediapipe` or
`ultralytics` to be installed, since `ScriptedBackend` -- what every test
and the future perception bench (issue #11) actually drive -- needs
neither. Both real backends live behind optional extras in
`pyproject.toml` (`mediapipe`, `yolo`); the service must import and its
tests must run with neither installed.
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from perceive.phantom import DEFAULT_MAX_CONFIDENCE, Candidate, KnownPhantoms

LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
"""The canonical landmark set every backend maps onto, and the only one
`perceive.classify` ever reasons about. Minimal on purpose: enough to tell
lying from sitting from standing (issue #8's list), nothing a model might
plausibly disagree on (wrists, fingers, face detail)."""


@dataclass(frozen=True)
class Landmark:
    """One landmark in normalised frame coordinates (0.0-1.0, origin top-left).

    `visibility` is the backend's own confidence that this specific point
    is where it says it is (occluded limbs score low), distinct from
    `PoseResult.confidence`, which is the overall person-detection score.
    """

    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class PoseResult:
    """One detected person's pose, normalised and backend-agnostic.

    `landmarks` maps names in `LANDMARK_NAMES` to a `Landmark`. A backend
    that has a position for a point but little confidence in it reports it
    with low `visibility`; a backend with no position at all for a point
    (ultralytics zeroes the coordinates of keypoints under 0.5 confidence)
    omits the key, and `perceive.classify` falls back to `bbox` for
    anything it cannot measure from the landmarks present. `bbox` is
    `(x_min, y_min, x_max, y_max)`, normalised. `confidence` is the
    backend's overall person-detection score, independent of any single
    landmark.
    """

    landmarks: dict[str, Landmark]
    bbox: tuple[float, float, float, float]
    confidence: float


class PoseBackend(Protocol):
    """The interface `perceive.main` drives: one frame in, one pose or none out."""

    def detect(self, jpeg: bytes) -> PoseResult | None:
        """Return the most confident person detected in `jpeg`, or `None`."""
        ...


def _decode_rgb(jpeg: bytes) -> np.ndarray | None:
    """Decode `jpeg` to an RGB `numpy` array, or `None` if it will not decode.

    Shared by both real backends: MediaPipe consumes this RGB array directly,
    while the YOLO backend converts it to the BGR convention Ultralytics uses
    for numpy inputs. A decode failure degrades to "no person seen" (`None`)
    rather than raising -- a single corrupt frame must not crash the service
    (HANDOFF.md rule 4).
    """
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            return np.asarray(image.convert("RGB"))
    except Exception:  # noqa: BLE001 - any decode failure means "no frame"
        return None


class MediaPipeBackend:
    """Wraps `mediapipe.solutions.pose.Pose`. The default backend (HANDOFF.md
    section 12: "MediaPipe, evaluate both in issue 8").

    `static_image_mode=True` by default because `perceive` hands it
    independent JPEGs from the bus, not a continuous video stream mediapipe
    itself can track frame-to-frame -- letting it assume temporal
    continuity it does not have would silently degrade accuracy. At 2 fps
    the frames are nearly continuous, though, and video mode carries a
    detection across frames the single-image detector drops, so
    `PERCEIVE_MEDIAPIPE_VIDEO_MODE` exposes the other setting for the
    perception bench to measure.
    """

    _LANDMARK_INDEX: dict[str, int] = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    """MediaPipe Pose's 33-point index for the 9 points in `LANDMARK_NAMES`."""

    def __init__(
        self, min_detection_confidence: float = 0.5, *, static_image_mode: bool = True
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for PERCEIVE_POSE_BACKEND=mediapipe. "
                "Install it with `pip install .[mediapipe]`."
            ) from exc

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, jpeg: bytes) -> PoseResult | None:
        """Run mediapipe Pose on `jpeg` and map the result onto `LANDMARK_NAMES`."""
        image = _decode_rgb(jpeg)
        if image is None:
            return None

        result = self._pose.process(image)
        if result.pose_landmarks is None:
            return None

        landmarks: dict[str, Landmark] = {}
        xs: list[float] = []
        ys: list[float] = []
        visibilities: list[float] = []
        for name, index in self._LANDMARK_INDEX.items():
            point = result.pose_landmarks.landmark[index]
            landmarks[name] = Landmark(x=point.x, y=point.y, visibility=point.visibility)
            xs.append(point.x)
            ys.append(point.y)
            visibilities.append(point.visibility)

        # mediapipe does not expose one overall "this is a person" score
        # for a single-image `Pose` call; mean landmark visibility across
        # the points we actually use is a reasonable stand-in, since a
        # confidently detected person has confidently visible joints.
        confidence = sum(visibilities) / len(visibilities)
        bbox = (min(xs), min(ys), max(xs), max(ys))
        return PoseResult(landmarks=landmarks, bbox=bbox, confidence=confidence)

    def close(self) -> None:
        """Release the underlying mediapipe graph."""
        self._pose.close()


class YoloPoseBackend:
    """Wraps an `ultralytics` YOLO-pose model (default weights: YOLO11s-pose;
    `PERCEIVE_YOLO_MODEL` also accepts any other ultralytics pose checkpoint,
    e.g. a YOLOv8-pose one).

    The comparison arm for issue #8's open question (HANDOFF.md section 12).
    Maps the model's 17 COCO keypoints onto `LANDMARK_NAMES`; the 8 COCO
    points with no equivalent here (eyes, ears, elbows, wrists) are dropped,
    not carried through, since `perceive.classify` never looks at them.

    `imgsz` (`PERCEIVE_YOLO_IMGSZ`, default 640) is passed explicitly to
    every inference call instead of relying on ultralytics' own default, so
    the inference resolution is a config knob rather than an implicit
    consequence of whatever the ultralytics version defaults to.

    `phantoms_file` (`PERCEIVE_PHANTOMS_FILE`, default unset -- filtering
    off) guards against a fixed non-person object the model reports as
    "person" with an unmoving box in every frame, including an empty room
    (2026-09-14 bedroom evidence: a headboard-area false positive scored
    high enough to win selection over an empty bed or a person elsewhere in
    frame). Unlike an earlier design, this is *not* learned at runtime --
    that turned out both ineffective (the phantom and the real person never
    actually co-occur in one frame on the real clips) and unsafe (a
    sleeping person's own static box could get permanently misflagged by
    an unrelated mover elsewhere). See `perceive.phantom` for the
    calibrated-box rule this backend applies instead, and
    `perceive.calibrate_phantoms` for producing the file with the room
    empty.
    """

    _KEYPOINT_INDEX: dict[str, int] = {
        "nose": 0,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }
    """COCO-17 keypoint index, as produced by ultralytics' pose models."""

    def __init__(
        self,
        model_path: str = "yolo11s-pose.pt",
        *,
        imgsz: int = 640,
        detect_conf: float | None = None,
        phantoms_file: str | None = None,
        phantom_max_confidence: float | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for PERCEIVE_POSE_BACKEND=yolo. "
                "Install it with `pip install .[yolo]`."
            ) from exc

        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._detect_conf = detect_conf

        # `phantoms_file`/`phantom_max_confidence` default to `None` (rather
        # than baking the env read into `build_backend`'s signature) so a
        # caller that never mentions phantoms at all -- like
        # `video_eval.predict`, which calls `build_backend` without these
        # kwargs -- still gets `PERCEIVE_PHANTOMS_FILE` from its own
        # environment, exactly like running the real service would.
        resolved_path = (
            phantoms_file
            if phantoms_file is not None
            else os.environ.get("PERCEIVE_PHANTOMS_FILE", "").strip() or None
        )
        resolved_max_confidence = (
            phantom_max_confidence
            if phantom_max_confidence is not None
            else float(
                os.environ.get("PERCEIVE_PHANTOM_MAX_CONFIDENCE", str(DEFAULT_MAX_CONFIDENCE))
            )
        )
        self._known_phantoms = KnownPhantoms.load(
            resolved_path, max_confidence=resolved_max_confidence
        )

    def detect(self, jpeg: bytes) -> PoseResult | None:
        """Run the configured YOLO-pose model on `jpeg` and map the best detection
        onto `LANDMARK_NAMES`."""
        detected = self._detect_candidates(jpeg)
        if detected is None:
            return None
        result, candidates = detected
        if result.keypoints is None:
            return None

        # Multiple people can appear in frame; `perceive` tracks the one
        # real person this system is built for. `KnownPhantoms.select`
        # takes every candidate and returns the most confident one that
        # isn't sitting on a calibrated phantom box below the confidence
        # floor, `None` if every candidate in this frame is excluded.
        selected = self._known_phantoms.select(candidates)
        if selected is None:
            return None
        best = selected.index

        points = result.keypoints.xyn[best].tolist()
        point_conf = (
            result.keypoints.conf[best].tolist()
            if result.keypoints.conf is not None
            else [1.0] * len(points)
        )

        landmarks: dict[str, Landmark] = {}
        for name, index in self._KEYPOINT_INDEX.items():
            x, y = points[index]
            if x == 0.0 and y == 0.0:
                # ultralytics zeroes the coordinates of any keypoint whose
                # confidence is under 0.5. The top-left corner is not a
                # position: fed to the geometry it made a lying person's
                # hidden hips read as an upright torso and stretched every
                # body extent to the frame edge (2026-09-13 bedroom clips).
                continue
            landmarks[name] = Landmark(x=float(x), y=float(y), visibility=float(point_conf[index]))

        x_min, y_min, x_max, y_max = result.boxes.xyxyn[best].tolist()
        return PoseResult(
            landmarks=landmarks,
            bbox=(x_min, y_min, x_max, y_max),
            confidence=selected.confidence,
        )

    def detect_candidates(self, jpeg: bytes) -> list[Candidate]:
        """Return every raw candidate in `jpeg`, without phantom filtering."""
        detected = self._detect_candidates(jpeg)
        return [] if detected is None else detected[1]

    def _detect_candidates(self, jpeg: bytes) -> tuple[object, list[Candidate]] | None:
        image = _decode_rgb(jpeg)
        if image is None:
            return None

        # Ultralytics interprets numpy image arrays as OpenCV-style BGR.
        image = np.ascontiguousarray(image[..., ::-1])
        model_kwargs: dict[str, object] = {"verbose": False, "imgsz": self._imgsz}
        if self._detect_conf is not None:
            model_kwargs["conf"] = self._detect_conf
        results = self._model(image, **model_kwargs)
        if not results:
            return None
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        confidences = result.boxes.conf.tolist()
        boxes = result.boxes.xyxyn.tolist()
        candidates = [
            Candidate(box=tuple(box), confidence=float(confidence), index=i)
            for i, (box, confidence) in enumerate(zip(boxes, confidences))
        ]
        return result, candidates


class ScriptedBackend:
    """Returns a pre-supplied sequence of `PoseResult`s (or `None`s), ignoring
    the actual JPEG bytes entirely.

    This is what makes `perceive` testable with no model weights, no
    camera, and no GPU (HANDOFF.md section 4). Tests and the future
    perception bench (issue #11) script an exact sequence of poses -- lying
    in bed, sitting up, standing, walking to the door, on the floor -- and
    assert on the `PersonState` events that come out the other end.

    `cycle=True` repeats the sequence indefinitely once exhausted, useful
    for a long-running smoke test; the default (`False`) exhausts to `None`
    ("no person seen"), which is also a legitimate scripted ending -- e.g.
    a sequence that ends with the person leaving the frame.
    """

    def __init__(self, results: Sequence[PoseResult | None], *, cycle: bool = False) -> None:
        self._results = list(results)
        self._cycle = cycle
        self._index = 0

    def detect(self, jpeg: bytes) -> PoseResult | None:
        del jpeg  # the scripted backend never looks at the actual frame
        if not self._results:
            return None
        if self._index >= len(self._results):
            if not self._cycle:
                return None
            self._index = 0
        result = self._results[self._index]
        self._index += 1
        return result


def build_backend(
    kind: str,
    *,
    model_path: str | None = None,
    static_image_mode: bool = True,
    yolo_imgsz: int = 640,
    detect_conf: float | None = None,
    phantoms_file: str | None = None,
    phantom_max_confidence: float | None = None,
) -> PoseBackend:
    """Return the configured `PoseBackend` for `kind` (`mediapipe`, `yolo`, `scripted`).

    `model_path`, `yolo_imgsz`, `detect_conf`, `phantoms_file` and
    `phantom_max_confidence`
    are only used by `yolo` (`PERCEIVE_YOLO_MODEL`, `PERCEIVE_YOLO_IMGSZ`,
    `PERCEIVE_PHANTOMS_FILE`, `PERCEIVE_PHANTOM_MAX_CONFIDENCE`);
    `static_image_mode` only by `mediapipe` (`PERCEIVE_MEDIAPIPE_VIDEO_MODE`
    inverted). Leaving `phantoms_file`/`phantom_max_confidence` at their
    `None` default (as `video_eval.predict` does, calling this without
    either) makes `YoloPoseBackend` read `PERCEIVE_PHANTOMS_FILE` and
    `PERCEIVE_PHANTOM_MAX_CONFIDENCE` from its own environment, so the
    offline evaluator picks up the same calibrated phantoms the live
    service would without needing to know about them here. `scripted`
    builds an empty `ScriptedBackend` (always reports "no person") -- a
    legitimate config for smoke-testing the rest of the pipeline with
    `docker compose up` and no model weights installed, but production
    nights should use `mediapipe` or `yolo`; real scripted sequences are
    built directly by tests and the perception bench, not through this
    factory. Raises `ValueError` for an unrecognised `kind`, treated as a
    configuration error worth failing loudly on at startup (HANDOFF.md rule
    4).
    """
    if kind == "mediapipe":
        return MediaPipeBackend(static_image_mode=static_image_mode)
    if kind == "yolo":
        return YoloPoseBackend(
            model_path=model_path or "yolo11s-pose.pt",
            imgsz=yolo_imgsz,
            detect_conf=detect_conf,
            phantoms_file=phantoms_file,
            phantom_max_confidence=phantom_max_confidence,
        )
    if kind == "scripted":
        return ScriptedBackend([])
    raise ValueError(f"unknown PERCEIVE_POSE_BACKEND: {kind!r}")
