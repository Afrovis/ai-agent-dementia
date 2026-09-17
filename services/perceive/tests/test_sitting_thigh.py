from perceive.backends import Landmark, PoseResult
from perceive.classify import ClassifyThresholds, classify_pose
from perceive.main import PerceiveConfig, build_tracker


def _pose(points: dict[str, tuple[float, float]]) -> PoseResult:
    landmarks = {name: Landmark(x=x, y=y, visibility=0.9) for name, (x, y) in points.items()}
    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]
    return PoseResult(
        landmarks=landmarks, bbox=(min(xs), min(ys), max(xs), max(ys)), confidence=0.9
    )


def _upright(knee_y: float | None) -> PoseResult:
    """Torso vertical, ankles hanging well below the hips: `standing` by the
    ankle rule alone. `knee_y` sets how far the thigh drops (torso is 0.2)."""
    points = {
        "nose": (0.30, 0.35),
        "left_shoulder": (0.28, 0.40),
        "right_shoulder": (0.32, 0.40),
        "left_hip": (0.28, 0.60),
        "right_hip": (0.32, 0.60),
        "left_ankle": (0.33, 0.90),
        "right_ankle": (0.37, 0.90),
    }
    if knee_y is not None:
        points["left_knee"] = (0.33, knee_y)
        points["right_knee"] = (0.37, knee_y)
    return _pose(points)


ON = ClassifyThresholds(sitting_thigh_ratio=0.55)


def test_level_thigh_on_the_bed_edge_is_sitting_up():
    state, _ = classify_pose(_upright(knee_y=0.62), "bed", ON)

    assert state == "sitting_up"


def test_dropping_thigh_is_still_standing():
    state, _ = classify_pose(_upright(knee_y=0.75), "bed", ON)

    assert state == "standing"


def test_rule_needs_visible_knees():
    state, _ = classify_pose(_upright(knee_y=None), "bed", ON)

    assert state == "standing"


def test_rule_is_off_by_default():
    state, _ = classify_pose(_upright(knee_y=0.62), "bed", ClassifyThresholds())

    assert state == "standing"


def test_env_var_reaches_the_tracker():
    config = PerceiveConfig.from_env({"PERCEIVE_SITTING_THIGH_RATIO": "0.55"})

    assert build_tracker(config).thresholds.sitting_thigh_ratio == 0.55
    assert build_tracker(PerceiveConfig.from_env({})).thresholds.sitting_thigh_ratio == 0.0
