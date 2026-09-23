"""Gaze geometry and publishing thresholds, without a model or Redis."""

import pytest
from nc_shared.events import Gaze

from perceive.backends import Landmark, PoseResult
from perceive.gaze import gaze_for, should_publish_gaze


def _pose(*, nose_visibility: float = 0.9, nose_x: float = 0.3) -> PoseResult:
    return PoseResult(
        landmarks={"nose": Landmark(nose_x, 0.25, nose_visibility)},
        bbox=(0.2, 0.2, 0.6, 0.8),
        confidence=0.9,
    )


def _gaze(target: str, x: float | None = None, y: float | None = None) -> Gaze:
    return Gaze(source="perceive", target=target, x=x, y=y)


def test_visible_nose_is_face_point():
    gaze = gaze_for("standing", _pose(nose_visibility=0.5), None)
    assert (gaze.target, gaze.x, gaze.y) == ("face", 0.3, 0.25)


def test_hidden_nose_uses_top_of_box():
    gaze = gaze_for("standing", _pose(nose_visibility=0.49), None)
    assert gaze.target == "face"
    assert gaze.x == 0.4
    assert abs(gaze.y - 0.26) < 1e-12


def test_in_bed_uses_polygon_area_centroid():
    polygon = [(0.0, 0.0), (0.8, 0.0), (0.8, 0.4), (0.0, 0.8)]
    gaze = gaze_for("in_bed", _pose(), polygon)
    assert gaze.target == "bed"
    assert abs(gaze.x - 0.35555555555555557) < 1e-12
    assert abs(gaze.y - 0.3111111111111111) < 1e-12


def test_degenerate_polygon_uses_vertex_mean():
    gaze = gaze_for("in_bed", _pose(), [(0.0, 0.0), (0.3, 0.3), (0.9, 0.9)])
    assert gaze.target == "bed"
    assert abs(gaze.x - 0.4) < 1e-12
    assert abs(gaze.y - 0.4) < 1e-12


def test_in_bed_without_bed_zone_falls_back_to_face():
    gaze = gaze_for("in_bed", _pose(), None)
    assert (gaze.target, gaze.x, gaze.y) == ("face", 0.3, 0.25)


def test_absent_has_no_point_even_with_pose():
    gaze = gaze_for("absent", _pose(), [(0.0, 0.0)])
    assert (gaze.target, gaze.x, gaze.y) == ("none", None, None)


def test_no_pose_while_in_bed_still_looks_at_bed():
    # A sleeper under the covers is often undetected while in_bed is held.
    gaze = gaze_for("in_bed", None, [(0.2, 0.4), (0.6, 0.4), (0.6, 0.8), (0.2, 0.8)])
    assert (gaze.target, gaze.x, gaze.y) == ("bed", pytest.approx(0.4), pytest.approx(0.6))


def test_no_pose_has_no_point_when_up():
    gaze = gaze_for("standing", None, None)
    assert (gaze.target, gaze.x, gaze.y) == ("none", None, None)


def test_coordinates_are_clamped():
    gaze = gaze_for("standing", _pose(nose_x=1.2), None)
    assert (gaze.x, gaze.y) == (1.0, 0.25)


def test_sub_threshold_jitter_is_not_published():
    last = _gaze("face", 0.5, 0.5)
    assert not should_publish_gaze(_gaze("face", 0.529, 0.48), last)


def test_over_threshold_move_is_published():
    last = _gaze("face", 0.5, 0.5)
    assert should_publish_gaze(_gaze("face", 0.531, 0.5), last)


def test_target_change_is_published():
    assert should_publish_gaze(_gaze("bed", 0.5, 0.5), _gaze("face", 0.5, 0.5))
    assert should_publish_gaze(_gaze("none"), _gaze("face", 0.5, 0.5))


def test_heartbeat_always_publishes():
    gaze = _gaze("none")
    assert should_publish_gaze(gaze, gaze, heartbeat=True)
