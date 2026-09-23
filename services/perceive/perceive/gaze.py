"""Map the confirmed person state and current pose to a bedside gaze point."""

from __future__ import annotations

from dataclasses import dataclass

from nc_shared.events import Gaze

from perceive.backends import PoseResult
from perceive.zones import Polygon


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _centroid(polygon: Polygon) -> tuple[float, float]:
    twice_area = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        weighted_x += (x1 + x2) * cross
        weighted_y += (y1 + y2) * cross
    if abs(twice_area) < 1e-12:
        return (
            sum(x for x, _ in polygon) / len(polygon),
            sum(y for _, y in polygon) / len(polygon),
        )
    return weighted_x / (3 * twice_area), weighted_y / (3 * twice_area)


def gaze_for(state: str, pose: PoseResult | None, bed_polygon: Polygon | None) -> Gaze:
    """Choose a point from the confirmed state, with normalized image coordinates."""
    # The bed is checked before the pose: a sleeper under the covers is often
    # undetected while the tracker holds `in_bed`, and the eyes should still
    # rest on the bed rather than centre and dim.
    if state == "in_bed" and bed_polygon:
        x, y = _centroid(bed_polygon)
        return Gaze(source="perceive", target="bed", x=_clamp(x), y=_clamp(y))
    if state == "absent" or pose is None:
        return Gaze(source="perceive", target="none")

    nose = pose.landmarks.get("nose")
    if nose is not None and nose.visibility >= 0.5:
        x, y = nose.x, nose.y
    else:
        x_min, y_min, x_max, y_max = pose.bbox
        x, y = (x_min + x_max) / 2, y_min + 0.1 * (y_max - y_min)
    return Gaze(source="perceive", target="face", x=_clamp(x), y=_clamp(y))


def should_publish_gaze(
    gaze: Gaze, last_published: Gaze | None, *, heartbeat: bool = False
) -> bool:
    """Compare with the last published point, not the previous frame's point."""
    if heartbeat or last_published is None or gaze.target != last_published.target:
        return True
    if gaze.target == "none":
        return False
    assert gaze.x is not None and gaze.y is not None
    assert last_published.x is not None and last_published.y is not None
    return abs(gaze.x - last_published.x) > 0.03 or abs(gaze.y - last_published.y) > 0.03


@dataclass
class GazePublishState:
    latest: Gaze | None = None
    last_published: Gaze | None = None
