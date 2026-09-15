"""Tests for `perceive.zones`: point-in-polygon, precedence, and the yaml
config loader's fallback behaviour. No Redis, no camera."""

from pathlib import Path

from perceive.backends import LANDMARK_NAMES, Landmark, PoseResult
from perceive.classify import centroid_of, zone_for_pose
from perceive.zones import ZoneMap, load_zones

BED = [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)]


def _pose(
    landmark_point: tuple[float, float],
    bbox: tuple[float, float, float, float],
    *,
    complete: bool = True,
) -> PoseResult:
    names = LANDMARK_NAMES if complete else LANDMARK_NAMES[:-1]
    landmarks = {
        name: Landmark(x=landmark_point[0], y=landmark_point[1], visibility=0.9) for name in names
    }
    return PoseResult(landmarks=landmarks, bbox=bbox, confidence=0.9)


def test_zone_for_point_inside_bed_polygon():
    zones = ZoneMap(polygons={"bed": [(0.0, 0.0), (0.4, 0.0), (0.4, 0.8), (0.0, 0.8)]})
    assert zones.zone_for_point(0.2, 0.4) == "bed"


def test_zone_for_point_outside_every_polygon_is_other():
    zones = ZoneMap(polygons={"bed": [(0.0, 0.0), (0.4, 0.0), (0.4, 0.8), (0.0, 0.8)]})
    assert zones.zone_for_point(0.9, 0.9) == "other"


def test_zone_for_point_with_no_zones_configured_is_other():
    assert ZoneMap().zone_for_point(0.5, 0.5) == "other"


def test_zone_precedence_bed_before_door_before_bathroom_path():
    # All three polygons cover the same point on purpose, to exercise the
    # fixed precedence documented in perceive.zones._PRECEDENCE.
    overlapping = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    zones = ZoneMap(
        polygons={"bed": overlapping, "door": overlapping, "bathroom_path": overlapping}
    )
    assert zones.zone_for_point(0.5, 0.5) == "bed"

    zones_no_bed = ZoneMap(polygons={"door": overlapping, "bathroom_path": overlapping})
    assert zones_no_bed.zone_for_point(0.5, 0.5) == "door"

    zones_only_path = ZoneMap(polygons={"bathroom_path": overlapping})
    assert zones_only_path.zone_for_point(0.5, 0.5) == "bathroom_path"


def test_zone_for_pose_returns_bed_when_centroid_and_bbox_centre_are_in_bed():
    zones = ZoneMap(polygons={"bed": BED})

    assert zone_for_pose(zones, _pose((0.2, 0.5), (0.1, 0.2, 0.3, 0.8))) == "bed"


def test_zone_for_pose_returns_other_when_only_centroid_is_in_bed():
    zones = ZoneMap(polygons={"bed": BED})

    assert zone_for_pose(zones, _pose((0.2, 0.5), (0.2, 0.2, 0.8, 0.8))) == "other"


def test_zone_for_pose_uses_remaining_precedence_when_only_centroid_is_in_bed():
    door = [(0.1, 0.2), (0.3, 0.2), (0.3, 0.8), (0.1, 0.8)]
    bathroom_path = [(0.0, 0.0), (0.35, 0.0), (0.35, 1.0), (0.0, 1.0)]
    zones = ZoneMap(polygons={"bed": BED, "door": door, "bathroom_path": bathroom_path})
    pose = _pose((0.2, 0.5), (0.2, 0.2, 0.8, 0.8))

    assert zone_for_pose(zones, pose) == "door"


def test_zone_for_pose_leaves_non_bed_centroid_result_unchanged():
    zones = ZoneMap(polygons={"bed": BED})
    pose = _pose((0.7, 0.5), (0.1, 0.2, 0.3, 0.8))

    assert zone_for_pose(zones, pose) == "other"


def test_zone_for_pose_with_missing_landmarks_matches_point_lookup():
    zones = ZoneMap(polygons={"bed": BED})
    pose = _pose((0.8, 0.5), (0.1, 0.2, 0.3, 0.8), complete=False)

    assert zone_for_pose(zones, pose) == zones.zone_for_point(*centroid_of(pose)) == "bed"


def test_load_zones_reads_the_example_yaml_shipped_in_the_repo():
    repo_root = Path(__file__).resolve().parents[3]
    example = repo_root / "config" / "zones.example.yaml"
    assert example.exists(), "config/zones.example.yaml must exist"

    zones = load_zones(example)

    # A point squarely inside the shipped example bed polygon.
    assert zones.zone_for_point(0.2, 0.5) == "bed"


def test_load_zones_falls_back_to_example_when_primary_missing(tmp_path):
    example = tmp_path / "zones.example.yaml"
    example.write_text("bed: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]\n")

    zones = load_zones(tmp_path / "zones.yaml")  # does not exist

    assert zones.zone_for_point(0.5, 0.5) == "bed"


def test_load_zones_degrades_to_empty_when_nothing_exists(tmp_path):
    zones = load_zones(tmp_path / "does-not-exist.yaml")
    assert zones.polygons == {}
    assert zones.zone_for_point(0.5, 0.5) == "other"


def test_load_zones_degrades_to_empty_on_malformed_yaml(tmp_path):
    bad = tmp_path / "zones.yaml"
    bad.write_text("not: [valid, - yaml: :::\n")

    zones = load_zones(bad)

    assert zones.polygons == {}


def test_load_zones_ignores_unconfigured_zone_keys(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text("bed: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]\ndoor: []\n")

    zones = load_zones(path)

    assert set(zones.polygons) == {"bed"}


def test_load_zones_reads_zones_path_from_env(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("door: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]\n")

    zones = load_zones(env={"ZONES_PATH": str(path)})

    assert zones.zone_for_point(0.5, 0.5) == "door"
