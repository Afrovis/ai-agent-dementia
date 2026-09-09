"""Tests for `perceive.zones`: point-in-polygon, precedence, and the yaml
config loader's fallback behaviour. No Redis, no camera."""

from pathlib import Path

from perceive.zones import ZoneMap, load_zones


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
