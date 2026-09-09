"""Tests for `dashboard.zones_store`: validation, atomic writes, and
round-tripping through the real `perceive.zones.load_zones` (no Redis, no
camera, no browser)."""

from __future__ import annotations

import os

import pytest
from perceive.zones import load_zones

from dashboard.zones_store import (
    MAX_POINTS,
    ZONE_NAMES,
    load_existing_zones,
    save_zones,
    validate_zones,
)

VALID_PAYLOAD = {
    "bed": [[0.0, 0.2], [0.4, 0.2], [0.4, 0.8], [0.0, 0.8]],
    "door": [[0.85, 0.0], [1.0, 0.0], [1.0, 1.0]],
    "bathroom_path": [],
}


def test_validate_zones_accepts_a_well_formed_payload():
    parsed, errors = validate_zones(VALID_PAYLOAD)
    assert errors == []
    assert parsed["bed"] == VALID_PAYLOAD["bed"]
    assert parsed["door"] == VALID_PAYLOAD["door"]
    assert parsed["bathroom_path"] == []


def test_validate_zones_rounds_coordinates():
    payload = {
        "bed": [[0.123456789, 0.0], [0.5, 0.0], [0.5, 0.5]],
        "door": [],
        "bathroom_path": [],
    }
    parsed, errors = validate_zones(payload)
    assert errors == []
    assert parsed["bed"][0] == [0.1235, 0.0]


def test_validate_zones_rejects_too_few_points():
    payload = {"bed": [[0.1, 0.1], [0.2, 0.2]], "door": [], "bathroom_path": []}
    parsed, errors = validate_zones(payload)
    assert parsed is None
    assert any("bed" in e and "at least" in e for e in errors)


def test_validate_zones_rejects_coordinate_outside_0_1():
    payload = {"bed": [[0.1, 0.1], [1.5, 0.2], [0.2, 0.9]], "door": [], "bathroom_path": []}
    parsed, errors = validate_zones(payload)
    assert parsed is None
    assert any("out of range" in e for e in errors)


def test_validate_zones_rejects_non_numeric_coordinate():
    payload = {"bed": [[0.1, 0.1], ["oops", 0.2], [0.2, 0.9]], "door": [], "bathroom_path": []}
    parsed, errors = validate_zones(payload)
    assert parsed is None
    assert any("must be numbers" in e for e in errors)


def test_validate_zones_rejects_too_many_points():
    payload = {
        "bed": [[0.001 * i, 0.001 * i] for i in range(MAX_POINTS + 1)],
        "door": [],
        "bathroom_path": [],
    }
    parsed, errors = validate_zones(payload)
    assert parsed is None
    assert any("at most" in e for e in errors)


def test_validate_zones_rejects_a_non_dict_payload():
    parsed, errors = validate_zones(["not", "a", "dict"])
    assert parsed is None
    assert errors


def test_validate_zones_collects_multiple_errors_at_once():
    payload = {
        "bed": [[0.1, 0.1]],  # too few
        "door": [["nope", 0.2], [0.2, 0.9], [0.3, 0.3]],  # bad coordinate
        "bathroom_path": [],
    }
    parsed, errors = validate_zones(payload)
    assert parsed is None
    assert len(errors) >= 2


def test_save_zones_writes_atomically_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "zones.yaml"
    parsed, errors = validate_zones(VALID_PAYLOAD)
    assert errors == []

    result_path = save_zones(parsed, target)

    assert result_path == target
    assert target.exists()
    leftovers = list(tmp_path.iterdir())
    assert leftovers == [target], f"unexpected files left behind: {leftovers}"


def test_save_zones_round_trips_through_the_real_perceive_loader(tmp_path):
    target = tmp_path / "zones.yaml"
    parsed, errors = validate_zones(VALID_PAYLOAD)
    assert errors == []

    save_zones(parsed, target)

    zone_map = load_zones(target)
    assert zone_map.zone_for_point(0.2, 0.5) == "bed"
    assert zone_map.zone_for_point(0.95, 0.5) == "door"
    # bathroom_path was submitted empty, which is legal and means "not
    # configured" -- perceive.zones degrades that to "other".
    assert zone_map.polygons.get("bathroom_path") is None


def test_save_zones_overwrite_never_leaves_a_partial_file(tmp_path):
    target = tmp_path / "zones.yaml"
    first, _ = validate_zones(VALID_PAYLOAD)
    save_zones(first, target)
    original_bytes = target.read_bytes()

    second, _ = validate_zones(
        {"bed": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]], "door": [], "bathroom_path": []}
    )
    save_zones(second, target)

    updated = target.read_bytes()
    assert updated != original_bytes
    assert list(tmp_path.iterdir()) == [target]
    # A file that parses at all, at every point in this test, is the
    # atomicity property we actually care about: `os.replace` guarantees a
    # reader (or the next save) never observes a half-written file.
    zone_map = load_zones(target)
    assert zone_map.polygons["bed"]


def test_load_existing_zones_returns_empty_polygons_when_nothing_exists(tmp_path):
    result = load_existing_zones(tmp_path / "zones.yaml")
    assert result == {name: [] for name in ZONE_NAMES}


def test_load_existing_zones_reads_back_a_saved_file(tmp_path):
    target = tmp_path / "zones.yaml"
    parsed, _ = validate_zones(VALID_PAYLOAD)
    save_zones(parsed, target)

    result = load_existing_zones(target)

    assert result["bed"] == VALID_PAYLOAD["bed"]
    assert result["door"] == VALID_PAYLOAD["door"]


def test_zones_path_resolution_uses_zones_path_env_var(tmp_path, monkeypatch):
    target = tmp_path / "custom-zones.yaml"
    parsed, _ = validate_zones(VALID_PAYLOAD)
    save_zones(parsed, None, env={"ZONES_PATH": str(target)})

    assert target.exists()
    result = load_existing_zones(None, env={"ZONES_PATH": str(target)})
    assert result["bed"] == VALID_PAYLOAD["bed"]


@pytest.fixture(autouse=True)
def _no_real_env_leak(monkeypatch):
    # Belt and braces: never let a stray ZONES_PATH in the real environment
    # make one of these tests touch the repo's config/zones.yaml.
    monkeypatch.delenv("ZONES_PATH", raising=False)
    yield
    assert "ZONES_PATH" not in os.environ
