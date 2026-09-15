import numpy as np
import pytest
import yaml

from perceive.calibrate_bed import calibrate, grow_upward, mask_to_polygon, vote, write_bed_zone


def _rect_mask(height=10, width=20, rows=(2, 7), cols=(4, 15)):
    mask = np.zeros((height, width), dtype=bool)
    mask[rows[0] : rows[1] + 1, cols[0] : cols[1] + 1] = True
    return mask


def test_vote_keeps_pixels_most_frames_agree_on():
    a = np.array([[True, True, False]])
    b = np.array([[True, False, False]])
    c = np.array([[True, True, True]])

    assert vote([a, b, c], 0.5).tolist() == [[True, True, False]]


def test_mask_to_polygon_outlines_a_rectangle_by_its_corners():
    polygon = mask_to_polygon(_rect_mask())

    assert set(polygon) == {(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)}


def test_mask_to_polygon_keeps_a_curved_edge():
    # A bed-like blob whose right edge bulges out and back: every outline
    # point is nearly in line with its neighbours, none of it is straight.
    height, width = 120, 160
    mask = np.zeros((height, width), dtype=bool)
    for y in range(20, 110):
        mask[y, 0 : int(40 + 80 * np.sin(np.pi * (y - 20) / 90))] = True

    polygon = mask_to_polygon(mask)

    xs, ys = np.array([p[0] for p in polygon]), np.array([p[1] for p in polygon])
    area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    assert area == pytest.approx(mask.mean(), rel=0.05)


def test_mask_to_polygon_of_an_empty_mask_is_empty():
    assert mask_to_polygon(np.zeros((10, 20), dtype=bool)) == []


def test_calibrate_ignores_frames_without_a_bed():
    masks = iter([None, _rect_mask(), None, _rect_mask()])

    polygon = calibrate([b"f"] * 4, lambda _jpeg: next(masks), grow_up=0.0)

    assert set(polygon) == {(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)}


def test_calibrate_with_no_bed_anywhere_returns_nothing():
    assert calibrate([b"f", b"f"], lambda _jpeg: None) == []


def test_grow_upward_moves_the_top_edge_and_keeps_the_bottom():
    polygon = [(0.2, 0.5), (0.8, 0.5), (0.8, 0.9), (0.2, 0.9)]

    grown = grow_upward(polygon, 0.25)

    assert grown[0] == (0.2, pytest.approx(0.4))
    assert grown[2] == (0.8, 0.9)


def test_grow_upward_clamps_at_the_top_of_the_frame():
    assert grow_upward([(0.0, 0.1), (1.0, 0.9)], 1.0)[0] == (0.0, 0.0)


def test_write_bed_zone_replaces_bed_and_keeps_other_zones(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text(
        yaml.safe_dump({"bed": [[0, 0], [1, 0], [1, 1]], "door": [[0.9, 0], [1, 0], [1, 1]]})
    )

    write_bed_zone(path, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)])

    zones = yaml.safe_load(path.read_text())
    assert zones["bed"] == [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]]
    assert zones["door"] == [[0.9, 0], [1, 0], [1, 1]]


def test_write_bed_zone_creates_a_missing_file(tmp_path):
    path = tmp_path / "config" / "zones.yaml"

    write_bed_zone(path, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.4)])

    assert yaml.safe_load(path.read_text()) == {"bed": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]]}


def test_a_calibrated_polygon_can_be_written(tmp_path):
    path = tmp_path / "zones.yaml"

    write_bed_zone(path, grow_upward(mask_to_polygon(_rect_mask()), 0.15))

    assert len(yaml.safe_load(path.read_text())["bed"]) == 4
