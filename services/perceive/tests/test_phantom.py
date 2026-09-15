"""Tests for `perceive.phantom`: pure, no model weights, camera, or GPU.

Replaces an earlier runtime-co-occurrence design (2026-09-14): real bedroom
clips showed the phantom box and the real person never actually share a
frame, so that trigger never fired, and worse, it could permanently
misflag a genuinely still, sleeping person's own track if anything else in
frame moved. `KnownPhantoms` instead applies a short, pre-calibrated list of
boxes with no cross-frame memory at all.
"""

from perceive.phantom import Box, Candidate, KnownPhantoms, cluster_static_boxes

PHANTOM_BOX: Box = (0.19, 0.31, 0.26, 0.46)
"""The fixed false-positive box from the 2026-09-14 bedroom evidence."""


def test_known_phantom_box_is_excluded_below_max_confidence():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    candidate = Candidate(box=PHANTOM_BOX, confidence=0.5, index=0)

    assert phantoms.select([candidate]) is None


def test_known_phantom_box_at_or_above_max_confidence_is_not_excluded():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    candidate = Candidate(box=PHANTOM_BOX, confidence=0.7, index=0)

    selected = phantoms.select([candidate])
    assert selected is candidate


def test_larger_person_box_overlapping_the_phantom_is_selected():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    # Much larger than the calibrated phantom box -- IoU well under 0.85 --
    # so it is not treated as "the same object" at all.
    person_box = (0.05, 0.20, 0.60, 0.55)
    candidates = [
        Candidate(box=PHANTOM_BOX, confidence=0.5, index=0),
        Candidate(box=person_box, confidence=0.4, index=1),
    ]

    selected = phantoms.select(candidates)
    assert selected is not None
    assert selected.index == 1


def test_no_known_phantoms_behaves_like_plain_highest_confidence():
    phantoms = KnownPhantoms([])
    candidates = [
        Candidate(box=PHANTOM_BOX, confidence=0.5, index=0),
        Candidate(box=(0.5, 0.4, 0.7, 0.9), confidence=0.3, index=1),
    ]

    selected = phantoms.select(candidates)
    assert selected is not None
    assert selected.index == 0


def test_all_candidates_excluded_returns_none():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    selected = phantoms.select([Candidate(box=PHANTOM_BOX, confidence=0.1, index=0)])
    assert selected is None


def test_load_missing_file_is_safe_and_inactive(tmp_path):
    phantoms = KnownPhantoms.load(str(tmp_path / "does-not-exist.yaml"))
    assert phantoms.boxes == []
    # Inactive: nothing gets excluded even at box-identical, near-zero confidence.
    candidate = Candidate(box=PHANTOM_BOX, confidence=0.01, index=0)
    assert phantoms.select([candidate]) is candidate


def test_load_none_path_is_safe_and_inactive():
    phantoms = KnownPhantoms.load(None)
    assert phantoms.boxes == []


def test_load_empty_path_is_safe_and_inactive():
    phantoms = KnownPhantoms.load("")
    assert phantoms.boxes == []


def test_load_malformed_yaml_is_safe_and_inactive(tmp_path):
    path = tmp_path / "phantoms.yaml"
    path.write_text("not: [valid, yaml: because: of: this")
    phantoms = KnownPhantoms.load(str(path))
    assert phantoms.boxes == []


def test_load_wrong_shape_yaml_is_safe_and_inactive(tmp_path):
    path = tmp_path / "phantoms.yaml"
    path.write_text("just_a_string")
    phantoms = KnownPhantoms.load(str(path))
    assert phantoms.boxes == []


def test_load_malformed_box_entry_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "phantoms.yaml"
    path.write_text(
        "phantoms:\n"
        "  - box: [0.1, 0.2, 0.3]\n"  # too short, malformed
        "  - box: [0.19, 0.31, 0.26, 0.46]\n"
    )
    phantoms = KnownPhantoms.load(str(path))
    assert phantoms.boxes == [(0.19, 0.31, 0.26, 0.46)]


def test_load_valid_file_round_trips_through_select(tmp_path):
    path = tmp_path / "phantoms.yaml"
    path.write_text("phantoms:\n  - box: [0.19, 0.31, 0.26, 0.46]\n")
    phantoms = KnownPhantoms.load(str(path), max_confidence=0.7)
    candidate = Candidate(box=PHANTOM_BOX, confidence=0.5, index=0)
    assert phantoms.select([candidate]) is None


# --- cluster_static_boxes (used by perceive.calibrate_phantoms) ---


def test_cluster_keeps_a_stable_box_seen_every_frame():
    boxes = [[PHANTOM_BOX] for _ in range(20)]
    clusters = cluster_static_boxes(boxes)
    assert len(clusters) == 1
    assert clusters[0] == PHANTOM_BOX


def test_cluster_drops_a_moving_box():
    # Center moves steadily; never settles, so no cluster passes the drift bar.
    boxes = [[(0.1 + i * 0.02, 0.3, 0.2 + i * 0.02, 0.5)] for i in range(20)]
    clusters = cluster_static_boxes(boxes)
    assert clusters == []


def test_cluster_drops_a_rarely_seen_box():
    # Static, but only in a fifth of the frames: below the 30% bar.
    boxes = [[PHANTOM_BOX]] * 4 + [[] for _ in range(16)]
    clusters = cluster_static_boxes(boxes)
    assert clusters == []


def test_cluster_keeps_frequent_box_ignoring_empty_frames():
    boxes = [[PHANTOM_BOX] if i % 2 == 0 else [] for i in range(20)]
    clusters = cluster_static_boxes(boxes)
    assert clusters == [PHANTOM_BOX]


def test_cluster_keeps_two_static_objects_seen_in_the_same_frames():
    other_box = (0.6, 0.6, 0.7, 0.8)
    boxes = [[PHANTOM_BOX, other_box] for _ in range(10)]
    clusters = cluster_static_boxes(boxes)
    assert len(clusters) == 2
    assert PHANTOM_BOX in clusters
    assert other_box in clusters


def test_cluster_keeps_a_static_box_seen_in_four_of_eleven_frames():
    boxes = [[PHANTOM_BOX] for _ in range(4)] + [[] for _ in range(7)]
    assert cluster_static_boxes(boxes) == [PHANTOM_BOX]


def test_cluster_of_no_frames_is_empty():
    assert cluster_static_boxes([]) == []
