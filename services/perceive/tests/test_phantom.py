"""Tests for `perceive.phantom`: pure, no model weights, camera, or GPU.

Replaces an earlier runtime-co-occurrence design (2026-09-14): real bedroom
clips showed the phantom box and the real person never actually share a
frame, so that trigger never fired, and worse, it could permanently
misflag a genuinely still, sleeping person's own track if anything else in
frame moved. `KnownPhantoms` instead applies a short, pre-calibrated list of
boxes with no cross-frame memory at all.
"""

from perceive.phantom import (
    MAX_TRACK_GAP_CALLS,
    Box,
    Candidate,
    KnownPhantoms,
    cluster_static_boxes,
)

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


# --- track-continuity rescue (2026-09-15 bedroom evidence: a calibrated
# phantom scoring above `max_confidence` beats a real person mid-fall
# scoring lower, in a run of consecutive frames) ---

TRACKED_PERSON_BOX: Box = (0.53, 0.57, 0.73, 0.73)
"""A stand-in for the real person's box in the frame that established the
track, close to (but not identical to) the nearby candidate used below --
mirrors the small frame-to-frame drift measured on the real clip."""

NEARBY_PERSON_BOX: Box = (0.537, 0.572, 0.73, 0.728)
"""The next frame's real-person box: well within `TRACK_PROXIMITY_RADIUS`
(0.3) of `TRACKED_PERSON_BOX`'s centre, same as the ~0.10 drift measured on
the real clip."""

FAR_AWAY_BOX: Box = (0.05, 0.05, 0.15, 0.15)
"""A box nowhere near the track -- centre distance from `TRACKED_PERSON_BOX`
is well over `TRACK_PROXIMITY_RADIUS`, mirroring the ~0.49 distance measured
between the real clip's phantom and the actual tracked person."""


def test_confident_phantom_is_overridden_by_a_nearby_recent_track():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    # First call establishes a confirmed track on the real person.
    phantoms.select([Candidate(box=TRACKED_PERSON_BOX, confidence=0.9, index=0)])

    # Second call: the phantom box scores above `max_confidence` (so
    # `is_excluded` does not drop it and it would otherwise win the raw
    # argmax), but the real person is right where the track predicts, at
    # lower confidence.
    phantom_candidate = Candidate(box=PHANTOM_BOX, confidence=0.8, index=0)
    real_candidate = Candidate(box=NEARBY_PERSON_BOX, confidence=0.53, index=1)
    selected = phantoms.select([phantom_candidate, real_candidate])

    assert selected is real_candidate


def test_confident_phantom_wins_with_no_prior_track():
    # Same two candidates, but nothing has been confirmed yet: unchanged
    # "trust the strong score" behaviour.
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    phantom_candidate = Candidate(box=PHANTOM_BOX, confidence=0.8, index=0)
    real_candidate = Candidate(box=NEARBY_PERSON_BOX, confidence=0.53, index=1)

    selected = phantoms.select([phantom_candidate, real_candidate])

    assert selected is phantom_candidate


def test_confident_phantom_wins_when_only_other_candidate_is_far_from_track():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    phantoms.select([Candidate(box=TRACKED_PERSON_BOX, confidence=0.9, index=0)])

    phantom_candidate = Candidate(box=PHANTOM_BOX, confidence=0.8, index=0)
    far_candidate = Candidate(box=FAR_AWAY_BOX, confidence=0.53, index=1)
    selected = phantoms.select([phantom_candidate, far_candidate])

    assert selected is phantom_candidate


def test_top_pick_not_matching_a_phantom_is_unaffected_by_track_state():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    phantoms.select([Candidate(box=TRACKED_PERSON_BOX, confidence=0.9, index=0)])

    # The top pick this call sits nowhere near a known phantom box, so the
    # rescue never engages regardless of the recent track.
    plain_candidate = Candidate(box=(0.5, 0.4, 0.7, 0.9), confidence=0.6, index=0)
    other_candidate = Candidate(box=FAR_AWAY_BOX, confidence=0.3, index=1)
    selected = phantoms.select([plain_candidate, other_candidate])

    assert selected is plain_candidate


def test_confident_phantom_wins_once_the_track_has_gone_stale():
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    phantoms.select([Candidate(box=TRACKED_PERSON_BOX, confidence=0.9, index=0)])

    # Let the track age past `MAX_TRACK_GAP_CALLS` with frames that see
    # nothing at all, exactly like a person having left the frame.
    for _ in range(MAX_TRACK_GAP_CALLS + 1):
        assert phantoms.select([]) is None

    phantom_candidate = Candidate(box=PHANTOM_BOX, confidence=0.8, index=0)
    real_candidate = Candidate(box=NEARBY_PERSON_BOX, confidence=0.53, index=1)
    selected = phantoms.select([phantom_candidate, real_candidate])

    assert selected is phantom_candidate


def test_phantom_box_pick_never_seeds_a_track_for_a_later_rescue():
    # No prior real track: the phantom wins outright once, at high
    # confidence. That phantom-box pick must not itself become "the last
    # confirmed track" and start rescuing future phantom picks.
    phantoms = KnownPhantoms([PHANTOM_BOX], max_confidence=0.7)
    phantom_candidate = Candidate(box=PHANTOM_BOX, confidence=0.8, index=0)
    assert phantoms.select([phantom_candidate]) is phantom_candidate

    far_candidate = Candidate(box=FAR_AWAY_BOX, confidence=0.3, index=1)
    selected_again = phantoms.select([phantom_candidate, far_candidate])

    assert selected_again is phantom_candidate


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
