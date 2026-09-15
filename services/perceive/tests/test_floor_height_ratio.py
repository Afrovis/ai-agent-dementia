"""Tests for the two floor-detection fixes added after
docs/FLOOR_DETECTION_HANDOFF.md sections 4 and 9: height relative to a
self-calibrated ground line (Fix 1), and the fall-drop cue (Fix 3). Hand-built
`PoseResult`s and injected ground lines throughout, no model weights, no
Redis, same as `test_classify.py` and `test_floor_and_dropout_rules.py`.
"""

import pytest

from perceive.backends import Landmark, PoseResult
from perceive.classify import (
    ClassifyThresholds,
    StateTracker,
    fit_ground_line,
    height_ratio_for,
    load_ground_line,
    save_ground_line,
)
from perceive.main import _parse_ground_line
from tests.test_classify import standing_pose

# A ground line close to the one docs/FLOOR_DETECTION_HANDOFF.md section 9's
# video-3 replay implies: expected height ~0.603 at ankle_y=0.77.
GROUND_LINE = (1.926, -0.880)


def _pose_with(top: float, *, ankle_y: float = 0.77, confidence: float = 0.9) -> PoseResult:
    """A `standing_pose()`-shaped body with its ankles pinned to `ankle_y`
    (the fixed ground row for these tests) and its bounding-box top swapped
    for `top`, the one number these tests need to move to walk a body from
    standing to falling to on the floor."""
    base = standing_pose()
    landmarks = dict(base.landmarks)
    for name in ("left_ankle", "right_ankle"):
        lm = landmarks[name]
        landmarks[name] = Landmark(x=lm.x, y=ankle_y, visibility=0.9)
    bbox = (base.bbox[0], top, base.bbox[2], base.bbox[3])
    return PoseResult(landmarks=landmarks, bbox=bbox, confidence=confidence)


def _pose_without_ankles() -> PoseResult:
    base = standing_pose()
    landmarks = {
        name: lm for name, lm in base.landmarks.items() if name not in ("left_ankle", "right_ankle")
    }
    return PoseResult(landmarks=landmarks, bbox=base.bbox, confidence=0.9)


def _pose_without_ankles_or_knees() -> PoseResult:
    base = standing_pose()
    landmarks = {
        name: lm
        for name, lm in base.landmarks.items()
        if name not in ("left_ankle", "right_ankle", "left_knee", "right_knee")
    }
    return PoseResult(landmarks=landmarks, bbox=base.bbox, confidence=0.9)


# --- height_ratio_for: pure helper --------------------------------------------------


def test_height_ratio_none_without_a_ground_line():
    assert height_ratio_for(_pose_with(top=0.45), None) is None


def test_height_ratio_uses_ankles():
    ratio = height_ratio_for(_pose_with(top=0.251), GROUND_LINE)
    assert ratio == pytest.approx(0.86, abs=0.01)


def test_height_ratio_falls_back_to_knees_when_no_ankle_is_visible():
    with_ankles = height_ratio_for(_pose_with(top=0.45), GROUND_LINE)
    without_ankles = height_ratio_for(_pose_without_ankles(), GROUND_LINE)
    # Both are computable (the knee fallback kicks in), and not identical to
    # "no ratio at all".
    assert with_ankles is not None
    assert without_ankles is not None


def test_height_ratio_none_with_no_ankle_or_knee_visible():
    assert height_ratio_for(_pose_without_ankles_or_knees(), GROUND_LINE) is None


def test_height_ratio_none_when_expected_height_is_too_small():
    # a=0, b=0.02: expected = 0.02 for every ground_y, below the 0.05 floor.
    assert height_ratio_for(_pose_with(top=0.45), (0.0, 0.02)) is None


def test_height_ratio_none_when_height_is_too_small():
    # ankle_y=0.77, box top=0.76: height = 0.01, at or below the 0.02 floor.
    assert height_ratio_for(_pose_with(top=0.76), GROUND_LINE) is None


def test_height_ratio_none_when_ground_point_is_above_the_box_top():
    # A mislabelled ankle placed above the box's own top: this is exactly
    # the shape of bad-keypoint detection that produced a -3.29 ratio.
    pose = _pose_with(top=0.90, ankle_y=0.77)
    assert height_ratio_for(pose, GROUND_LINE) is None


def test_height_ratio_none_when_ground_point_is_far_below_the_box_bottom():
    base = standing_pose()
    landmarks = dict(base.landmarks)
    for name in ("left_ankle", "right_ankle"):
        lm = landmarks[name]
        landmarks[name] = Landmark(x=lm.x, y=0.90, visibility=0.9)
    # box bottom well above the "ankle": 0.90 > 0.5 + 0.05.
    bbox = (base.bbox[0], 0.10, base.bbox[2], 0.50)
    pose = PoseResult(landmarks=landmarks, bbox=bbox, confidence=0.9)
    assert height_ratio_for(pose, GROUND_LINE) is None


def test_height_ratio_none_when_the_ratio_is_implausibly_high():
    # height=0.67 against an expected of 0.2: ratio 3.35, echoing the real
    # bad-keypoint reading of 2.5.
    pose = _pose_with(top=0.10, ankle_y=0.77)
    assert height_ratio_for(pose, (0.0, 0.2)) is None


def test_height_ratio_none_when_the_ratio_is_implausibly_low():
    # height=0.10 against an expected of 1.0: ratio 0.1, below the 0.15 floor.
    pose = _pose_with(top=0.67, ankle_y=0.77)
    assert height_ratio_for(pose, (0.0, 1.0)) is None


def test_height_ratio_accepts_a_plausible_ratio_at_the_boundaries():
    # Sanity check that the bounds do not reject ordinary values.
    assert height_ratio_for(_pose_with(top=0.529), GROUND_LINE) is not None


# --- fit_ground_line: Theil-Sen --------------------------------------------------


def test_fit_ground_line_abstains_below_min_samples():
    samples = [(0.5 + 0.01 * i, 0.4 + 0.01 * i) for i in range(19)]
    assert fit_ground_line(samples) is None


def test_fit_ground_line_recovers_a_known_line():
    # height = 2*ground_y + 1, sampled at 25 different ground_y values so
    # the spread check passes and the fit has something to bite on.
    samples = [(0.2 + 0.02 * i, 2.0 * (0.2 + 0.02 * i) + 1.0) for i in range(25)]
    fitted = fit_ground_line(samples)
    assert fitted is not None
    a, b = fitted
    assert a == pytest.approx(2.0, abs=1e-6)
    assert b == pytest.approx(1.0, abs=1e-6)


def test_fit_ground_line_degenerate_spread_falls_back_to_a_flat_line():
    # ground_y barely varies (spread well under the 0.05 floor); height
    # does, a little, so a slope fit would be noise.
    samples = [(0.60 + 0.0001 * i, 0.40 + 0.001 * i) for i in range(25)]
    fitted = fit_ground_line(samples)
    assert fitted is not None
    a, b = fitted
    assert a == 0.0
    assert b == pytest.approx(0.412, abs=0.01)


# --- PERCEIVE_GROUND_LINE parsing (perceive.main) ---------------------------------


def test_parse_ground_line_empty_is_none():
    assert _parse_ground_line("") is None
    assert _parse_ground_line("   ") is None


def test_parse_ground_line_parses_a_comma_pair():
    assert _parse_ground_line("1.926,-0.880") == (1.926, -0.880)


def test_parse_ground_line_ignores_garbage():
    assert _parse_ground_line("not-a-line") is None


# --- ground line persistence -------------------------------------------------------


def test_ground_line_persistence_round_trip(tmp_path):
    path = tmp_path / "ground_line.json"
    save_ground_line(str(path), (1.5, -0.7))
    assert load_ground_line(str(path)) == (1.5, -0.7)


def test_load_ground_line_missing_file_is_none(tmp_path):
    assert load_ground_line(str(tmp_path / "does-not-exist.json")) is None


def test_state_tracker_loads_a_persisted_ground_line_at_startup(tmp_path):
    path = tmp_path / "ground_line.json"
    save_ground_line(str(path), GROUND_LINE)

    tracker = StateTracker(ground_line_file=str(path), confirm_frames=1)
    # No online samples were ever fed in -- if the persisted line had not
    # loaded, the fall-drop rule below (which needs a ground line) could
    # never fire.
    tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0)
    result = tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)

    assert result is not None
    assert result[0] == "on_floor"


def test_explicit_ground_line_disables_online_fitting(tmp_path):
    """`ground_line` set means there is nothing left to learn -- confirmed
    standing/walking frames must not touch the persisted file at all."""
    path = tmp_path / "ground_line.json"
    tracker = StateTracker(ground_line=GROUND_LINE, ground_line_file=str(path), confirm_frames=1)

    for i in range(25):
        tracker.update(_pose_with(top=0.10 + 0.001 * i, confidence=0.9), "other", now=float(i))

    assert not path.exists()


# --- calibration poisoning: _collect_ground_sample rejections ----------------------


def test_online_calibration_ignores_standing_in_the_bed_zone():
    """Someone standing on the mattress is upright by the same geometry a
    person standing on the floor is, but their ankle is nowhere near the
    real floor row -- collecting it would teach the ground line the height
    of the bed."""
    tracker = StateTracker(confirm_frames=1)

    for i in range(25):
        tracker.update(
            _pose_with(top=0.10 + 0.02 * i, ankle_y=0.77, confidence=0.9), "bed", now=float(i)
        )

    assert tracker._fitted_ground_line is None


def test_online_calibration_ignores_an_ankle_outside_the_box():
    """A mislabelled ankle must not poison the fit any more than it should
    poison a single ratio reading (`_ground_point_in_box`)."""
    tracker = StateTracker(confirm_frames=1)

    base = standing_pose()
    landmarks = dict(base.landmarks)
    for name in ("left_ankle", "right_ankle"):
        lm = landmarks[name]
        # Far below the box the detector itself drew.
        landmarks[name] = Landmark(x=lm.x, y=0.99, visibility=0.9)
    bbox = (base.bbox[0], 0.10, base.bbox[2], 0.5)
    pose = PoseResult(landmarks=landmarks, bbox=bbox, confidence=0.9)

    for i in range(25):
        tracker.update(pose, "other", now=float(i))

    assert tracker._fitted_ground_line is None


def test_online_calibration_still_learns_from_valid_standing_frames():
    """Control: the same sequence with a plausible ankle, outside the bed
    zone, does calibrate -- the two rejections above are not simply
    disabling calibration outright."""
    tracker = StateTracker(confirm_frames=1)

    for i in range(25):
        # Spread ground_y enough to clear fit_ground_line's spread floor.
        tracker.update(
            _pose_with(top=0.10, ankle_y=0.70 + 0.01 * i, confidence=0.9), "other", now=float(i)
        )

    assert tracker._fitted_ground_line is not None


# --- Fix 1: two-consecutive-detection height-ratio rule ----------------------------


def test_two_consecutive_low_ratio_detections_confirm_on_floor():
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=5)

    first = tracker.update(_pose_with(top=0.529, confidence=0.9), "other", now=0.0)
    assert first is None

    second = tracker.update(_pose_with(top=0.529, confidence=0.9), "other", now=0.5)
    assert second is not None
    assert second[0] == "on_floor"


def test_a_single_low_ratio_detection_does_not_confirm_on_floor():
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=5)
    result = tracker.update(_pose_with(top=0.529, confidence=0.9), "other", now=0.0)
    assert result is None


def test_a_gap_over_three_seconds_resets_the_ratio_streak():
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=5)

    assert tracker.update(_pose_with(top=0.529, confidence=0.9), "other", now=0.0) is None
    # over 3s later: does not count as the second of two *consecutive* hits
    result = tracker.update(_pose_with(top=0.529, confidence=0.9), "other", now=4.0)
    assert result is None


def test_a_disagreeing_detection_resets_the_ratio_streak():
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=5)

    assert tracker.update(_pose_with(top=0.529, confidence=0.9), "other", now=0.0) is None
    # A confident, mid-ratio (0.68, above floor_height_ratio but below the
    # 0.8 fall-drop reference threshold) detection in between: disqualifies
    # for the streak without arming the unrelated fall-drop bypass.
    assert tracker.update(_pose_with(top=0.360, confidence=0.9), "other", now=0.5) is None
    result = tracker.update(_pose_with(top=0.529, confidence=0.9), "other", now=1.0)
    assert result is None


def test_low_ratio_in_the_bed_zone_never_confirms_on_floor():
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=5)

    first = tracker.update(_pose_with(top=0.529, confidence=0.9), "bed", now=0.0)
    second = tracker.update(_pose_with(top=0.529, confidence=0.9), "bed", now=0.5)

    assert first is None or first[0] != "on_floor"
    assert second is None or second[0] != "on_floor"


def test_low_confidence_detection_does_not_count_towards_the_streak():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(min_confidence=0.5, presence_confidence=0.25),
        ground_line=GROUND_LINE,
        confirm_frames=5,
    )

    assert tracker.update(_pose_with(top=0.529, confidence=0.3), "other", now=0.0) is None
    assert tracker.update(_pose_with(top=0.529, confidence=0.3), "other", now=0.5) is None


# --- Fix 3: fall drop -----------------------------------------------------------


def test_fall_drop_replay_of_the_video_3_sequence():
    """docs/FLOOR_DETECTION_HANDOFF.md section 9 / the task brief's replay:
    a standing detection, a fast collapse through mid-ratios, then a run of
    low-confidence detections near the floor. `on_floor` should be reported
    from the first detection whose height has genuinely dropped, and then
    held through the low-confidence tail instead of falling back to
    `absent`.
    """
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=3)

    # Standing, high ratio: the fall's reference point.
    assert tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0) is None
    # Mid-collapse: ratio still above floor_height_ratio, drop not yet big enough.
    assert tracker.update(_pose_with(top=0.360, confidence=0.9), "other", now=0.5) is None

    # The box top has now risen 0.20 since the reference, well past the
    # 0.15 fall_drop default, within the 2.5s fall_window_seconds default:
    # a single confident, low-ratio detection is enough.
    dropped = tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)
    assert dropped is not None
    assert dropped[0] == "on_floor"

    # The fall's reference has now expired (>2.5s old), so these two
    # low-confidence frames rely on the existing "seen but not readable"
    # hold, not the fall-drop bypass -- either way `on_floor` must survive.
    assert tracker.update(_pose_with(top=0.613, confidence=0.38), "other", now=4.0) is None
    assert tracker.update(_pose_with(top=0.613, confidence=0.29), "other", now=5.5) is None
    assert tracker.snapshot()[0] == "on_floor"


def test_fall_drop_does_not_fire_for_a_slow_controlled_sit_down():
    """The same 0.15 box-top change as a fall, spread over 10s instead of a
    couple of seconds, must never trigger the fall-drop bypass -- the
    reference upright frame is long stale (`fall_window_seconds=2.5`) by
    the time the change has accumulated."""
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=10)

    tops = [0.20 + 0.015 * i for i in range(11)]  # 0.20 -> 0.35 over 10s
    results = [
        tracker.update(_pose_with(top=top, confidence=0.9), "other", now=float(i))
        for i, top in enumerate(tops)
    ]

    assert all(r is None or r[0] != "on_floor" for r in results)


def test_fall_drop_never_fires_in_the_bed_zone():
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=3)

    tracker.update(_pose_with(top=0.251, confidence=0.9), "bed", now=0.0)
    result = tracker.update(_pose_with(top=0.450, confidence=0.79), "bed", now=1.5)

    assert result is None or result[0] != "on_floor"


def test_fall_drop_anchor_requires_confidence():
    """A shaky, low-confidence detection must not get to claim "this is
    what upright looks like here" -- review found ratios as wild as -3.29
    and 2.5 coming from bad keypoints, mostly on low-confidence frames."""
    tracker = StateTracker(ground_line=GROUND_LINE, confirm_frames=3)

    # Same reference frame as the video-3 replay, but too shaky to anchor on.
    tracker.update(_pose_with(top=0.251, confidence=0.4), "other", now=0.0)
    result = tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)

    assert result is None or result[0] != "on_floor"


def test_floor_suspect_does_not_arm_after_bending_over_then_leaving_through_the_door():
    """A drop was genuinely observed (someone bending down), but the last
    detection before they vanished was in the door zone, on the way out --
    not evidence of a person left on the floor."""
    tracker = StateTracker(
        thresholds=ClassifyThresholds(hold_floor=False),
        ground_line=GROUND_LINE,
        confirm_frames=1,
    )

    # Upright reference, confident, in the room.
    tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0)
    # A confident, genuine drop (top risen 0.15+ since the reference) --
    # but the ratio stays just above floor_height_ratio, so this alone
    # does not report on_floor; and it happens as they cross into the door.
    dropped = tracker.update(_pose_with(top=0.402, confidence=0.79), "door", now=0.5)
    assert dropped is None or dropped[0] != "on_floor"

    # They are then lost entirely, last seen in the door zone.
    result = tracker.update(None, "other", now=1.0)

    assert tracker.floor_suspect is None
    assert result == ("absent", 0.0)


def test_floor_suspect_does_not_arm_walking_away_from_the_camera():
    """Walking away: the box shrinks and the top drifts, slowly, but the
    ratio stays near upright throughout -- no drop is ever observed, so
    there is nothing to arm `floor_suspect` from when they leave frame."""
    tracker = StateTracker(
        thresholds=ClassifyThresholds(hold_floor=False),
        ground_line=GROUND_LINE,
        confirm_frames=1,
    )

    tops = [0.25 + 0.002 * i for i in range(11)]  # 0.25 -> 0.27, ratio stays ~0.8+
    for i, top in enumerate(tops):
        tracker.update(_pose_with(top=top, confidence=0.9), "other", now=float(i))

    result = tracker.update(None, "other", now=11.0)

    assert tracker.floor_suspect is None
    assert result == ("absent", 0.0)


def test_floor_suspect_set_on_loss_after_a_drop_and_cleared_on_upright_redetection():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(hold_floor=False),
        ground_line=GROUND_LINE,
        confirm_frames=1,
    )

    tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0)
    dropped = tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)
    assert dropped is not None
    assert dropped[0] == "on_floor"

    assert tracker.floor_suspect is None

    # The person is lost entirely, right after the drop.
    assert tracker.update(None, "other", now=2.5) is None
    assert tracker.floor_suspect == 2.5

    # No `absent` while suspect, however long the gap.
    assert tracker.update(None, "other", now=10.0) is None
    assert tracker.floor_suspect == 2.5  # unchanged, not re-armed

    # A clear, upright redetection resolves it.
    tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=11.0)
    assert tracker.floor_suspect is None


def test_floor_suspect_clears_on_a_detection_in_the_bed_zone():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(hold_floor=False),
        ground_line=GROUND_LINE,
        confirm_frames=1,
    )
    tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0)
    tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)
    tracker.update(None, "other", now=2.5)
    assert tracker.floor_suspect == 2.5

    tracker.update(_pose_with(top=0.60, confidence=0.9), "bed", now=3.0)
    assert tracker.floor_suspect is None


def test_floor_suspect_expires_after_floor_suspect_seconds():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(hold_floor=False, floor_suspect_seconds=5.0),
        ground_line=GROUND_LINE,
        confirm_frames=1,
    )
    tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0)
    tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)
    assert tracker.update(None, "other", now=2.5) is None
    assert tracker.floor_suspect == 2.5

    # Well past floor_suspect_seconds: the suspicion lapses and `absent`
    # resumes on the very next empty frame (hold_floor is disabled here).
    result = tracker.update(None, "other", now=9.0)
    assert tracker.floor_suspect is None
    assert result == ("absent", 0.0)


def test_floor_suspect_never_reports_on_floor_by_itself():
    tracker = StateTracker(
        thresholds=ClassifyThresholds(hold_floor=False),
        ground_line=GROUND_LINE,
        confirm_frames=1,
    )
    tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0)
    tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)

    for i, now in enumerate((2.5, 4.0, 6.0, 8.0)):
        result = tracker.update(None, "other", now=now)
        assert result is None, f"frame {i} at t={now} must hold, not publish anything"


# --- no ground line: everything above is inert --------------------------------------


def test_without_a_ground_line_the_new_rules_never_fire():
    tracker = StateTracker(confirm_frames=1)  # no ground_line, no ground_line_file

    assert tracker.update(_pose_with(top=0.251, confidence=0.9), "other", now=0.0) is not None
    result = tracker.update(_pose_with(top=0.450, confidence=0.79), "other", now=1.5)

    # Neither the two-consecutive ratio rule nor the fall-drop bypass can
    # apply -- height_ratio_for always returns None without a ground line.
    assert result is None or result[0] != "on_floor"
    assert tracker.floor_suspect is None
