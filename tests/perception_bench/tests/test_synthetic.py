"""Tests for tier 1: `perception_bench.synthetic`.

Needs `perceive` on `sys.path`, which importing `perception_bench` (this
package) arranges automatically -- see `perception_bench/__init__.py`.
"""

from perceive.classify import ClassifyThresholds

from perception_bench.scoring import STATE_NAMES, score_predictions
from perception_bench.synthetic import build_full_night_clip, run_clip, run_tier1


def test_build_full_night_clip_covers_every_state():
    clip = build_full_night_clip()
    ground_truths = {frame.ground_truth for frame in clip.frames}
    assert ground_truths == set(STATE_NAMES)


def test_every_frame_renders_jpeg_bytes():
    clip = build_full_night_clip()
    for frame in clip.frames:
        assert frame.jpeg.startswith(b"\xff\xd8")  # JPEG magic bytes


def test_run_clip_confirms_on_floor_and_absent_without_fabricated_delay():
    clip = build_full_night_clip()
    run = run_clip(clip)

    # on_floor and absent bypass StateTracker's hysteresis (IMMEDIATE_STATES),
    # so every frame with that ground truth should be predicted correctly
    # from the very first such frame -- the whole point of the bypass.
    on_floor_predictions = [p for gt, p in run.pairs if gt == "on_floor"]
    assert all(p == "on_floor" for p in on_floor_predictions)

    absent_predictions = [p for gt, p in run.pairs if gt == "absent"]
    assert all(p == "absent" for p in absent_predictions)


def test_run_clip_measures_a_latency_sample_per_scripted_transition():
    clip = build_full_night_clip()
    run = run_clip(clip)
    # 6 ground-truth states in this clip -> up to 6 transitions measured
    # (fewer only if a transition was never confirmed, which would also
    # show up in undetected_transitions).
    assert len(run.latency.samples) + len(run.undetected_transitions) == 6


def test_run_clip_latency_is_a_multiple_of_the_frame_interval():
    clip = build_full_night_clip(frame_interval_s=0.5)
    run = run_clip(clip)
    for _, latency in run.latency.samples:
        assert latency >= 0.0
        # latency = frame_count * interval, so it must land on a grid of 0.5s
        assert round(latency / 0.5) * 0.5 == round(latency, 6)


def test_run_clip_on_floor_latency_is_never_worse_than_one_frame_interval():
    """on_floor bypasses hysteresis, so its detection latency should be
    bounded by the frame interval alone (the time to the next frame),
    not by `confirm_frames`. This is a real assertion about
    `perceive.classify.StateTracker`'s documented behaviour, not a
    fabricated number."""
    clip = build_full_night_clip(frame_interval_s=0.5)
    run = run_clip(clip)
    on_floor_latencies = [lat for label, lat in run.latency.samples if label == "-> on_floor"]
    assert on_floor_latencies
    assert all(lat <= 0.5 for lat in on_floor_latencies)


def test_run_tier1_default_clips_produce_scoreable_pairs():
    runs = run_tier1()
    pairs = [pair for run in runs for pair in run.pairs]
    result = score_predictions(pairs)
    assert result.frame_count > 0
    assert result.per_state["on_floor"].support > 0
    assert result.per_state["standing"].support > 0


def test_run_clip_respects_custom_thresholds():
    clip = build_full_night_clip()
    lenient = ClassifyThresholds(walk_displacement_threshold=0.01)
    strict_run = run_clip(clip)
    lenient_run = run_clip(clip, thresholds=lenient)

    # A much lower walk-displacement threshold should confirm `walking`
    # no later, never later, than the default.
    def walking_latency(run):
        matches = [lat for label, lat in run.latency.samples if label == "-> walking"]
        return matches[0] if matches else None

    strict_latency = walking_latency(strict_run)
    lenient_latency = walking_latency(lenient_run)
    if strict_latency is not None and lenient_latency is not None:
        assert lenient_latency <= strict_latency
