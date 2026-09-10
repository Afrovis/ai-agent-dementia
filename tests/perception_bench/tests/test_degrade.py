"""Tests for `perception_bench.degrade`. No `perceive` import needed."""

import random

import numpy as np
from PIL import Image

from perception_bench.degrade import apply_night_degradation


def _colour_image(width: int = 40, height: int = 30) -> Image.Image:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :, 0] = 200  # solid red
    return Image.fromarray(array, mode="RGB")


def test_apply_night_degradation_removes_colour_information():
    degraded = apply_night_degradation(_colour_image(), rng=random.Random(0))
    array = np.asarray(degraded).astype(np.int16)
    # Not exactly grey (independent per-channel noise means individual
    # pixels can differ a little across channels), but the *average*
    # channel spread should be small, unlike the solid-red input where it
    # was 200 everywhere.
    mean_channel_spread = np.mean(np.abs(array[:, :, 0] - array[:, :, 1]))
    assert mean_channel_spread < 15


def test_apply_night_degradation_darkens_corners_relative_to_centre():
    degraded = apply_night_degradation(
        _colour_image(width=100, height=80), noise_std=0.0, rng=random.Random(1)
    )
    array = np.asarray(degraded).astype(np.int16)
    centre = array[40, 50].mean()
    corner = array[0, 0].mean()
    assert corner < centre


def test_apply_night_degradation_does_not_mutate_input():
    original = _colour_image()
    original_bytes = original.tobytes()
    apply_night_degradation(original, rng=random.Random(0))
    assert original.tobytes() == original_bytes


def test_apply_night_degradation_is_reproducible_with_same_rng_seed():
    a = apply_night_degradation(_colour_image(), rng=random.Random(42))
    b = apply_night_degradation(_colour_image(), rng=random.Random(42))
    assert np.array_equal(np.asarray(a), np.asarray(b))
