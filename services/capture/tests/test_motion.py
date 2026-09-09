"""Tests for `capture.motion`: pure functions, tiny in-process Pillow fixtures."""

import io

from PIL import Image

from capture.motion import SIGNATURE_SIZE, frame_signature, motion_score


def _jpeg(color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> bytes:
    """Build a tiny solid-color JPEG in memory -- no binary fixtures on disk."""
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_frame_signature_has_the_expected_length():
    signature = frame_signature(_jpeg((10, 10, 10)))
    assert len(signature) == SIGNATURE_SIZE * SIGNATURE_SIZE


def test_frame_signature_returns_empty_tuple_for_garbage_bytes():
    assert frame_signature(b"not a jpeg at all") == ()


def test_motion_score_is_near_zero_for_identical_frames():
    jpeg = _jpeg((128, 64, 200))
    a = frame_signature(jpeg)
    b = frame_signature(jpeg)
    assert motion_score(a, b) < 0.01


def test_motion_score_is_high_for_very_different_frames():
    black = frame_signature(_jpeg((0, 0, 0)))
    white = frame_signature(_jpeg((255, 255, 255)))
    assert motion_score(black, white) > 0.9


def test_motion_score_is_between_zero_and_one():
    dim = frame_signature(_jpeg((10, 10, 10)))
    bright = frame_signature(_jpeg((60, 60, 60)))
    score = motion_score(dim, bright)
    assert 0.0 <= score <= 1.0


def test_motion_score_treats_empty_signature_as_maximum_motion():
    signature = frame_signature(_jpeg((1, 2, 3)))
    assert motion_score((), signature) == 1.0
    assert motion_score(signature, ()) == 1.0
    assert motion_score((), ()) == 1.0


def test_motion_score_treats_mismatched_lengths_as_maximum_motion():
    assert motion_score((1, 2, 3), (1, 2)) == 1.0
