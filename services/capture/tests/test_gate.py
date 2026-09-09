"""Tests for `capture.gate.MotionGate`, driven with synthetic time, no sleeping."""

import io

from PIL import Image

from capture.gate import MotionGate


def _jpeg(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (64, 64), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


STILL = _jpeg((20, 20, 20))
MOVED = _jpeg((220, 220, 220))


def test_first_frame_is_always_admitted():
    gate = MotionGate()
    assert gate.admit(STILL, now=0.0) is True


def test_frame_faster_than_active_period_is_dropped():
    gate = MotionGate(active_fps=2.0)
    assert gate.admit(STILL, now=0.0) is True
    # 0.2s later is faster than the 0.5s active period (1/2fps).
    assert gate.admit(MOVED, now=0.2) is False


def test_moving_frame_admitted_once_the_active_period_elapses():
    gate = MotionGate(active_fps=2.0, motion_threshold=0.02)
    assert gate.admit(STILL, now=0.0) is True
    assert gate.admit(MOVED, now=0.5) is True
    assert gate.current_fps == 2.0


def test_drops_to_idle_fps_after_static_seconds_with_no_motion():
    gate = MotionGate(active_fps=2.0, idle_fps=0.5, static_seconds=30.0)
    assert gate.admit(STILL, now=0.0) is True
    # Keep publishing the identical frame at the active rate; no motion ever.
    now = 0.5
    while now < 30.0:
        gate.admit(STILL, now=now)
        now += 0.5

    # Now well past static_seconds with no motion: idle_fps (0.5) applies,
    # so a frame just 0.5s after the last publish should still be dropped.
    assert gate.admit(STILL, now=now) is False
    assert gate.current_fps == 0.5

    # But one a full idle period (2s) later is admitted, as a heartbeat.
    admitted = gate.admit(STILL, now=now + 2.0)
    assert admitted is True


def test_motion_after_static_period_returns_to_active_fps():
    gate = MotionGate(active_fps=2.0, idle_fps=0.5, static_seconds=1.0)
    assert gate.admit(STILL, now=0.0) is True
    # Go static past static_seconds.
    gate.admit(STILL, now=2.0)
    assert gate.current_fps == 0.5

    # A new admit with motion should notice it and go back to active_fps.
    admitted = gate.admit(MOVED, now=4.0)
    assert admitted is True
    assert gate.current_fps == 2.0
