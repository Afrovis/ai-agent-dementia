"""Tests for `capture.main`'s loop body, using `FakeBus` and a fake `FrameSource`.

No camera, no Redis, no Ollama: frames are tiny in-process JPEGs, and time
is threaded through explicitly rather than read from the wall clock, per
HANDOFF.md's testing convention.
"""

import io

from nc_shared.bus import FakeBus
from nc_shared.events import Frame, Health, RawFrame
from nc_shared.replay import CAPPED_MAXLEN
from PIL import Image

from capture.gate import MotionGate
from capture.main import CaptureConfig, build_gate, maybe_emit_health, run_once
from capture.sources import BrowserBusSource


def _jpeg(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (32, 32), color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeSource:
    """A `FrameSource` test double: yields a fixed queue of frames, then `None`."""

    def __init__(self, frames):
        self._frames = list(frames)

    def read(self):
        return self._frames.pop(0) if self._frames else None


def test_run_once_publishes_frame_when_source_has_one_and_gate_admits():
    bus = FakeBus()
    source = FakeSource([(_jpeg((10, 10, 10)), 32, 32, 1920, 1080, "browser")])
    gate = MotionGate()

    published = run_once(source, gate, bus, now_fn=lambda: 0.0)

    assert published is True
    entries = bus._streams["frames"]  # noqa: SLF001
    assert len(entries) == 1
    event = Frame.model_validate_json(entries[0].data)
    assert event.width == 32
    assert event.height == 32
    assert event.source_width == 1920
    assert event.source_height == 1080
    assert event.source_kind == "browser"


def test_run_once_returns_false_when_source_has_nothing():
    bus = FakeBus()
    source = FakeSource([])
    gate = MotionGate()

    assert run_once(source, gate, bus, now_fn=lambda: 0.0) is False
    assert bus._streams == {}  # noqa: SLF001


def test_run_once_drops_frame_the_gate_rejects():
    bus = FakeBus()
    still = _jpeg((10, 10, 10))
    source = FakeSource(
        [(still, 1, 1, 1920, 1080, "browser"), (still, 1, 1, 1920, 1080, "browser")]
    )
    gate = MotionGate(active_fps=2.0)
    clock = iter([0.0, 0.1])  # second frame arrives faster than the 0.5s period

    assert run_once(source, gate, bus, now_fn=lambda: next(clock)) is True
    assert run_once(source, gate, bus, now_fn=lambda: 0.1) is False
    assert len(bus._streams["frames"]) == 1  # noqa: SLF001


def test_build_gate_uses_config_values():
    config = CaptureConfig(active_fps=3.0, idle_fps=1.0, static_seconds=15.0, motion_threshold=0.1)
    gate = build_gate(config)
    assert gate.active_fps == 3.0
    assert gate.idle_fps == 1.0
    assert gate.static_seconds == 15.0
    assert gate.motion_threshold == 0.1


def test_capture_config_from_env_uses_defaults_when_unset():
    config = CaptureConfig.from_env(env={})
    assert config.source == "browser"
    assert config.device == "0"
    assert config.active_fps == 2.0
    assert config.idle_fps == 0.5
    assert config.static_seconds == 30.0
    assert config.motion_threshold == 0.02


def test_capture_config_from_env_reads_every_key():
    env = {
        "CAPTURE_SOURCE": "usb",
        "CAPTURE_DEVICE": "1",
        "CAPTURE_FPS": "5",
        "CAPTURE_IDLE_FPS": "1",
        "CAPTURE_STATIC_SECONDS": "10",
        "CAPTURE_MOTION_THRESHOLD": "0.05",
    }
    config = CaptureConfig.from_env(env=env)
    assert config.source == "usb"
    assert config.device == "1"
    assert config.active_fps == 5.0
    assert config.idle_fps == 1.0
    assert config.static_seconds == 10.0
    assert config.motion_threshold == 0.05


def test_maybe_emit_health_emits_on_first_call():
    bus = FakeBus()
    last = maybe_emit_health(bus, None, now=0.0)

    assert last == 0.0
    entries = bus._streams["health"]  # noqa: SLF001
    assert len(entries) == 1
    event = Health.model_validate_json(entries[0].data)
    assert event.service == "capture"
    assert event.ok is True


def test_maybe_emit_health_waits_for_the_interval():
    bus = FakeBus()
    last = maybe_emit_health(bus, None, now=0.0, interval=30.0)
    last = maybe_emit_health(bus, last, now=10.0, interval=30.0)

    assert last == 0.0
    assert len(bus._streams["health"]) == 1  # noqa: SLF001


def test_maybe_emit_health_emits_again_once_interval_elapses():
    bus = FakeBus()
    last = maybe_emit_health(bus, None, now=0.0, interval=30.0)
    last = maybe_emit_health(bus, last, now=31.0, interval=30.0)

    assert last == 31.0
    assert len(bus._streams["health"]) == 2  # noqa: SLF001


def _scene(shade: int, seed: int = 0) -> bytes:
    """A 320x240 JPEG; a non-zero `seed` slides a bright block across it."""
    image = Image.new("L", (320, 240), shade)
    if seed:
        image.paste(255, (seed % 200, 10, seed % 200 + 60, 90))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG")
    return buffer.getvalue()


def test_published_rate_follows_motion_end_to_end():
    """Drive the real source, gate, and publish path at the browser's 2 fps.

    This is issue #7's acceptance line ("2 fps normal, 0.5 fps when static
    for 30s") measured on the actual pipeline rather than on the gate alone:
    `RawFrame` events land on `frames_raw` exactly as the embodiment browser
    bridge publishes them, `BrowserBusSource` consumes them, and the frames
    the gate admits are counted off the `frames` stream.

    The static phase is measured in two parts on purpose. The first 30 s are
    still published at the active rate, because `static_seconds` has not
    elapsed yet; only afterwards does the gate settle to the idle heartbeat.
    Collapsing both into one number would hide which of the two is wrong.
    """
    bus = FakeBus()
    source = BrowserBusSource(bus)
    gate = MotionGate()
    clock = [1000.0]

    def run_seconds(seconds: float, *, moving: bool) -> int:
        published = 0
        for i in range(int(seconds * 2)):  # the browser bridge sends 2 fps
            bus.publish(
                RawFrame(
                    source="embodiment",
                    jpeg=_scene(40, seed=10 + i * 9 if moving else 0),
                    width=320,
                    height=240,
                    source_kind="browser",
                ),
                maxlen=CAPPED_MAXLEN["frames_raw"],
            )
            if run_once(source, gate, bus, now_fn=lambda: clock[0]):
                published += 1
            clock[0] += 0.5
        return published

    assert run_seconds(20, moving=True) == 40  # 2.0 fps while the scene moves
    assert run_seconds(30, moving=False) == 60  # still 2.0 fps during the countdown
    assert run_seconds(60, moving=False) == 30  # 0.5 fps once static for 30 s
    assert run_seconds(20, moving=True) == 40  # back to 2.0 fps on the next motion


def test_frames_stream_carries_capture_as_the_producer():
    """`capture` republishes under its own name, per HANDOFF.md section 5."""
    bus = FakeBus()
    bus.ensure_group("frames", "perceive")
    source = BrowserBusSource(bus)
    bus.publish(
        RawFrame(
            source="embodiment",
            jpeg=_scene(40),
            width=320,
            height=240,
            source_width=1920,
            source_height=1080,
            source_kind="browser",
        ),
        maxlen=CAPPED_MAXLEN["frames_raw"],
    )

    assert run_once(source, MotionGate(), bus, now_fn=lambda: 1000.0) is True

    ((_msg_id, event),) = bus.read("frames", "perceive", "consumer-1")
    assert isinstance(event, Frame)
    assert not isinstance(event, RawFrame)
    assert event.source == "capture"
    assert event.source_kind == "browser"
    assert event.source_width == 1920
    assert event.source_height == 1080
