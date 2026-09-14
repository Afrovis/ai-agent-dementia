"""Pluggable frame sources for `capture`.

Issue #7: "Sources: browser webcam via the media bridge (MVP), USB camera,
RTSP camera with IR night vision (v1)." Every source implements the same
tiny interface -- `read() -> (jpeg, width, height, source_width,
source_height, source_kind) | None` --
so `capture.main.run_once` does not care which one it is talking to; the
motion gate and publish step downstream are identical either way.

`BrowserBusSource` needs only `nc_shared.bus`, so it is exercised in tests
with `FakeBus` and no camera. `OpenCvSource` needs the optional `opencv`
extra (`pip install .[camera]`); importing this module must not fail when
that extra is absent, since the browser MVP path is the default and must
keep working with no OpenCV installed at all -- so `cv2` is imported lazily,
inside `OpenCvSource.__init__`, not at module scope.
"""

from __future__ import annotations

from typing import Literal, Protocol

RAW_FRAME_STREAM = "frames_raw"
RAW_FRAME_GROUP = "capture"

FrameTuple = tuple[bytes, int, int, int, int, str]
"""Encoded bytes/dimensions, intrinsic source dimensions, and source kind."""


class FrameSource(Protocol):
    """The interface `capture.main` drives: one frame at a time, or none ready."""

    def read(self) -> FrameTuple | None:
        """Return the next available frame, or `None` if none is ready right now."""
        ...


class BrowserBusSource:
    """Reads `RawFrame` events published by the `embodiment` browser bridge.

    Consumes the `frames_raw` stream under the `capture` consumer group, so
    delivery is at-least-once and a restart picks up where it left off
    rather than replaying the whole backlog. Each message is acked as soon
    as it has been decoded into a `FrameTuple` -- at that point `capture`
    fully owns the frame (nothing else reads `frames_raw`), so there is
    nothing left to redeliver it for.
    """

    def __init__(
        self,
        bus,
        *,
        consumer: str = "capture-1",
        count: int = 10,
        block_ms: int = 1000,
    ) -> None:
        self._bus = bus
        self._consumer = consumer
        self._count = count
        self._block_ms = block_ms
        self._buffer: list[tuple[str, object]] = []
        self._bus.ensure_group(RAW_FRAME_STREAM, RAW_FRAME_GROUP)

    def read(self) -> FrameTuple | None:
        """Return the next buffered `RawFrame`, fetching a fresh batch if needed."""
        if not self._buffer:
            self._buffer = self._bus.read(
                RAW_FRAME_STREAM,
                RAW_FRAME_GROUP,
                self._consumer,
                count=self._count,
                block_ms=self._block_ms,
            )
        if not self._buffer:
            return None
        msg_id, event = self._buffer.pop(0)
        self._bus.ack(RAW_FRAME_STREAM, RAW_FRAME_GROUP, msg_id)
        return (
            event.jpeg,
            event.width,
            event.height,
            event.source_width or event.width,
            event.source_height or event.height,
            event.source_kind,
        )


class OpenCvSource:
    """Reads frames from a USB camera index or an RTSP URL via OpenCV.

    `device` is a USB camera index passed as a string (e.g. `"0"`) when
    `source_kind="usb"`, or an RTSP URL when `source_kind="rtsp"`. Frames
    are JPEG-encoded here so the rest of `capture` (the motion gate, the
    `Frame` event) only ever deals with JPEG bytes, matching every other
    frame source.

    `opencv-python` is an optional dependency (the `camera` extra in
    `pyproject.toml`): the browser MVP path never needs it, so importing
    this class must not require it to be installed. The import is deferred
    to `__init__`, and raises a clear `RuntimeError` (not `ImportError`
    straight from `cv2`) telling the operator what to install, only when a
    camera source is actually selected.
    """

    def __init__(self, device: str, source_kind: Literal["usb", "rtsp"]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for the 'usb'/'rtsp' capture sources. "
                "Install it with `pip install .[camera]`."
            ) from exc

        self._cv2 = cv2
        self._source_kind = source_kind
        capture_target: int | str = int(device) if source_kind == "usb" else device
        self._capture = cv2.VideoCapture(capture_target)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open {source_kind} capture device {device!r}")

    def read(self) -> FrameTuple | None:
        """Grab and JPEG-encode one frame, or `None` if the device gave nothing."""
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        height, width = frame.shape[:2]
        ok, encoded = self._cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return (encoded.tobytes(), width, height, width, height, self._source_kind)

    def close(self) -> None:
        """Release the underlying `cv2.VideoCapture`."""
        self._capture.release()


def build_source(kind: str, device: str, bus) -> FrameSource:
    """Return the configured `FrameSource` for `kind` (`browser`, `usb`, `rtsp`).

    `bus` is only used by `browser`; `device` is only used by `usb`/`rtsp`.
    Raises `ValueError` for an unrecognised `kind`, which `capture.main`
    treats as a configuration error worth failing loudly on at startup
    rather than silently idling (HANDOFF.md rule 4).
    """
    if kind == "browser":
        return BrowserBusSource(bus)
    if kind in ("usb", "rtsp"):
        return OpenCvSource(device, kind)
    raise ValueError(f"unknown CAPTURE_SOURCE: {kind!r}")
