"""Live bed calibration requested on the debug stream."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from queue import SimpleQueue
from threading import Thread

import redis
from nc_shared.bus import Bus
from nc_shared.events import BedZoneStatus, CalibrateBed

from perceive.zones import Polygon, ZoneMap, load_zones

DEBUG_STREAM = "debug"
DEBUG_GROUP = "perceive"
CALIBRATE_GROUP = "perceive-calibrate"
FRAME_COUNT = 20
TIMEOUT_SECONDS = 60.0
STATUS_INTERVAL_SECONDS = 30.0

logger = logging.getLogger("perceive")


def _log(message: str, **fields: object) -> None:
    logger.info(json.dumps({"service": "perceive", "message": message, **fields}))


def _thread_submit(job: Callable[[], None]) -> None:
    Thread(target=job, daemon=True).start()


def _drain(bus: Bus, stream: str) -> None:
    while messages := bus.read(stream, CALIBRATE_GROUP, "calibrate-1", count=50, block_ms=1):
        for msg_id, _event in messages:
            bus.ack(stream, CALIBRATE_GROUP, msg_id)


def read_live_frames(redis_url: str, *, count: int = FRAME_COUNT) -> list[bytes]:
    """Collect new raw frames; use gated frames if the raw source is absent."""
    client = redis.Redis.from_url(redis_url)
    try:
        bus = Bus(client)
        deadline = time.monotonic() + TIMEOUT_SECONDS
        stream = "frames_raw"
        bus.ensure_group("frames_raw", CALIBRATE_GROUP)
        _drain(bus, stream)
        jpegs: list[bytes] = []
        raw_deadline = time.monotonic() + 4.0
        while len(jpegs) < count and time.monotonic() < deadline:
            if not jpegs and stream == "frames_raw" and time.monotonic() >= raw_deadline:
                stream = "frames"
                bus.ensure_group("frames", CALIBRATE_GROUP)
                _drain(bus, stream)
            remaining_ms = max(1, min(1000, int((deadline - time.monotonic()) * 1000)))
            messages = bus.read(
                stream,
                CALIBRATE_GROUP,
                "calibrate-1",
                count=count - len(jpegs),
                block_ms=remaining_ms,
            )
            for msg_id, frame in messages:
                jpegs.append(frame.jpeg)
                bus.ack(stream, CALIBRATE_GROUP, msg_id)
        return jpegs
    finally:
        client.close()


@lru_cache(maxsize=1)
def _segmenter():
    from perceive.calibrate_bed import yolo_segmenter

    return yolo_segmenter()


def _calibrate(jpegs: list[bytes]) -> Polygon:
    from perceive.calibrate_bed import calibrate

    return calibrate(jpegs, _segmenter())


def _write(path: Path, polygon: Sequence[tuple[float, float]]) -> None:
    from perceive.calibrate_bed import write_bed_zone

    write_bed_zone(path, polygon)


class BedCalibration:
    """Owns request handling and status; workers only enqueue their outcome."""

    def __init__(
        self,
        bus,
        zones: ZoneMap,
        path: Path,
        frame_source: Callable[[], list[bytes]],
        *,
        calibrate: Callable[[list[bytes]], Polygon] = _calibrate,
        write: Callable[[Path, Polygon], None] = _write,
        submit: Callable[[Callable[[], None]], None] = _thread_submit,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bus = bus
        self.zones = zones
        self.path = path
        self.frame_source = frame_source
        self.calibrate = calibrate
        self.write = write
        self.submit = submit
        self.now = now
        self.monotonic = monotonic
        self.running = False
        self.group_ready = False
        self.last_status_at: float | None = None
        self.results: SimpleQueue[str | None] = SimpleQueue()

    def publish_status(self, calibration: str = "idle", detail: str | None = None) -> None:
        polygon = self.zones.polygons.get("bed")
        self.bus.publish(
            BedZoneStatus(
                source="perceive",
                has_bed=polygon is not None,
                polygon=polygon,
                calibration=calibration,
                detail=detail,
            ),
            maxlen=100,
        )
        self.last_status_at = self.monotonic()
        _log(
            "published BedZoneStatus",
            calibration=calibration,
            has_bed=polygon is not None,
            detail=detail,
        )

    def poll_requests(self, *, consumer: str = "perceive-1") -> None:
        if not self.group_ready:
            self.bus.ensure_group(DEBUG_STREAM, DEBUG_GROUP)
            self.group_ready = True
        for msg_id, event in self.bus.read(
            DEBUG_STREAM, DEBUG_GROUP, consumer, count=10, block_ms=1
        ):
            self.bus.ack(DEBUG_STREAM, DEBUG_GROUP, msg_id)
            if not isinstance(event, CalibrateBed):
                continue
            age = (self.now() - event.ts).total_seconds()
            if age > 60:
                _log("ignored stale CalibrateBed", age_seconds=round(age, 1))
            elif self.running:
                _log("ignored CalibrateBed while calibration is running")
            else:
                self.running = True
                self.publish_status("running")
                _log("started CalibrateBed")
                self.submit(self._job)

    def _job(self) -> None:
        try:
            jpegs = self.frame_source()
            if not jpegs:
                raise ValueError("no camera frames arrived")
            polygon = self.calibrate(jpegs)
            if not polygon:
                raise ValueError(f"no bed found in {len(jpegs)} frames")
            self.write(self.path, polygon)
            self.results.put(None)
        except Exception as exc:  # noqa: BLE001 - calibration must never stop frame processing
            self.results.put(str(exc) or type(exc).__name__)

    def take_result(self) -> ZoneMap:
        if self.results.empty():
            return self.zones
        detail = self.results.get_nowait()
        self.running = False
        if detail is None:
            self.zones = load_zones(self.path)
            self.publish_status("done")
        else:
            self.publish_status("failed", detail=detail[:200])
        return self.zones

    def maybe_publish_status(self) -> None:
        if (
            self.last_status_at is None
            or self.monotonic() - self.last_status_at >= STATUS_INTERVAL_SECONDS
        ):
            self.publish_status("running" if self.running else "idle")
