from datetime import UTC, datetime, timedelta

from nc_shared.bus import FakeBus
from nc_shared.events import BedZoneStatus, CalibrateBed, DebugControl

from perceive.bed_debug import BedCalibration
from perceive.calibrate_bed import write_bed_zone
from perceive.zones import load_zones

NOW = datetime(2026, 9, 22, tzinfo=UTC)
POLYGON = [(0.1, 0.2), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)]


def statuses(bus):
    return [
        event
        for _, event in bus.read("debug", "test-status", "test", count=20)
        if isinstance(event, BedZoneStatus)
    ]


def controller(tmp_path, *, frames=None, calibrate=None, submit=lambda job: job(), monotonic=None):
    bus = FakeBus()
    path = tmp_path / "zones.yaml"
    job = BedCalibration(
        bus,
        load_zones(path),
        path,
        frame_source=lambda: [b"jpeg"] * 20 if frames is None else frames,
        calibrate=calibrate or (lambda _jpegs: POLYGON),
        write=write_bed_zone,
        submit=submit,
        now=lambda: NOW,
        monotonic=monotonic or (lambda: 0.0),
    )
    return bus, job, path


def test_request_publishes_running_then_done_and_reloads_zones(tmp_path):
    bus, job, path = controller(tmp_path)
    job.publish_status()
    bus.publish(CalibrateBed(source="embodiment", ts=NOW), maxlen=100)
    job.poll_requests()
    assert [s.calibration for s in statuses(bus)] == ["idle", "running"]
    assert job.take_result().polygons["bed"] == POLYGON
    assert load_zones(path).polygons["bed"] == POLYGON
    assert [(s.calibration, s.has_bed, s.polygon) for s in statuses(bus)] == [
        ("done", True, POLYGON)
    ]


def test_failure_keeps_existing_zone(tmp_path):
    bus, job, path = controller(tmp_path, calibrate=lambda _jpegs: [])
    write_bed_zone(path, POLYGON)
    job.zones = load_zones(path)
    bus.publish(CalibrateBed(source="embodiment", ts=NOW))
    job.poll_requests()
    assert job.take_result().polygons["bed"] == POLYGON
    status = statuses(bus)[-1]
    assert status.calibration == "failed"
    assert status.detail == "no bed found in 20 frames"
    assert status.polygon == POLYGON


def test_no_frames_reports_failure(tmp_path):
    bus, job, _path = controller(tmp_path, frames=[])
    bus.publish(CalibrateBed(source="embodiment", ts=NOW))
    job.poll_requests()
    job.take_result()
    assert statuses(bus)[-1].detail == "no camera frames arrived"


def test_stale_and_other_debug_events_are_acked_and_ignored(tmp_path):
    bus, job, _path = controller(tmp_path)
    bus.publish(CalibrateBed(source="embodiment", ts=NOW - timedelta(seconds=61)))
    bus.publish(DebugControl(source="agent"))
    job.poll_requests()
    assert not job.running
    assert bus.pending("debug", "perceive") == []
    assert statuses(bus) == []


def test_second_request_while_running_is_ignored(tmp_path):
    pending = []
    bus, job, _path = controller(tmp_path, submit=pending.append)
    bus.publish(CalibrateBed(source="embodiment", ts=NOW))
    bus.publish(CalibrateBed(source="embodiment", ts=NOW))
    job.poll_requests()
    assert len(pending) == 1
    assert [s.calibration for s in statuses(bus)] == ["running"]
    pending[0]()
    job.take_result()
    assert statuses(bus)[-1].calibration == "done"


def test_startup_status_reflects_existing_bed(tmp_path):
    bus, job, path = controller(tmp_path)
    write_bed_zone(path, POLYGON)
    job.zones = load_zones(path)
    job.publish_status()
    status = statuses(bus)[0]
    assert status.calibration == "idle"
    assert status.has_bed is True
    assert status.polygon == POLYGON


def test_status_is_republished_after_thirty_seconds(tmp_path):
    current = [0.0]
    bus, job, _path = controller(tmp_path, monotonic=lambda: current[0])
    job.publish_status()
    current[0] = 29.0
    job.maybe_publish_status()
    assert len(statuses(bus)) == 1
    current[0] = 30.0
    job.maybe_publish_status()
    assert [s.calibration for s in statuses(bus)] == ["idle"]


def test_example_fallback_bed_is_not_a_configured_bed(tmp_path):
    bus, job, path = controller(tmp_path)
    write_bed_zone(tmp_path / "zones.example.yaml", POLYGON)
    job.zones = load_zones(path)
    assert job.zones.polygons["bed"] == POLYGON
    job.publish_status()
    status = statuses(bus)[0]
    assert status.has_bed is False
    assert status.polygon == POLYGON
    assert "placeholder" in status.detail
