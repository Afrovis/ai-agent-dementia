from datetime import UTC, datetime

from PIL import Image

from video_eval.common import read_jsonl
from video_eval.replay import build_raw_replay, compare_e2e


def test_build_raw_replay_uses_browser_frames_and_half_second_timestamps(tmp_path):
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    frames = []
    for index in range(2):
        path = bridge / f"f_{index}.jpg"
        Image.new("RGB", (320, 240), "black").save(path)
        frames.append(
            {"frame_index": index, "t_s": index * 0.5, "bridge_path": f"bridge/{path.name}"}
        )
    output = tmp_path / "raw.jsonl"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        build_raw_replay(
            frames,
            tmp_path,
            output,
            variant="squash",
            start=start,
            source_width=1920,
            source_height=1080,
        )
        == 2
    )
    rows = read_jsonl(output)
    assert rows[0]["stream"] == "frames_raw"
    assert rows[0]["payload"]["source_kind"] == "browser"
    assert rows[0]["payload"]["source_width"] == 1920
    assert rows[0]["payload"]["source_height"] == 1080
    assert rows[1]["ts"] == "2026-01-01T00:00:00.500000+00:00"


def test_compare_e2e_checks_perception_and_session_timing():
    start = datetime(2026, 1, 1, tzinfo=UTC)

    def line(seconds, event_type, payload):
        return {
            "stream": "person" if event_type == "PersonState" else "session",
            "event_type": event_type,
            "ts": start.fromtimestamp(start.timestamp() + seconds, UTC).isoformat(),
            "payload": {"source": "test", "ts": start.isoformat(), **payload},
        }

    bus = [
        line(10, "PersonState", {"state": "sitting_up", "zone": "bed"}),
        line(10.5, "SessionState", {"phase": "OBSERVING", "goal": "return_to_bed"}),
        line(30, "SessionState", {"phase": "ENGAGED", "goal": "return_to_bed"}),
    ]
    reference = {
        "timeline": [
            {"from_s": 0, "to_s": 10, "state": "in_bed", "zone": "bed"},
            {"from_s": 10, "to_s": 40, "state": "sitting_up", "zone": "bed"},
        ]
    }
    predictions = [{"published": True, "state": "sitting_up", "zone": "bed"}]
    result = compare_e2e(bus, reference, predictions, start=start)
    assert result["passed"] is True
    assert result["checks"]["restroom_goal_within_5s"] is None


def test_short_absence_does_not_allow_unexpected_escalation():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bus = [
        {
            "stream": "session",
            "event_type": "SessionState",
            "ts": datetime.fromtimestamp(start.timestamp() + 2, UTC).isoformat(),
            "payload": {"phase": "ESCALATED", "goal": "wait_for_caregiver"},
        }
    ]
    reference = {"timeline": [{"from_s": 0, "to_s": 5, "state": "absent", "zone": "other"}]}
    result = compare_e2e(bus, reference, [], start=start, absent_limit_seconds=10)
    assert result["checks"]["no_escalation_without_reference_risk"] is False
