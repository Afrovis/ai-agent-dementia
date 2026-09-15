import json

import yaml

from video_eval.common import write_jsonl
from video_eval.score import _gate_verdicts, event_metrics, score_clip


def test_event_metrics_reports_delay_miss_and_false_transition():
    timeline = [
        {"from_s": 0, "to_s": 2, "state": "in_bed", "zone": "bed"},
        {"from_s": 2, "to_s": 4, "state": "standing", "zone": "other"},
        {"from_s": 4, "to_s": 6, "state": "on_floor", "zone": "other"},
    ]
    predictions = [
        {"t_s": 0, "state": "in_bed", "zone": "bed"},
        {"t_s": 2.5, "state": "walking", "zone": "other"},
        {"t_s": 5.0, "state": "on_floor", "zone": "other"},
        {"t_s": 5.5, "state": "sitting_up", "zone": "other"},
    ]
    result = event_metrics(predictions, timeline)
    assert [row["delay_s"] for row in result["delays"]] == [0.5, 1.0]
    assert result["false_transition_count"] == 1


def test_event_metrics_accepts_early_onset_within_tolerance():
    timeline = [{"from_s": 10, "to_s": 20, "state": "on_floor", "zone": "other"}]
    predictions = [
        {"t_s": 8.5, "state": "standing", "zone": "other"},
        {"t_s": 9.0, "state": "on_floor", "zone": "other"},
        {"t_s": 10.0, "state": "on_floor", "zone": "other"},
    ]

    event = event_metrics(predictions, timeline)["delays"][0]

    assert event["missed"] is False
    assert event["delay_s"] == 0.0
    assert event["onset_offset_s"] == -1.0


def test_event_metrics_does_not_accept_early_run_older_than_tolerance():
    timeline = [{"from_s": 10, "to_s": 20, "state": "on_floor", "zone": "other"}]
    predictions = [
        {"t_s": 6.5, "state": "on_floor", "zone": "other"},
        {"t_s": 10.0, "state": "on_floor", "zone": "other"},
        {"t_s": 11.0, "state": "standing", "zone": "other"},
        {"t_s": 12.0, "state": "on_floor", "zone": "other"},
    ]

    event = event_metrics(predictions, timeline)["delays"][0]

    assert event["missed"] is False
    assert event["delay_s"] == 2.0
    assert event["onset_offset_s"] == 2.0


def test_event_metrics_keeps_post_label_onset_delay():
    timeline = [{"from_s": 10, "to_s": 20, "state": "on_floor", "zone": "other"}]
    predictions = [
        {"t_s": 10.0, "state": "standing", "zone": "other"},
        {"t_s": 11.5, "state": "on_floor", "zone": "other"},
    ]

    event = event_metrics(predictions, timeline)["delays"][0]

    assert event["delay_s"] == 1.5
    assert event["onset_offset_s"] == 1.5


def test_event_metrics_does_not_match_later_fall_after_early_first_fall():
    timeline = [
        {"from_s": 0, "to_s": 120, "state": "standing", "zone": "other"},
        {"from_s": 120, "to_s": 125, "state": "on_floor", "zone": "other"},
        {"from_s": 125, "to_s": 135, "state": "standing", "zone": "other"},
        {"from_s": 135, "to_s": 140, "state": "on_floor", "zone": "other"},
    ]
    predictions = [
        {"t_s": 0, "state": "standing", "zone": "other"},
        {"t_s": 119, "state": "on_floor", "zone": "other"},
        {"t_s": 125, "state": "standing", "zone": "other"},
        {"t_s": 135, "state": "on_floor", "zone": "other"},
    ]

    floor_events = [
        event
        for event in event_metrics(predictions, timeline)["delays"]
        if event["value"] == "on_floor"
    ]

    assert [event["delay_s"] for event in floor_events] == [0.0, 0.0]
    assert [event["onset_offset_s"] for event in floor_events] == [-1.0, 0.0]
    frames = {
        "all": {
            "exact": {
                "per_state": {
                    "standing": {"recall": 1.0},
                    "on_floor": {"recall": 1.0},
                }
            }
        }
    }
    gate = _gate_verdicts(frames, {"delays": floor_events})["on_floor_delay"]
    assert gate["value_s"] == 0.0


def test_missed_floor_transition_fails_measured_latency_gate():
    frames = {
        "all": {
            "exact": {
                "per_state": {
                    "standing": {"recall": None},
                    "on_floor": {"recall": 0.0},
                }
            }
        }
    }
    events = {"delays": [{"kind": "state", "value": "on_floor", "delay_s": None, "missed": True}]}
    gate = _gate_verdicts(frames, events)["on_floor_delay"]
    assert gate["measured"] is True
    assert gate["met"] is False
    assert gate["missed"] == 1


def test_score_requires_confirmed_reference_and_writes_both_reports(tmp_path):
    clip = tmp_path / "clips" / "clip"
    frames = [{"frame_index": index, "t_s": index * 0.5} for index in range(8)]
    predictions = []
    for index in range(8):
        state = "standing" if index < 4 else "on_floor"
        predictions.append(
            {
                "frame_index": index,
                "t_s": index * 0.5,
                "state": state,
                "zone": "other",
                "gated": index == 0,
                "detected": None if index == 0 else True,
                "published": index in {0, 4},
            }
        )
    write_jsonl(clip / "frames.jsonl", frames)
    write_jsonl(clip / "predictions" / "model-tag.jsonl", predictions)
    reference = {
        "clip_id": "clip",
        "frame_interval_s": 0.5,
        "confirmed_by": "Recorder",
        "confirmed_at": "2026-09-13",
        "timeline": [
            {"from_s": 0, "to_s": 2, "state": "standing", "zone": "other"},
            {"from_s": 2, "to_s": 4, "state": "on_floor", "zone": "other"},
        ],
    }
    (clip / "labels").mkdir(parents=True)
    (clip / "labels" / "reference.yaml").write_text(yaml.safe_dump(reference))

    result = score_clip("clip", root=tmp_path, tag="model-tag")
    assert result["reports"][0]["gates"]["standing_recall"]["met"] is True
    machine = json.loads((clip / "reports" / "model-tag.json").read_text())
    assert machine["frames"]["all"]["upright_collapsed"]["per_state"]["upright"]["recall"] == 1
    assert machine["gates"]["on_floor_delay"]["met"] is True
    assert (clip / "reports" / "model-tag.md").exists()
