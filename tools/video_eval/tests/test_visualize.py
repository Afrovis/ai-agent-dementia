import json

import yaml

from video_eval.common import write_jsonl
from video_eval.visualize import (
    _current_manual,
    _fitted_rect,
    _load_inputs,
    _manual_state,
    _runtime_to_source,
)


def test_manual_markers_hold_until_the_next_timestamp():
    events = [
        {"t_s": 3.0, "action": "out_of_frame"},
        {"t_s": 6.0, "action": "in_frame"},
    ]

    assert _current_manual(events, 2.5) is None
    assert _current_manual(events, 3.0) == events[0]
    assert _current_manual(events, 5.5) == events[0]
    assert _current_manual(events, 6.0) == events[1]


def test_manual_actions_map_only_to_coarse_pipeline_states():
    assert _manual_state("in_bed_above_blanket") == "in_bed"
    assert _manual_state("sitting_on_floor") == "on_floor"
    assert _manual_state("walking") == "upright"
    assert _manual_state("out_of_frame") == "absent"


def test_manual_actions_include_extended_recorder_script_names():
    assert _manual_state("under_blanket_in_bed") == "in_bed"
    assert _manual_state("lay_down_on_bed") == "in_bed"
    assert _manual_state("sit_on_bed") == "sitting_up"
    assert _manual_state("sit_on_floor") == "on_floor"
    assert _manual_state("lay_down_on_floor") == "on_floor"
    assert _manual_state("lay_down_on_ground") == "on_floor"
    assert _manual_state("leave_room") == "absent"
    assert _manual_state("leave_frame") == "absent"
    assert _manual_state("stand_up") == "upright"
    assert _manual_state("get_up_from_bed") == "upright"
    assert _manual_state("get_up_from_floor") == "upright"
    assert _manual_state("get_up_from_ground") == "upright"
    assert _manual_state("enter_room") == "upright"


def test_runtime_letterbox_coordinates_map_back_to_source_image():
    assert _runtime_to_source((0.5, 0.125), 16 / 9) == (0.5, 0.0)
    assert _runtime_to_source((0.5, 0.875), 16 / 9) == (0.5, 1.0)


def test_runtime_coordinates_follow_the_source_aspect():
    # A 3:2 camera letterboxed into 4:3 has 1/18 bars top and bottom.
    x, y = _runtime_to_source((0.5, 1 / 18), 3 / 2)
    assert (x, round(y, 9)) == (0.5, 0.0)
    # A portrait camera is pillarboxed instead.
    x, y = _runtime_to_source((0.25, 0.3), 2 / 3)
    assert (round(x, 9), y) == (0.0, 0.3)


def test_fitted_rect_matches_where_the_frame_is_pasted():
    assert _fitted_rect((24, 92, 900, 506), 16 / 9) == (24, 92, 900, 506)
    assert _fitted_rect((24, 92, 900, 506), 3 / 2) == (94, 92, 759, 506)


def test_pipeline_visualization_uses_pose_data_from_prediction(tmp_path):
    clip = tmp_path / "clips" / "clip"
    predictions = clip / "predictions"
    predictions.mkdir(parents=True)
    (clip / "clip.yaml").write_text(
        yaml.safe_dump({"script": [], "video": {"duration_s": 0.5}}), encoding="utf-8"
    )
    write_jsonl(
        clip / "frames.jsonl",
        [{"frame_index": 0, "t_s": 0.0, "review_path": "review.jpg"}],
    )
    row = {
        "frame_index": 0,
        "t_s": 0.0,
        "bbox": [0.4, 0.2, 0.6, 0.8],
        "landmarks": {"nose": [0.5, 0.3, 0.9]},
    }
    write_jsonl(predictions / "selected.jsonl", [row])
    (predictions / "selected.meta.json").write_text(
        json.dumps({"parameters": {"backend": "yolo", "variant": "letterbox"}}),
        encoding="utf-8",
    )

    inputs = _load_inputs(tmp_path, "clip", "selected")

    assert inputs.poses == [row]
    assert inputs.vision == []
