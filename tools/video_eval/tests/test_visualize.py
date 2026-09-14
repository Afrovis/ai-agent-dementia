from video_eval.visualize import _current_manual, _manual_state, _runtime_to_source


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


def test_runtime_letterbox_coordinates_map_back_to_source_image():
    assert _runtime_to_source((0.5, 0.125)) == (0.5, 0.0)
    assert _runtime_to_source((0.5, 0.875)) == (0.5, 1.0)
