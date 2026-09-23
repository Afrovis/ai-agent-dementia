from scene_lab.body import Body


def test_confirmation_flicker_heartbeat_and_zone():
    events = []
    now = [0.0]
    body = Body(events.append, lambda: now[0])
    body.move("sitting_up", "bed")
    for now[0] in (0, 0.5):
        body.tick(now[0])
    assert events == []
    body.tick(1.0)
    assert [(e.state, e.zone, e.source) for e in events] == [("sitting_up", "bed", "scene_lab")]
    body.move("standing", "door")
    body.tick(1.5)
    body.move("sitting_up", "bed")  # one-frame detector flicker
    body.tick(2.0)
    assert len(events) == 1
    body.tick(61.0)
    assert len(events) == 2
    assert events[-1].zone == "bed"
    assert events[-1].scene_note is None


def test_transition_has_intermediate_states():
    events = []
    now = [0.0]
    body = Body(events.append, lambda: now[0], confirm_frames=1)
    body.move("walking", "bathroom_path", over_s=6)
    for value in (0, 2, 4, 6):
        now[0] = value
        body.tick(value)
    assert [event.state for event in events] == ["sitting_up", "standing", "walking"]
    assert all(event.zone == "bathroom_path" for event in events)


def test_low_confidence_does_not_confirm_new_state():
    events = []
    body = Body(events.append, lambda: 0, noise={"low_confidence": 1})
    body.move("walking", "door")
    for value in (0, 0.5, 1, 1.5):
        body.tick(value)
    assert events == []
