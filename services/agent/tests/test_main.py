"""Tests for the fake agent's step table and per-step publish, using FakeBus.

Per HANDOFF.md section 4, every service must be testable with no camera,
mic, Ollama, or Redis. These tests exercise `run_once` and `STEPS` directly
against a `FakeBus`; they never call the infinite, real-time `run()` loop
or sleep for real durations.
"""

from nc_shared.bus import FakeBus
from nc_shared.events import Say, Show

from agent.main import STEPS, run_once

VALID_FACES = {"asleep", "awake", "speaking", "listening"}


def test_run_once_publishes_a_valid_show_for_every_step():
    bus = FakeBus()
    bus.ensure_group("show", "test-group")

    for step in STEPS:
        run_once(bus, step)

    read = bus.read("show", "test-group", "consumer-1", count=len(STEPS))
    assert len(read) == len(STEPS)
    for (_, event), step in zip(read, STEPS, strict=True):
        assert isinstance(event, Show)
        assert event.face in VALID_FACES
        assert event.face == step.face
        assert event.headline == step.headline
        assert event.body == step.body
        assert event.photo_id == step.photo_id
        assert event.brightness == step.brightness
        assert 0.0 <= event.brightness <= 1.0
        assert event.source == "agent"


def test_run_once_publishes_say_only_when_the_step_has_say_text():
    bus = FakeBus()
    bus.ensure_group("say", "test-group")

    for step in STEPS:
        run_once(bus, step)

    read = bus.read("say", "test-group", "consumer-1", count=len(STEPS))
    expected_says = [step for step in STEPS if step.say_text is not None]
    assert len(read) == len(expected_says)

    for (_, event), step in zip(read, expected_says, strict=True):
        assert isinstance(event, Say)
        assert event.text == step.say_text
        assert event.strategy == (step.say_strategy or "")
        assert event.interruptible == step.interruptible
        assert event.source == "agent"


def test_run_once_uses_the_given_session_id():
    bus = FakeBus()
    bus.ensure_group("show", "test-group")

    run_once(bus, STEPS[0], session_id="custom-session")

    _, event = bus.read("show", "test-group", "consumer-1")[0]
    assert event.session_id == "custom-session"


def test_steps_cycle_through_every_face_state():
    faces = {step.face for step in STEPS}
    assert faces == VALID_FACES


def test_steps_vary_brightness():
    brightnesses = {step.brightness for step in STEPS}
    assert len(brightnesses) > 1
    assert all(0.0 <= b <= 1.0 for b in brightnesses)


def test_at_least_one_step_has_a_photo_id():
    assert any(step.photo_id is not None for step in STEPS)


def test_photo_ids_are_demo_ids():
    # A fresh checkout has no caregiver-uploaded photos, so the fake agent
    # may only reference the `demo_` images that `embodiment` ships. Any
    # other id would 404 on every cycle of `docker compose up`.
    assert all(step.photo_id.startswith("demo_") for step in STEPS if step.photo_id is not None)
