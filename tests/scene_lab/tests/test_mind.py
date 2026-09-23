"""Claude mind contracts, with no Claude or service calls."""

import asyncio
from pathlib import Path

from scene_lab.mind import ClaudeMind, build_prompt
from scene_lab.scene import Persona, load_scene

PERSONA = Persona(summary="Jean looks for Tom", hidden_need="restroom", hearing="poor")
CONTEXT = {
    "t": 12.5,
    "trigger": "agent playback ended",
    "headline": "Good night, Jean",
    "history": [
        {"kind": "say", "text": "Where is Tom?"},
        {"kind": "heard", "text": "Tom is nearby and everything is settled"},
        {"kind": "move", "state": "walking", "zone": "door"},
    ],
}


def test_prompt_context_and_poor_hearing():
    first = build_prompt(PERSONA, CONTEXT)
    assert first == build_prompt(PERSONA, CONTEXT)
    assert "Jean looks for Tom" in first
    assert "restroom" in first
    assert "Where is Tom?" in first
    assert "Good night, Jean" in first
    assert "agent playback ended" in first
    assert "Tom is nearby and everything is settled" not in first
    assert "The bedside device said to you:" in first
    assert "never the bedside device" in first
    assert "You moved: walking in door" in first


def test_valid_plan_and_retry():
    calls = []

    def runner(_system, prompt, _schema, model, _effort):
        calls.append((prompt, model))
        if len(calls) == 1:
            return {"structured_output": {"beats": [{"say": "Hello", "style": "loud"}]}}
        return {
            "structured_output": {
                "beats": [{"say": "Tom?", "style": "normal"}],
                "end_scene": False,
                "note": "concerned",
            }
        }

    mind = ClaudeMind(PERSONA, runner=runner)
    plan = asyncio.run(mind.decide(CONTEXT))
    assert len(calls) == 2
    assert calls[0][1] == "sonnet"
    assert "Previous output failed validation" in calls[1][0]
    assert plan.beats == [{"say": "Tom?", "style": "normal"}]
    assert [row["ok"] for row in mind.calls] == [False, True]
    assert all("latency_s" in row for row in mind.calls)


def test_double_failure_ends_and_records_harness_error():
    mind = ClaudeMind(PERSONA, runner=lambda *_: {"structured_output": {"bad": True}})
    plan = asyncio.run(mind.decide(CONTEXT))
    assert plan.end_scene
    assert len(mind.calls) == 2
    assert mind.failures and "mind failure" in mind.failures[0]


def test_bad_json_retries_once():
    calls = 0

    def runner(*_args):
        nonlocal calls
        calls += 1
        return "{broken" if calls == 1 else '{"beats": [], "end_scene": true, "note": "done"}'

    mind = ClaudeMind(PERSONA, runner=runner)
    assert asyncio.run(mind.decide(CONTEXT)).end_scene
    assert calls == 2
    assert not mind.failures


def test_interrupted_sentence_is_cut_before_prompt():
    from scene_lab.page import Heard

    heard = Heard("Tom is nearby and everything is settled", 1, 2, True, 0.4)
    prompt = build_prompt(
        Persona(summary="Jean"), {"history": [{"kind": "heard", "text": heard.heard_text}]}
    )
    assert "Tom is nearby" in prompt
    assert "everything is settled" not in prompt


def test_persona_scenes_load():
    scenes = Path(__file__).parents[1] / "scenes"
    cards = [load_scene(path) for path in sorted(scenes.glob("persona-*.yaml"))]
    assert len(cards) == 5
    assert all(card.mind == "claude" and card.duration_s <= 600 for card in cards)


def test_scheduler_replaces_remaining_beats_and_debounces():
    from scene_lab.run import Plan, schedule
    from scene_lab.scene import Scene

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

        async def sleep(self, seconds):
            self.now += seconds
            await asyncio.sleep(0)

    class Body:
        def tick(self, _now):
            pass

        def move(self, *_args):
            pass

    class Audio:
        def __init__(self, clock):
            self.clock = clock
            self.lines = []
            self.active_until = 0

        @property
        def speaking(self):
            return self.clock.now < self.active_until

        def queue(self, line):
            self.lines.append((self.clock.now, line))
            self.active_until = self.clock.now + 1.0

        def finish_current_only(self):
            pass

    class Voice:
        def synthesize(self, text, _style):
            return text

    class Mind:
        calls = 0

        async def decide(self, _context):
            self.calls += 1
            await asyncio.sleep(0)
            if self.calls == 1:
                return Plan(
                    beats=[{"wait": 0.5}, {"say": "First plan"}, {"wait": 5}, {"say": "Old line"}]
                )
            return Plan(beats=[{"say": "New line"}])

    scene = Scene.model_validate(
        {
            "id": "replace",
            "category": "conversation",
            "start": "02:00",
            "duration_s": 6,
            "profile": "default",
            "mind": "claude",
            "mind_silence_s": 1,
            "persona": {"summary": "Jean"},
            "opening": [{"at": 0, "say": "Current line"}],
        }
    )
    clock = Clock()
    audio, mind = Audio(clock), Mind()
    asyncio.run(
        schedule(
            scene, Body(), None, Voice(), audio, clock=clock, sleep=clock.sleep, tail_s=0, mind=mind
        )
    )
    spoken = [line for _, line in audio.lines]
    # The opening runs first, then the mind; a later plan replaces the remaining beats.
    assert spoken[:2] == ["Current line", "First plan"]
    assert "New line" in spoken
    assert "Old line" not in spoken
    assert audio.lines[1][0] >= 0.5
    assert mind.calls >= 2
