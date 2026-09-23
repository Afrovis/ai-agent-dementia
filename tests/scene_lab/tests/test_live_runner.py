"""No-service checks for the scripted runner and its adapters."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scene_lab.bugs import RunDir
from scene_lab.diff import compare
from scene_lab.frombench import convert
from scene_lab.recorder import record_scene
from scene_lab.run import schedule
from scene_lab.scene import Scene, load_scene
from scene_lab.stack import Stack
from scene_lab.trace import Trace, TraceEvent


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


class Body:
    def __init__(self, clock):
        self.clock = clock
        self.moves = []
        self.frames = []

    def move(self, state, zone, over_s):
        self.moves.append((round(self.clock(), 2), state, zone, over_s))

    def tick(self, now):
        self.frames.append(now)


class Voice:
    def synthesize(self, text, style):
        return text


class Audio:
    def __init__(self, clock):
        self.clock = clock
        self.queued = []

    def queue(self, samples):
        self.queued.append((round(self.clock(), 2), samples))


def card():
    return Scene.model_validate(
        {
            "id": "test-scene",
            "category": "conversation",
            "start": "02:40",
            "duration_s": 10,
            "profile": "default",
            "persona": {"summary": "A person"},
            "opening": [
                {"at": 2, "move": {"state": "standing", "zone": "bed"}},
                {"at": 4, "say": "Hello"},
                {"at": 6, "end": True},
            ],
        }
    )


def test_scheduler_beat_times_and_tail():
    clock = Clock()
    body, audio = Body(clock), Audio(clock)
    state = asyncio.run(
        schedule(card(), body, None, Voice(), audio, clock=clock, sleep=clock.sleep, tail_s=2)
    )
    assert body.moves == [(2, "standing", "bed", 0)]
    assert audio.queued == [(4, "Hello")]
    assert clock.now == 8
    assert state["mind_lines"][0]["text"] == "Hello"
    assert not state["capped"]


def test_scheduler_cap_and_interrupt():
    clock = Clock()
    scene = card().model_copy(update={"opening": [], "duration_s": 10})
    state = asyncio.run(
        schedule(
            scene,
            Body(clock),
            None,
            Voice(),
            Audio(clock),
            clock=clock,
            sleep=clock.sleep,
            tail_s=0,
            max_scene_s=3,
        )
    )
    assert state["capped"]

    async def interrupt(_seconds):
        raise KeyboardInterrupt

    state = asyncio.run(
        schedule(card(), Body(clock), None, Voice(), Audio(clock), clock=clock, sleep=interrupt)
    )
    assert state["interrupted"]


def test_stack_command_and_flush_guard(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="")

    monkeypatch.setattr("scene_lab.stack.subprocess.run", fake_run)
    stack = Stack(tmp_path, env={"NIGHTSIM_DATA": str(tmp_path / "data")})
    stack.up()
    assert calls[0][0][:8] == [
        "docker",
        "compose",
        "-p",
        "nightsim",
        "-f",
        "docker-compose.yml",
        "-f",
        "tests/scene_lab/compose.sim.yml",
    ]
    assert calls[0][1]["env"]["NIGHTSIM_DATA"] == str(tmp_path / "data")
    with pytest.raises(ValueError, match="16379"):
        Stack(tmp_path, redis_port=6379).flush()


def test_recorder_fake_export(tmp_path):
    source = tmp_path / "card.yaml"
    source.write_text("id: demo\n")
    stamp = "2026-09-22T00:00:00+00:00"
    export = tmp_path / "fixture.jsonl"
    export.write_text(
        json.dumps(
            {
                "stream": "person",
                "event_type": "PersonState",
                "ts": stamp,
                "recorded_at": 100,
                "payload": {"state": "sitting_up", "zone": "bed"},
            }
        )
        + "\n"
    )
    stack = SimpleNamespace(
        logs=lambda *_: "agent log", repo_root=tmp_path, env={"AGENT_LLM_MODEL": "test-model"}
    )
    run = RunDir("live", root=tmp_path / "runs")
    folder, entries = record_scene(
        run,
        "demo",
        source,
        datetime.now(UTC),
        stack,
        preflight={"commit": "abc", "contention": False},
        mind_lines=[{"t": 1, "text": "Hello", "style": "normal"}],
        capped=True,
        export_file=export,
    )
    assert all(
        (folder / name).exists()
        for name in (
            "scene.yaml",
            "export.jsonl",
            "agent.log",
            "mind.jsonl",
            "trace.jsonl",
            "report.json",
            "report.md",
        )
    )
    assert any(entry.check == "SM-3" and entry.origin == "agent" for entry in entries)
    assert json.loads((folder / "report.json").read_text())["contention"] is False
    assert (run.path / "bugs.md").exists()


def test_frombench_and_diff(tmp_path):
    path = convert("restroom-01", tmp_path)
    scene = load_scene(path)
    assert scene.opening[0].move.state == "sitting_up"
    assert any(beat.say == "I need the toilet." for beat in scene.opening)
    assert scene.opening[-1].end is True
    live = Trace(
        id="a",
        source="live",
        events=[
            TraceEvent(t=1, kind="output", type="Say", data={}),
            TraceEvent(t=7, kind="output", type="Notify", data={}),
        ],
    )
    offline = Trace(
        id="b",
        source="decision_bench",
        events=[
            TraceEvent(t=2, kind="output", type="Say", data={}),
            TraceEvent(t=12, kind="output", type="Show", data={}),
        ],
    )
    rows = compare(live, offline, 3)
    assert [row["status"] for row in rows] == ["matched", "only-live", "only-in-process"]


def test_mind_plan_runs_while_clock_advances():
    from scene_lab.run import Plan

    clock = Clock()
    body, audio = Body(clock), Audio(clock)
    scene = card().model_copy(
        update={"opening": [card().opening[0]], "duration_s": 7, "mind": "claude"}
    )

    class Mind:
        async def decide(self, context):
            assert context["reason"] in {"scene start", "body_reached_waypoint"}
            await asyncio.sleep(0)
            return Plan(beats=[{"wait": 1}, {"say": "Later"}], end_scene=False)

    async def yielding_sleep(seconds):
        clock.now += seconds
        await asyncio.sleep(0)

    asyncio.run(
        schedule(
            scene,
            body,
            None,
            Voice(),
            audio,
            clock=clock,
            sleep=yielding_sleep,
            tail_s=0,
            mind=Mind(),
        )
    )
    assert audio.queued and audio.queued[0][1] == "Later"
    assert audio.queued[0][0] >= 3
