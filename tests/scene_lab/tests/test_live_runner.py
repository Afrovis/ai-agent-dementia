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
    # An end beat exactly at the cap is a planned end, not a cap stop.
    clock = Clock()
    planned = card().model_copy(update={"opening": [{"at": 3, "end": True}], "duration_s": 3})
    planned = type(planned).model_validate(planned.model_dump())
    state = asyncio.run(
        schedule(
            planned,
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
    assert not state["capped"]

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
            TraceEvent(t=1, kind="output", type="Say", data={"strategy": "path_light"}),
            TraceEvent(t=7, kind="output", type="Notify", data={"level": "info"}),
            TraceEvent(t=9, kind="output", type="Say", data={"strategy": "guided_return"}),
        ],
    )
    offline = Trace(
        id="b",
        source="decision_bench",
        events=[
            TraceEvent(t=2, kind="output", type="Say", data={"strategy": "path_light"}),
            TraceEvent(t=2, kind="output", type="Say", data={"strategy": "guided_return"}),
            TraceEvent(t=12, kind="output", type="Show", data={"face": "awake"}),
        ],
    )
    rows = compare(live, offline, 3)
    # Same decisions line up by content; a late one is "shifted", not unmatched.
    assert [row["status"] for row in rows] == ["matched", "only-live", "shifted", "only-in-process"]
    assert rows[2]["delta_s"] == 7


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


def test_stream_tap_keeps_capped_history_and_skips_media(tmp_path):
    from nc_shared.events import Activity, AudioChunk

    from scene_lab.recorder import StreamTap

    def fields(event):
        return {
            b"event_type": type(event).__name__.encode(),
            b"data": event.model_dump_json().encode(),
        }

    class FakeRedis:
        def __init__(self):
            self.streams = {"activity": [], "audio_in": []}
            self.reads = []
            self.next_id = 0

        def add(self, stream, event):
            entries = self.streams[stream]
            self.next_id += 1
            entries.append((f"{self.next_id}-0", fields(event)))
            del entries[:-2]  # capped like the real activity stream

        def xrange(self, stream, min="-", max="+"):
            self.reads.append(stream)
            entries = self.streams.get(stream, [])
            if min == "-":
                return list(entries)
            low = int(min.lstrip("(").split("-")[0])
            return [(i, f) for i, f in entries if int(i.split("-")[0]) > low]

    client = FakeRedis()
    tap = StreamTap(client, streams=["activity", "audio_in"])
    for n in range(5):
        client.add(
            "activity",
            Activity(
                source="embodiment",
                service="embodiment",
                kind="playback",
                phase="start",
                detail=f"p{n}",
            ),
        )
        client.add(
            "audio_in", AudioChunk(source="embodiment", pcm16=b"\x00\x00", sample_rate=16000)
        )
        tap.poll()
    path = tap.write(tmp_path / "tap.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # All five survive although the stream only ever held two at a time.
    assert [r["payload"]["detail"] for r in rows] == [f"p{n}" for n in range(5)]
    assert "audio_in" not in client.reads
