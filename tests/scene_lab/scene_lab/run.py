"""Real-time scripted scene scheduling and nightsim orchestration."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import redis
import yaml
from nc_shared.bus import Bus

from .body import Body
from .bugs import RunDir, harness_error
from .director import Director, coverage, noise_level
from .mind import ClaudeMind
from .page import VirtualPage
from .recorder import agent_model, record_scene
from .scene import Beat, Scene, load_scene
from .stack import Stack, main_models
from .thresholds import load
from .voice import AudioStream, RoomNoise, Voice

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Plan:
    beats: list[dict | Beat] = field(default_factory=list)
    end_scene: bool = False
    note: str = ""


class Mind(Protocol):
    async def decide(self, context: dict) -> Plan: ...


class ScriptMind:
    async def decide(self, context: dict) -> Plan:
        return Plan()


DIRECTOR_TIMEOUT_S = 180.0


async def schedule(
    scene: Scene,
    body,
    page,
    voice,
    audio,
    *,
    clock=time.monotonic,
    sleep=asyncio.sleep,
    tail_s: float = 60,
    max_scene_s: float = 600,
    on_line=None,
    mind: Mind | None = None,
    page_task: asyncio.Task | None = None,
    decision_events: asyncio.Queue | None = None,
) -> dict:
    """Run beats on monotonic time while body frames and page audio continue."""
    start = clock()
    next_frame = start
    stopped = False
    capped = False
    interrupted = False
    lines = []
    history: list[dict] = []
    index = 0
    beats = list(scene.opening)
    mind = mind or ScriptMind()
    pending_mind: asyncio.Task | None = None
    speaking_before = False
    last_activity = 0.0
    next_silence = scene.mind_silence_s
    interrupt_on_agent = False
    agent_playing = False
    started_mind = False
    waypoints: list[float] = []
    stop_at = min(max_scene_s, scene.duration_s)
    try:
        while True:
            now = clock()
            elapsed = now - start
            if page_task is not None and page_task.done():
                page_task.result()
                raise ConnectionError("virtual page disconnected")
            # The scripted opening runs first; the mind takes over once it is used up.
            reason = None
            if not started_mind and scene.mind == "claude" and index >= len(beats):
                reason = "scene start"
                started_mind = True
            if decision_events is not None:
                while not decision_events.empty():
                    event = decision_events.get_nowait()
                    if isinstance(event, tuple):
                        event, payload = event
                    else:
                        payload = None
                    if event == "agent_playback_started":
                        agent_playing = True
                        last_activity = elapsed
                        if interrupt_on_agent:
                            # Move the next queued line to the front immediately.
                            for pos in range(index, len(beats)):
                                if beats[pos].say is not None:
                                    beats.insert(
                                        index, beats.pop(pos).model_copy(update={"at": elapsed})
                                    )
                                    interrupt_on_agent = False
                                    break
                    elif event in {"agent_playback_ended", "agent_playback_interrupted"}:
                        agent_playing = False
                        if payload is not None:
                            history.append({"kind": "heard", "text": payload.heard_text})
                        reason = "agent playback ended"
                        last_activity = elapsed
            speaking = bool(getattr(audio, "speaking", False))
            if speaking_before and not speaking:
                reason = "person_finished_speaking"
                last_activity = elapsed
            speaking_before = speaking
            if waypoints and elapsed >= waypoints[0]:
                waypoints.pop(0)
                reason = "body_reached_waypoint"
                last_activity = elapsed
            if (
                not speaking
                and not agent_playing
                and elapsed >= next_silence
                and elapsed - last_activity >= scene.mind_silence_s
            ):
                reason = f"{scene.mind_silence_s:g} s silence"
                next_silence = elapsed + scene.mind_silence_s
            if pending_mind is not None and pending_mind.done():
                plan = pending_mind.result()
                pending_mind = None
                if plan.end_scene:
                    stopped = True
                if hasattr(audio, "finish_current_only"):
                    audio.finish_current_only()
                beats[index:] = []
                offset = 0.0
                for planned in plan.beats:
                    if isinstance(planned, Beat):
                        beats.append(planned.model_copy(update={"at": elapsed + planned.at}))
                    elif "wait" in planned:
                        offset += float(planned["wait"])
                    elif "interrupt_if_agent_speaks" in planned:
                        interrupt_on_agent = True
                    else:
                        beats.append(Beat.model_validate({"at": elapsed + offset, **planned}))
                beats[index:] = sorted(beats[index:], key=lambda beat: beat.at)
            # A scripted scene has no mind: decision points must not replace its beats.
            if reason and (scene.mind == "script" or not started_mind):
                reason = None
            # The person's own plan is still running: finishing a line or reaching a waypoint
            # is not a reason to re-plan (that produced a nonstop monologue). The device
            # speaking and silence still are.
            if reason in {"person_finished_speaking", "body_reached_waypoint"} and index < len(
                beats
            ):
                reason = None
            if reason and pending_mind is None and not stopped:
                pending_mind = asyncio.create_task(
                    mind.decide(
                        {
                            "reason": reason,
                            "trigger": {
                                "person_finished_speaking": "person finished speaking",
                                "body_reached_waypoint": "waypoint reached",
                            }.get(reason, reason),
                            "t": elapsed,
                            "scene": scene.id,
                            "history": list(history),
                            "headline": getattr(page, "headline", ""),
                        }
                    )
                )
            while now >= next_frame:
                body.tick(next_frame)
                next_frame += 0.5
            while index < len(beats) and beats[index].at <= elapsed:
                beat = beats[index]
                if beat.say is not None and getattr(audio, "speaking", False):
                    break
                index += 1
                if beat.end:
                    stopped = True
                    break
                if beat.move:
                    body.move(beat.move.state, beat.move.zone, beat.move.over_s)
                    history.append(
                        {"kind": "move", "state": beat.move.state, "zone": beat.move.zone}
                    )
                    waypoints.append(elapsed + beat.move.over_s)
                    waypoints.sort()
                elif beat.say is not None:
                    samples = voice.synthesize(beat.say, beat.style)
                    audio.queue(samples)
                    last_activity = elapsed
                    row = {"t": round(elapsed, 3), "text": beat.say, "style": beat.style}
                    lines.append(row)
                    history.append({"kind": "say", "text": beat.say})
                    if on_line:
                        on_line(row)
            # Due beats run first: an end beat at exactly the cap is a planned end.
            if stopped:
                break
            if elapsed >= max_scene_s:
                capped = True
                break
            if elapsed >= stop_at:
                break
            next_beat = beats[index].at if index < len(beats) else stop_at
            delay = min(
                next_frame - now,
                start + next_beat - now,
                start + stop_at - now,
                start + max_scene_s - now,
            )
            await sleep(max(0.001, delay))
        if not capped:
            tail_end = min(start + max_scene_s, clock() + tail_s)
            while clock() < tail_end:
                now = clock()
                if page_task is not None and page_task.done():
                    page_task.result()
                    raise ConnectionError("virtual page disconnected")
                while now >= next_frame:
                    body.tick(next_frame)
                    next_frame += 0.5
                await sleep(max(0.001, min(next_frame - now, tail_end - now)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
    finally:
        if pending_mind is not None:
            pending_mind.cancel()
    return {
        "capped": capped,
        "interrupted": interrupted,
        "mind_lines": lines,
        "mind_calls": getattr(mind, "calls", []),
        "mind_errors": getattr(mind, "failures", []),
    }


async def run_live(
    source: Path,
    *,
    no_stack_up: bool = False,
    keep_stack: bool = False,
    runs_root: Path | None = None,
    run_dir: Path | None = None,
) -> Path:
    scene = load_scene(source)
    thresholds = load()
    run = RunDir("live", root=runs_root, path=run_dir, thresholds=thresholds)
    profile = REPO_ROOT / (
        "config/person.example.yaml" if scene.profile == "default" else scene.profile
    )
    strategies = REPO_ROOT / (
        "config/strategies.example.yaml" if scene.strategies == "default" else scene.strategies
    )
    env = {
        "NIGHTSIM_DATA": str(run.path / "stack-data"),
        "NIGHTSIM_PERSON": str(profile),
        "NIGHTSIM_STRATEGIES": str(strategies),
        "NIGHTSIM_MODELS": str(main_models(REPO_ROOT)),
    }
    (run.path / "stack-data").mkdir(exist_ok=True)
    stack = Stack(REPO_ROOT, env=env)
    preflight = stack.preflight()
    run.metadata = {
        "kind": "live",
        **preflight,
        "model": agent_model(REPO_ROOT, stack.env),
        "scene_count": 0,
    }
    try:
        if not no_stack_up:
            stack.up()
        await _execute_scene(scene, source, run, stack, preflight, thresholds)
    finally:
        if not keep_stack:
            stack.down()
        run.finish("live", preflight.get("commit", "-"), agent_model(REPO_ROOT, stack.env), 0, 1)
    return run.path


async def _execute_scene(scene, source, run, stack, preflight, thresholds):
    start = datetime.now(UTC)
    errors = []
    state = {
        "capped": False,
        "interrupted": False,
        "mind_lines": [],
        "mind_calls": [],
        "mind_errors": [],
    }
    mind = ClaudeMind(scene.persona) if scene.mind == "claude" else ScriptMind()
    try:
        stack.wait_ready()
        stack.reset()
        stack.wait_ready()
        bus = Bus(redis.Redis(host="localhost", port=16379))
        body = Body(bus.publish, time.monotonic, noise=scene.noise.model_dump())
        audio = AudioStream(RoomNoise())
        voice = Voice("en_US-amy-medium")
        decisions = asyncio.Queue()

        async def playback_ended(_heard):
            decisions.put_nowait(("agent_playback_ended", _heard))

        async def playback_interrupted(heard):
            decisions.put_nowait(("agent_playback_interrupted", heard))

        async def playback_started(_record):
            decisions.put_nowait("agent_playback_started")

        page = VirtualPage(
            "ws://localhost:18443",
            audio,
            on_playback_started=playback_started,
            on_playback_ended=playback_ended,
            on_playback_interrupted=playback_interrupted,
        )
        page_task = asyncio.create_task(page.run())
        await asyncio.sleep(0.2)
        if page_task.done():
            await page_task
        start = datetime.now(UTC)
        state = await schedule(
            scene,
            body,
            page,
            voice,
            audio,
            tail_s=thresholds.tail_s,
            max_scene_s=thresholds.max_scene_s,
            page_task=page_task,
            decision_events=decisions,
            mind=mind,
        )
        if page_task.done():
            await page_task
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        state["interrupted"] = isinstance(exc, KeyboardInterrupt)
    finally:
        if "page_task" in locals():
            page_task.cancel()
            try:
                await page_task
            except (asyncio.CancelledError, Exception):
                pass
        if state["interrupted"]:
            errors.append("KeyboardInterrupt: scene interrupted")
        return record_scene(
            run,
            scene.id,
            source,
            start,
            stack,
            preflight=preflight,
            mind_lines=state["mind_lines"],
            mind_calls=mind.calls if isinstance(mind, ClaudeMind) else [],
            capped=state["capped"],
            errors=errors + (mind.failures if isinstance(mind, ClaudeMind) else []),
        )


async def run_hours(
    hours: float,
    *,
    runs_root: Path | None = None,
    keep_stack: bool = False,
    max_scenes: int | None = None,
    director: Director | None = None,
    scene_runner=None,
    stack_factory=Stack,
    clock=time.monotonic,
) -> Path:
    """Use one nightsim stack for a bounded series of director scenes."""
    if hours <= 0:
        raise ValueError("hours must be positive")
    if max_scenes is not None and max_scenes < 1:
        raise ValueError("max_scenes must be positive")
    thresholds = load()
    run = RunDir("live", root=runs_root, thresholds=thresholds)
    default_person = REPO_ROOT / "config/person.example.yaml"
    default_strategies = REPO_ROOT / "config/strategies.example.yaml"
    stack_data = run.path / "stack-data"
    stack_data.mkdir(exist_ok=True)
    mounted_person = stack_data / "person.yaml"
    mounted_strategies = stack_data / "strategies.yaml"
    mounted_person.write_bytes(default_person.read_bytes())
    mounted_strategies.write_bytes(default_strategies.read_bytes())
    stack = stack_factory(
        REPO_ROOT,
        env={
            "NIGHTSIM_DATA": str(stack_data),
            "NIGHTSIM_PERSON": str(mounted_person),
            "NIGHTSIM_STRATEGIES": str(mounted_strategies),
            "NIGHTSIM_MODELS": str(main_models(REPO_ROOT)),
        },
    )
    preflight = stack.preflight()
    model = agent_model(REPO_ROOT, stack.env)
    run.metadata = {"kind": "live", **preflight, "model": model, "scene_count": 0}
    chosen = director or Director()
    scenes: list[dict] = []
    (run.path / "coverage.json").write_text(json.dumps(coverage(scenes), indent=2) + "\n")
    deadline = clock() + hours * 3600
    started = False
    try:
        stack.up()
        started = True
        stack.wait_ready()
        while clock() <= deadline - thresholds.max_scene_s:
            if max_scenes is not None and len(scenes) >= max_scenes:
                break
            remaining = deadline - clock()
            previous = scenes[-1] if scenes else None
            bugs = (
                [json.loads(line) for line in (run.path / "bugs.jsonl").read_text().splitlines()]
                if (run.path / "bugs.jsonl").exists()
                else []
            )
            from collections import Counter

            bug_counts = dict(Counter(item["fingerprint"] for item in bugs))
            state = {
                "run": run,
                "scenes": scenes,
                "bugs": bug_counts,
                "previous": previous,
                "time_left_s": remaining,
            }
            try:
                # The Claude call is synchronous; keep it off the event loop and bounded so a
                # hung CLI cannot stall an unattended run.
                scene = await asyncio.wait_for(
                    asyncio.to_thread(chosen.next_scene, state), timeout=DIRECTOR_TIMEOUT_S
                )
            except TimeoutError:
                scene = chosen.fallback(state, f"director timed out after {DIRECTOR_TIMEOUT_S} s")
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Ctrl-C while the director thinks: no new scene; still finish the run.
                break
            # Director latency is part of the deadline; do not start if its budget expired.
            remaining = deadline - clock()
            if remaining < thresholds.max_scene_s or scene.duration_s > remaining:
                break
            source = run.path / "scenes" / f"{scene.id}.yaml"
            source.parent.mkdir(exist_ok=True)
            source.write_text(yaml.safe_dump(scene.model_dump(mode="json"), sort_keys=False))
            # Compose binds these fixed files once. Write in place so a reset sees
            # the next card's allowed profile and strategies without recreating containers.
            profile = default_person if scene.profile == "default" else REPO_ROOT / scene.profile
            strategies = (
                default_strategies
                if scene.strategies == "default"
                else REPO_ROOT / scene.strategies
            )
            mounted_person.write_bytes(profile.read_bytes())
            mounted_strategies.write_bytes(strategies.read_bytes())
            interrupted = False
            try:
                if scene_runner is None:
                    _, entries = await _execute_scene(
                        scene, source, run, stack, preflight, thresholds
                    )
                else:
                    entries = await scene_runner(scene, source, run, stack, preflight, thresholds)
            except KeyboardInterrupt:
                interrupted = True
                folder = run.path / scene.id
                folder.mkdir(exist_ok=True)
                if not (folder / "report.md").exists():
                    (folder / "report.md").write_text(f"# {scene.id}\n\nScene interrupted.\n")
                entries = [harness_error(run.id, scene.id, "KeyboardInterrupt: scene interrupted")]
                run.append(entries)
            scenes.append(
                {
                    "id": scene.id,
                    "category": scene.category,
                    "persona_tag": scene.persona_tag,
                    "stressors": scene.stressors,
                    "noise": noise_level(scene),
                    "failures": [
                        {"id": e.check, "severity": e.severity, "fingerprint": e.fingerprint}
                        for e in entries
                        if e.origin == "agent"
                    ],
                    "harness_errors": [e.summary for e in entries if e.origin == "harness"],
                }
            )
            run.metadata["scene_count"] = len(scenes)
            (run.path / "coverage.json").write_text(json.dumps(coverage(scenes), indent=2) + "\n")
            run.render_bugs_md()
            if interrupted or any("KeyboardInterrupt" in e.summary for e in entries):
                break
    finally:
        if started and not keep_stack:
            stack.down()
        run.finish("live", preflight.get("commit", "-"), model, hours, len(scenes))
    return run.path
