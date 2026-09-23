"""Replay a scene card through decision_bench's injected-clock runner."""

from __future__ import annotations

from pathlib import Path

from decision_bench.runner import run_scenario
from decision_bench.schema import Checkpoint, Scenario, TimelineEvent
from decision_bench.stub_llm import StubLLM

from .scene import load_scene
from .trace import from_decision_bench


def run_card(path: Path, out: Path | None = None) -> Path:
    card = load_scene(path)
    timeline = []
    for beat in card.opening:
        if beat.move:
            timeline.append(
                TimelineEvent.model_validate(
                    {"t": beat.at, "person": {"state": beat.move.state, "zone": beat.move.zone}}
                )
            )
        elif beat.say:
            timeline.append(
                TimelineEvent.model_validate({"t": beat.at, "utterance": {"text": beat.say}})
            )
    if not timeline:
        raise ValueError("in-process replay requires move or say beats")
    scenario = Scenario(
        id=card.id,
        category=card.category,
        summary=card.persona.summary,
        start=card.start.strftime("%H:%M"),
        timeline=tuple(timeline),
        checkpoints=(
            Checkpoint(id="scene-end", question="What happened?", window=(0, card.duration_s)),
        ),
    )
    evidence = run_scenario(
        scenario, llm=StubLLM(), tail_seconds=max(0, card.duration_s - timeline[-1].t)
    )
    trace = from_decision_bench(evidence)
    target = out or path.with_name(f"{card.id}-inprocess-trace.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    trace.write_jsonl(target)
    return target
