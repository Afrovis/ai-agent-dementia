"""Replay a scene card through decision_bench's injected-clock runner."""

from __future__ import annotations

from pathlib import Path

from decision_bench.runner import run_scenario
from decision_bench.schema import Checkpoint, Scenario, TimelineEvent
from decision_bench.stub_llm import StubLLM

from .scene import load_scene
from .trace import from_decision_bench


def _llm(backend: str, model: str | None, url: str, timeout_s: float):
    if backend == "stub":
        return StubLLM()
    from agent.llm import local_llm

    return local_llm(backend, url=url, model=model, timeout_seconds=timeout_s)


def run_card(
    path: Path,
    out: Path | None = None,
    *,
    backend: str = "stub",
    model: str | None = None,
    url: str = "http://127.0.0.1:11434",
    timeout_s: float = 30.0,
    llm_latency: str = "none",
) -> Path:
    """Run the card's scripted beats in-process. Use the live run's model to isolate
    timing differences from model differences; stub is deterministic."""
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
        scenario,
        llm=_llm(backend, model, url, timeout_s),
        tail_seconds=max(0, card.duration_s - timeline[-1].t),
        llm_latency=llm_latency,
    )
    trace = from_decision_bench(evidence)
    trace.meta.update({"backend": backend, "model": model, "llm_latency": llm_latency})
    target = out or path.with_name(f"{card.id}-inprocess-trace.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    trace.write_jsonl(target)
    return target
