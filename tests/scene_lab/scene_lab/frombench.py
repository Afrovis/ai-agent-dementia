"""Convert a decision bench timeline into a scripted live scene."""

from __future__ import annotations

from pathlib import Path

import yaml
from decision_bench.schema import SCENARIOS_DIR, load_scenario

from .scene import load_scene
from .thresholds import load


def convert(source: str | Path, out: str | Path, *, tail_s: float | None = None) -> Path:
    source = Path(source)
    if not source.is_file():
        source = SCENARIOS_DIR / f"{source.stem}.yaml"
    scenario = load_scenario(source)
    tail = load().tail_s if tail_s is None else tail_s
    last = max(item.t for item in scenario.timeline)
    duration = min(600, last + tail)
    beats = []
    for item in scenario.timeline:
        if item.person:
            beats.append(
                {
                    "at": item.t,
                    "move": {"state": item.person.state, "zone": item.person.zone, "over_s": 0},
                }
            )
        else:
            beats.append({"at": item.t, "say": item.utterance.text})
    beats.append({"at": min(600, last + tail), "end": True})
    card = {
        "id": scenario.id,
        "category": scenario.category,
        "start": scenario.start.strftime("%H:%M"),
        "duration_s": duration,
        "profile": "config/person.example.yaml",
        "strategies": "default",
        "persona": {
            "summary": scenario.summary,
            "hearing": "normal",
            "patience": "medium",
            "interrupts": False,
        },
        "opening": beats,
        "stressors": [],
        "mind": "script",
        "noise": {"flicker": 0, "low_confidence": 0},
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{scenario.id}.yaml"
    target.write_text(yaml.safe_dump(card, sort_keys=False))
    load_scene(target)
    return target
