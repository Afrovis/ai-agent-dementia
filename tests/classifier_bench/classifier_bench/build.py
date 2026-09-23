"""Build per-action questions from paired model annotation runs."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping
from pathlib import Path

import yaml
from decision_bench.calibrate import actions, checkpoints, placement
from decision_bench.schema import Checkpoint, Scenario, load_scenario


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def serialize_state(scenario: Scenario, checkpoint: Checkpoint) -> str:
    """Serialize fixture state through the checkpoint window end.

    The result is deterministic, compact, human-readable, and uses only the scenario
    fixture. Events after the inclusive checkpoint window end are deliberately omitted so
    the classifier cannot see the future.
    """
    if checkpoint.window is None:
        raise ValueError(f"checkpoint {checkpoint.id!r} has no window")
    start, end = checkpoint.window
    lines = [
        f"summary: {scenario.summary}",
        f"start: {scenario.start.strftime('%H:%M')}",
        f"window: {_number(start)}-{_number(end)} seconds",
    ]
    for event in scenario.timeline:
        if event.t > end:
            break
        prefix = f"t={_number(event.t)}"
        if event.person is not None:
            person = event.person
            lines.append(f"{prefix} person={person.state}/{person.zone} conf={person.confidence:g}")
        else:
            assert event.utterance is not None
            utterance = event.utterance
            quoted = json.dumps(utterance.text, ensure_ascii=False)
            lines.append(f"{prefix} said: {quoted} conf={utterance.confidence:g}")
    lines.append(f"question: {checkpoint.question or ''}")
    return "\n".join(lines)


def build_questions(annotation_dir: Path, scenario_dir: Path) -> list[dict[str, object]]:
    """Derive questions for every action in two-run annotation files."""
    questions: list[dict[str, object]] = []
    for annotation_path in sorted(annotation_dir.glob("*.yaml")):
        raw = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            continue
        second = raw.get("second_opinion")
        if not isinstance(second, Mapping):
            continue
        scenario_id = str(raw.get("scenario") or annotation_path.stem)
        scenario = load_scenario(scenario_dir / f"{scenario_id}.yaml")
        scenario_checkpoints = {item.id: item for item in scenario.checkpoints}
        run1 = checkpoints(raw)
        run2 = checkpoints(second)
        for checkpoint_id in sorted(run1.keys() & run2.keys()):
            if checkpoint_id not in scenario_checkpoints:
                raise ValueError(f"{scenario_id}: unknown checkpoint {checkpoint_id!r}")
            first = run1[checkpoint_id]
            second_checkpoint = run2[checkpoint_id]
            source = scenario_checkpoints[checkpoint_id]
            state = serialize_state(scenario, source)
            for action in sorted(actions(first) | actions(second_checkpoint)):
                run1_placement = placement(first, action)
                run2_placement = placement(second_checkpoint, action)
                agreed = run1_placement == run2_placement
                kind, value = action
                questions.append(
                    {
                        "qid": f"{scenario_id}::{checkpoint_id}::{kind}:{value}",
                        "scenario": scenario_id,
                        "checkpoint": checkpoint_id,
                        "action": {kind: value},
                        "action_kind": kind,
                        "set": "agree" if agreed else "disputed",
                        "truth": run1_placement if agreed else None,
                        "run1": run1_placement,
                        "run2": run2_placement,
                        "state": state,
                        "question": source.question or "",
                    }
                )
    return questions


def write_questions(path: Path, questions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")


def build_summary(questions: list[dict[str, object]]) -> str:
    """Return counts and state-size statistics for a build."""
    states = [str(item["state"]) for item in questions]
    lengths = [len(item) for item in states]
    if not lengths:
        return "questions=0 agree=0 disputed=0\nstate chars min/median/max: n/a"
    agree = sum(item["set"] == "agree" for item in questions)
    stats = (min(lengths), statistics.median(lengths), max(lengths))
    tokens = tuple(value / 4 for value in stats)
    return (
        f"questions={len(questions)} agree={agree} disputed={len(questions) - agree}\n"
        f"state chars min/median/max: {stats[0]:g}/{stats[1]:g}/{stats[2]:g}\n"
        f"state rough tokens (chars/4) min/median/max: "
        f"{tokens[0]:.1f}/{tokens[1]:.1f}/{tokens[2]:.1f}"
    )
