"""Apply human-reviewed annotations to scenario fixtures."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from decision_bench.annotate import ANNOTATIONS_DIR, AnnotatorError, validate_annotation
from decision_bench.schema import (
    SCENARIOS_DIR,
    Checkpoint,
    Scenario,
    citation_problems,
    guideline_clauses,
    load_scenario,
)


class ReviewError(RuntimeError):
    """A review cannot safely be applied."""


class _FlowMap(dict):
    pass


class _FlowList(list):
    pass


class _Folded(str):
    pass


class _Dumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


_Dumper.add_representer(
    _FlowMap,
    lambda dumper, value: dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=True),
)
_Dumper.add_representer(
    _FlowList,
    lambda dumper, value: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", value, flow_style=True
    ),
)
_Dumper.add_representer(
    _Folded,
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:str", value, style=">"),
)


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ReviewError(f"missing annotation file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ReviewError(f"{path} must contain a mapping")
    return raw


def _actions(checkpoint: dict[str, Any], field: str) -> set[tuple[str, str]]:
    values = set()
    for action in checkpoint.get(field) or []:
        if isinstance(action, dict) and len(action) == 1:
            kind, value = next(iter(action.items()))
            values.add((str(kind), str(value)))
    return values


def _labels(checkpoint: dict[str, Any]) -> dict[str, object]:
    return {
        "acceptable": checkpoint.get("acceptable") or [],
        "must_not": checkpoint.get("must_not") or [],
        "escalate_by": checkpoint.get("escalate_by"),
        "trigger": checkpoint.get("trigger"),
        "cites": checkpoint.get("cites") or [],
    }


def _different(model: dict[str, Any], human: dict[str, Any]) -> bool:
    return any(
        (
            _actions(model, field) != _actions(human, field)
            if field in {"acceptable", "must_not"}
            else set(model.get(field) or []) != set(human.get(field) or [])
            if field == "cites"
            else model.get(field) != human.get(field)
        )
        for field in ("acceptable", "must_not", "escalate_by", "trigger", "cites")
    )


def _action_dicts(actions: tuple[object, ...]) -> list[_FlowMap]:
    return [_FlowMap(action.model_dump(exclude_none=True)) for action in actions]


def _checkpoint_dict(checkpoint: Checkpoint, source: Checkpoint) -> dict[str, object]:
    result: dict[str, object] = {"id": checkpoint.id}
    if source.window is not None:
        result["window"] = _FlowList(_clean_number(value) for value in source.window)
    if source.question is not None:
        result["question"] = (
            _Folded(source.question) if len(source.question) > 100 else source.question
        )
    if checkpoint.acceptable:
        result["acceptable"] = _action_dicts(checkpoint.acceptable)
    if checkpoint.must_not:
        result["must_not"] = _action_dicts(checkpoint.must_not)
    if checkpoint.escalate_by is not None:
        result["escalate_by"] = _clean_number(checkpoint.escalate_by)
        if checkpoint.trigger is not None:
            result["trigger"] = _clean_number(checkpoint.trigger)
        result["threshold_source"] = "caregiver"
    result["rationale"] = _Folded(checkpoint.rationale or "")
    if checkpoint.cites:
        result["cites"] = _FlowList(checkpoint.cites)
    return result


def _clean_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _checkpoints_yaml(checkpoints: list[Checkpoint], scenario: Scenario) -> str:
    source = {item.id: item for item in scenario.checkpoints}
    payload = [_checkpoint_dict(item, source[item.id]) for item in checkpoints]
    dumped = yaml.dump(
        {"checkpoints": payload},
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    return dumped


def _replace_checkpoints(original: str, block: str) -> str:
    lines = original.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("checkpoints:")]
    if len(starts) != 1:
        raise ReviewError("scenario must have exactly one top-level checkpoints key")
    start = starts[0]
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith((" ", "#", "\n", "\r")):
            raise ReviewError("checkpoints must be the last top-level key")
    return "".join(lines[:start]) + block


def _read_disagreements(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ReviewError(f"{path} must contain a YAML list")
    return raw


def apply_scenario(
    scenario_id: str,
    *,
    annotations_dir: Path = ANNOTATIONS_DIR,
    scenarios_dir: Path = SCENARIOS_DIR,
) -> tuple[int, int]:
    """Apply one completed review and return checkpoint/disagreement counts."""
    model_path = annotations_dir / "model" / f"{scenario_id}.yaml"
    review_path = annotations_dir / "review" / f"{scenario_id}.yaml"
    fixture_path = scenarios_dir / f"{scenario_id}.yaml"
    model_doc = _load_mapping(model_path)
    review_doc = _load_mapping(review_path)
    if review_doc.get("reviewed") is not True:
        raise ReviewError(f"{scenario_id}: review is not marked reviewed: true")
    if model_doc.get("scenario") != scenario_id or review_doc.get("scenario") != scenario_id:
        raise ReviewError(f"{scenario_id}: annotation scenario id does not match")
    try:
        scenario = load_scenario(fixture_path)
    except (ValueError, OSError) as exc:
        raise ReviewError(str(exc)) from exc
    citable = sorted(key for key, checked in guideline_clauses().items() if checked)
    review_checkpoints = review_doc.get("checkpoints")
    data = {"checkpoints": review_checkpoints}
    labelled, problems = validate_annotation(scenario, data, citable)
    if problems:
        raise ReviewError(f"{scenario_id}: " + "; ".join(problems))

    model_items = model_doc.get("checkpoints")
    if not isinstance(model_items, list) or not isinstance(review_checkpoints, list):
        raise ReviewError(f"{scenario_id}: checkpoints must be lists")
    model_by_id = {item.get("id"): item for item in model_items if isinstance(item, dict)}
    review_by_id = {item.get("id"): item for item in review_checkpoints if isinstance(item, dict)}
    disagreements = []
    for checkpoint in labelled:
        model_item = model_by_id.get(checkpoint.id)
        human_item = review_by_id[checkpoint.id]
        if not isinstance(model_item, dict):
            raise ReviewError(f"{scenario_id}: model annotation lacks {checkpoint.id}")
        if not _different(model_item, human_item):
            continue
        reason = human_item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ReviewError(f"{scenario_id}/{checkpoint.id}: changed labels need a reason")
        disagreements.append(
            {
                "scenario": scenario_id,
                "checkpoint": checkpoint.id,
                "date": date.today().isoformat(),
                "model": _labels(model_item),
                "human": _labels(human_item),
                "reason": reason,
            }
        )

    disagreement_path = annotations_dir / "disagreements.yaml"
    existing = [
        item
        for item in _read_disagreements(disagreement_path)
        if item.get("scenario") != scenario_id
    ]

    original = fixture_path.read_text(encoding="utf-8")
    rewritten = _replace_checkpoints(original, _checkpoints_yaml(labelled, scenario))
    fixture_path.write_text(rewritten, encoding="utf-8")
    try:
        loaded = load_scenario(fixture_path)
        cite_errors = citation_problems([loaded], guideline_clauses())
        if cite_errors:
            raise ReviewError("; ".join(cite_errors))
    except Exception as exc:
        fixture_path.write_text(original, encoding="utf-8")
        if isinstance(exc, ReviewError):
            raise
        raise ReviewError(f"{scenario_id}: rewritten fixture failed validation: {exc}") from exc

    # Written even when empty: an empty log after apply records that the
    # review happened and the human accepted every model label.
    updated = existing + disagreements
    disagreement_path.parent.mkdir(parents=True, exist_ok=True)
    disagreement_path.write_text(
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return len(labelled), len(disagreements)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply completed human annotation reviews.")
    parser.add_argument("ids", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    for scenario_id in args.ids:
        try:
            applied, disagreements = apply_scenario(scenario_id)
        except (ReviewError, AnnotatorError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{scenario_id}: {applied} checkpoints applied, {disagreements} disagreements logged")
    return 0
