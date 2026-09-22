"""Compare local first-pass annotations with two independent Opus runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import yaml

from decision_bench.annotate import ANNOTATIONS_DIR
from decision_bench.triage import scenario_flags


def _checkpoints(run: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        str(item["id"]): item
        for item in run.get("checkpoints") or ()
        if isinstance(item, Mapping) and "id" in item
    }


def _pair_flags(first: Mapping[str, object], second: Mapping[str, object]) -> dict[str, list[str]]:
    """Use the production tiebreak triage rule for a pair of runs."""
    return scenario_flags(
        {
            "checkpoints": list(first.get("checkpoints") or ()),
            "second_opinion": {"checkpoints": list(second.get("checkpoints") or ())},
        }
    )


def _actions(checkpoint: Mapping[str, object]) -> set[tuple[str, str]]:
    return {
        (str(kind), str(value))
        for field in ("acceptable", "must_not")
        for action in checkpoint.get(field) or ()
        if isinstance(action, Mapping)
        for kind, value in action.items()
    }


def _placement(checkpoint: Mapping[str, object], action: tuple[str, str]) -> str:
    for field in ("acceptable", "must_not"):
        if action in {
            (str(kind), str(value))
            for item in checkpoint.get(field) or ()
            if isinstance(item, Mapping)
            for kind, value in item.items()
        }:
            return field
    return "absent"


def _ratio(matches: int, eligible: int) -> dict[str, int | float | None]:
    return {
        "matches": matches,
        "eligible": eligible,
        "rate": matches / eligible if eligible else None,
    }


def _failure_list(metadata: Mapping[str, object]) -> list[str]:
    failures = metadata.get("parse_validation_failures", [])
    if isinstance(failures, list):
        return [str(item) for item in failures]
    if isinstance(failures, int):
        return ["unspecified parse/validation failure"] * failures
    return []


def calibrate(local_dir: Path, model_dir: Path) -> dict[str, object]:
    """Return full calibration detail for local files that have two Opus runs."""
    scenarios: list[dict[str, object]] = []
    action_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    placement_matches = placement_eligible = 0
    escalate_matches = escalate_eligible = 0
    trigger_matches = trigger_eligible = 0

    for local_path in sorted(local_dir.glob("*.yaml")):
        opus_path = model_dir / local_path.name
        if not opus_path.exists():
            continue
        local = yaml.safe_load(local_path.read_text(encoding="utf-8"))
        opus = yaml.safe_load(opus_path.read_text(encoding="utf-8"))
        if not isinstance(local, Mapping) or not isinstance(opus, Mapping):
            continue
        second = opus.get("second_opinion")
        if not isinstance(second, Mapping) or not isinstance(second.get("checkpoints"), list):
            continue

        opus1_flags = _pair_flags(opus, second)
        local_opus1_flags = _pair_flags(opus, local)
        opus2_local_flags = _pair_flags(second, local)
        opus1 = _checkpoints(opus)
        opus2 = _checkpoints(second)
        local_checkpoints = _checkpoints(local)
        checkpoint_details: dict[str, object] = {}

        for checkpoint_id in sorted(opus1.keys() & opus2.keys()):
            first = opus1[checkpoint_id]
            second_checkpoint = opus2[checkpoint_id]
            local_checkpoint = local_checkpoints.get(checkpoint_id, {})
            action_details = []
            for action in sorted(
                _actions(first) | _actions(second_checkpoint) | _actions(local_checkpoint)
            ):
                first_place = _placement(first, action)
                second_place = _placement(second_checkpoint, action)
                local_place = _placement(local_checkpoint, action)
                if first_place != second_place:
                    continue
                matches = local_place == first_place
                placement_eligible += 1
                placement_matches += matches
                action_key = f"{action[0]}: {action[1]}"
                action_totals[action_key][1] += 1
                action_totals[action_key][0] += matches
                action_details.append(
                    {
                        "action": {action[0]: action[1]},
                        "opus_placement": first_place,
                        "local_placement": local_place,
                        "agree": matches,
                    }
                )

            first_escalates = first.get("escalate_by") is not None
            second_escalates = second_checkpoint.get("escalate_by") is not None
            escalate_detail = None
            if first_escalates == second_escalates:
                escalate_eligible += 1
                matches = (local_checkpoint.get("escalate_by") is not None) == first_escalates
                escalate_matches += matches
                escalate_detail = {
                    "opus_present": first_escalates,
                    "local_present": local_checkpoint.get("escalate_by") is not None,
                    "agree": matches,
                }

            trigger_detail = None
            if (
                first_escalates
                and second_escalates
                and first.get("trigger") == second_checkpoint.get("trigger")
            ):
                trigger_eligible += 1
                matches = local_checkpoint.get("trigger") == first.get("trigger")
                trigger_matches += matches
                trigger_detail = {
                    "opus": first.get("trigger"),
                    "local": local_checkpoint.get("trigger"),
                    "agree": matches,
                }
            checkpoint_details[checkpoint_id] = {
                "actions": action_details,
                "escalate_by_presence": escalate_detail,
                "trigger": trigger_detail,
            }

        metadata = local.get("annotator")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        harmful = not local_opus1_flags and bool(opus1_flags) and bool(opus2_local_flags)
        scenarios.append(
            {
                "scenario": str(local.get("scenario") or local_path.stem),
                "opus_agree": not opus1_flags,
                "local_vs_opus1_agree": not local_opus1_flags,
                "local_vs_opus2_agree": not opus2_local_flags,
                "opus_disagreements": opus1_flags,
                "local_vs_opus1_disagreements": local_opus1_flags,
                "runtime_seconds": float(metadata.get("runtime_seconds", 0.0)),
                "parse_validation_failures": _failure_list(metadata),
                "harmful_local_agreement": harmful,
                "checkpoints": checkpoint_details,
            }
        )

    count = len(scenarios)
    opus_agree = sum(bool(item["opus_agree"]) for item in scenarios)
    local_agree = sum(bool(item["local_vs_opus1_agree"]) for item in scenarios)
    harmful = [item["scenario"] for item in scenarios if item["harmful_local_agreement"]]
    return {
        "local_dir": str(local_dir),
        "model_dir": str(model_dir),
        "scenarios": scenarios,
        "totals": {
            "scenarios": count,
            "opus_agree": {"count": opus_agree, "rate": opus_agree / count if count else None},
            "local_vs_opus1_agree": {
                "count": local_agree,
                "rate": local_agree / count if count else None,
            },
            "action_placement": _ratio(placement_matches, placement_eligible),
            "action_placement_by_action": {
                action: _ratio(values[0], values[1])
                for action, values in sorted(action_totals.items())
            },
            "escalate_by_presence": _ratio(escalate_matches, escalate_eligible),
            "trigger": _ratio(trigger_matches, trigger_eligible),
            "runtime_seconds": sum(float(item["runtime_seconds"]) for item in scenarios),
            "parse_validation_failures": sum(
                len(item["parse_validation_failures"]) for item in scenarios
            ),
            "harmful_local_agreements": harmful,
        },
    }


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def print_summary(report: Mapping[str, object]) -> None:
    scenarios = report["scenarios"]
    totals = report["totals"]
    assert isinstance(scenarios, list) and isinstance(totals, Mapping)
    print("scenario                         opus  local+opus1  runtime  failures")
    print("--------------------------------  ----  -----------  -------  --------")
    for item in scenarios:
        assert isinstance(item, Mapping)
        print(
            f"{str(item['scenario']):32}  "
            f"{'yes' if item['opus_agree'] else 'no ':4}  "
            f"{'yes' if item['local_vs_opus1_agree'] else 'no ':11}  "
            f"{float(item['runtime_seconds']):6.1f}s  "
            f"{len(item['parse_validation_failures']):8d}"
        )
    opus = totals["opus_agree"]
    local = totals["local_vs_opus1_agree"]
    actions = totals["action_placement"]
    escalate = totals["escalate_by_presence"]
    trigger = totals["trigger"]
    print()
    print(
        f"Opus run agreement:       {opus['count']}/{totals['scenarios']} "
        f"({_percent(opus['rate'])})"
    )
    print(
        f"Local + Opus 1 agreement: {local['count']}/{totals['scenarios']} "
        f"({_percent(local['rate'])})  <- second Opus calls skipped"
    )
    print(
        f"Action placement:         {actions['matches']}/{actions['eligible']} "
        f"({_percent(actions['rate'])})"
    )
    print(
        f"Escalate presence:        {escalate['matches']}/{escalate['eligible']} "
        f"({_percent(escalate['rate'])})"
    )
    print(
        f"Trigger:                  {trigger['matches']}/{trigger['eligible']} "
        f"({_percent(trigger['rate'])})"
    )
    harmful = totals["harmful_local_agreements"]
    print(
        f"Harmful local agreements: {len(harmful)}"
        + (f" ({', '.join(harmful)})" if harmful else "")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=ANNOTATIONS_DIR / "local")
    parser.add_argument("--model-dir", type=Path, default=ANNOTATIONS_DIR / "model")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = calibrate(args.local_dir, args.model_dir)
    print_summary(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0
