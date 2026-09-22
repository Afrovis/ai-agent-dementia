"""Tie-break existing two-run annotations with a third opinion.

`python -m decision_bench annotate --tiebreak` looks at scenarios that
already have a model draft and are not yet reviewed, without re-running the
first two Opus calls:

- scenarios whose two runs agree (`triage.checkpoint_flags` empty on every
  checkpoint) are applied straight away, no Opus call at all;
- scenarios `triage.scenario_blocked` cannot check (no second opinion, or a
  draft that predates self-flagging) are left for a human, as before;
- everything else gets exactly one more independent run, stored as
  `third_opinion`, and a 2-of-3 vote (`vote.py`) decides each checkpoint.

Every write to `annotations/model`, `annotations/review` or the fixture
happens serially in the main thread; only the Opus calls run in
`--jobs` threads.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from decision_bench.annotate import (
    AnnotatorError,
    Runner,
    annotate_scenario,
    citable_clauses,
    dump_yaml,
    opinion_doc,
    review_text,
    run_claude,
    status_flags,
)
from decision_bench.schema import GUIDELINES_PATH, load_default_profile, load_scenarios
from decision_bench.triage import scenario_blocked, scenario_flags
from decision_bench.vote import vote_scenario


def _reason_word(reason: str) -> str:
    prefix = "no majority on "
    return reason[len(prefix) :] if reason.startswith(prefix) else "missing"


def tiebreak_main(args: argparse.Namespace, *, runner: Runner = run_claude) -> int:
    scenarios = load_scenarios(args.scenarios)
    by_id = {scenario.id: scenario for scenario in scenarios}
    if args.scenario_ids:
        ids = list(args.scenario_ids)
        missing = sorted(set(ids) - set(by_id))
        if missing:
            raise SystemExit(f"unknown scenario(s): {', '.join(missing)}")
    else:
        model_dir = args.annotations / "model"
        ids = sorted(path.stem for path in model_dir.glob("*.yaml")) if model_dir.exists() else []

    profile = load_default_profile(args.profile)
    guidelines_text = GUIDELINES_PATH.read_text(encoding="utf-8")
    citable = citable_clauses()

    to_vote: list[tuple] = []
    to_apply: list[tuple] = []
    for scenario_id in ids:
        scenario = by_id.get(scenario_id)
        if scenario is None:
            raise SystemExit(f"unknown scenario: {scenario_id}")
        model_path = args.annotations / "model" / f"{scenario_id}.yaml"
        if not model_path.exists():
            raise SystemExit(f"no annotation draft for {scenario_id}: {model_path}")
        model_doc = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        review_path = args.annotations / "review" / f"{scenario_id}.yaml"
        review_doc = (
            yaml.safe_load(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
        )
        if isinstance(review_doc, dict) and review_doc.get("reviewed") is True and not args.force:
            print(f"{scenario_id}: already reviewed, skipped")
            continue
        if model_doc.get("third_opinion") is not None and not args.force:
            print(f"{scenario_id}: already has a third opinion, skipped")
            continue
        if scenario_flags(model_doc):
            if scenario_blocked(model_doc):
                print(f"{scenario_id}: blocked, cannot tie-break -> {review_path}")
                continue
            to_vote.append((scenario, model_doc, model_path, review_path))
        else:
            to_apply.append((scenario, model_doc, model_path, review_path))

    if args.dry_run:
        for scenario, _, _, _ in to_apply:
            print(f"{scenario.id}: agreed, would apply without a third run")
        for scenario, _, _, _ in to_vote:
            print(f"{scenario.id}: would run a third opinion and vote")
        return 0

    jobs = max(1, getattr(args, "jobs", 1))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            scenario.id: executor.submit(
                annotate_scenario,
                scenario,
                profile=profile | scenario.profile,
                guidelines_text=guidelines_text,
                citable=citable,
                model=args.model,
                effort=args.effort,
                runner=runner,
            )
            for scenario, _, _, _ in to_vote
        }

        for scenario, model_doc, model_path, review_path in to_apply:
            # The review file on disk may predate the two-agreeing-runs rule and
            # still say `reviewed: false`; rebuild it so `apply_scenario` sees the
            # current triage outcome.
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(review_text(scenario, model_doc), encoding="utf-8")
            _apply_result(scenario, model_doc, model_path, review_path, args)
            print(f"{scenario.id}: agreed -> applied (model)")

        for scenario, model_doc, model_path, review_path in to_vote:
            try:
                third = futures[scenario.id].result()
            except AnnotatorError as exc:
                raise SystemExit(str(exc)) from exc
            model_doc["third_opinion"] = opinion_doc(third)
            model_path.write_text(dump_yaml(model_doc), encoding="utf-8")
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(review_text(scenario, model_doc), encoding="utf-8")
            flags = status_flags(model_doc)
            if flags:
                where = ", ".join(f"{cid} ({_reason_word(flags[cid][0])})" for cid in flags)
                print(f"{scenario.id}: 3rd run, no majority on {where} -> {review_path}")
                continue
            votes = vote_scenario(model_doc)
            doubtful = sum(
                1 for vote in votes.values() if vote.doubtful_acceptable or vote.doubtful_must_not
            )
            _apply_result(scenario, model_doc, model_path, review_path, args)
            note = f", {doubtful} doubtful" if doubtful else ""
            print(f"{scenario.id}: 3rd run, voted -> applied (model){note}")
    return 0


def _apply_result(
    scenario, model_doc: dict[str, object], model_path: Path, review_path: Path, args
) -> None:
    if getattr(args, "no_apply", False):
        return
    from decision_bench.review import ReviewError, apply_scenario

    try:
        apply_scenario(scenario.id, annotations_dir=args.annotations, scenarios_dir=args.scenarios)
    except ReviewError as exc:
        raise SystemExit(str(exc)) from exc
