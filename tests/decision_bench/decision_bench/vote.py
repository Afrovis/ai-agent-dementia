"""2-of-3 majority vote over three independent annotator runs.

Used when two independent runs disagree on a checkpoint's content
(`triage.checkpoint_flags`, and the disagreement is not one `triage` refuses
to check at all): a third independent run is made and the three are voted
per checkpoint, per action. An action two of the three runs agree on becomes
the result; an action only one run set, and that did not reach a majority in
either direction, becomes `doubtful_acceptable` or `doubtful_must_not`
instead of being silently dropped, since scoring gives it half credit rather
than none (see `scoring.py`). A checkpoint with no majority anywhere it
matters (its accepted actions, or its escalation timing) is left for a human.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

RUN_NAMES = ("annotator", "second_opinion", "third_opinion")


def _action_set(checkpoint: Mapping[str, object], field_name: str) -> set[tuple[str, str]]:
    return {
        (str(kind), str(value))
        for action in checkpoint.get(field_name) or ()
        if isinstance(action, Mapping)
        for kind, value in action.items()
    }


def _action_dict(action: tuple[str, str]) -> dict[str, str]:
    kind, value = action
    return {kind: value}


def _lower_median(values: Sequence[float]) -> float:
    """The median of an odd count, or the smaller of an even count."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else ordered[middle - 1]


@dataclass(frozen=True)
class Vote:
    """The voted outcome for one checkpoint across up to three runs."""

    acceptable: list[dict[str, str]]
    must_not: list[dict[str, str]]
    doubtful_acceptable: list[dict[str, str]]
    doubtful_must_not: list[dict[str, str]]
    escalate_by: float | None
    trigger: float | None
    rationale: str
    cites: list[str]
    record: dict[str, object]
    no_majority: list[str] = field(default_factory=list)


def vote_checkpoint(runs: Sequence[Mapping[str, object] | None]) -> Vote:
    """Vote one checkpoint over its (up to three) independent runs.

    `runs` is ordered `(annotator, second_opinion, third_opinion)`; a `None`
    means that run had no checkpoint with this id.
    """
    present = [run for run in runs if run is not None]
    if len(present) < 2:
        return Vote([], [], [], [], None, None, "", [], {}, ["checkpoint missing from 2+ runs"])

    acceptable_sets = [_action_set(run, "acceptable") for run in present]
    must_not_sets = [_action_set(run, "must_not") for run in present]
    actions = set().union(*acceptable_sets, *must_not_sets)

    a_count = {action: sum(action in s for s in acceptable_sets) for action in actions}
    m_count = {action: sum(action in s for s in must_not_sets) for action in actions}

    acceptable = sorted(a for a in actions if a_count[a] >= 2)
    must_not = sorted(a for a in actions if a_count[a] < 2 and m_count[a] >= 2)
    doubtful_acceptable = sorted(
        a for a in actions if a_count[a] < 2 and m_count[a] < 2 and a_count[a] == 1
    )
    doubtful_must_not = sorted(
        a for a in actions if a_count[a] < 2 and m_count[a] < 2 and m_count[a] == 1
    )

    no_majority = []
    empty_acceptable_runs = sum(1 for s in acceptable_sets if not s)
    if not acceptable and empty_acceptable_runs < 2:
        # A single run proposing something the others left empty is just doubtful: two
        # of three agreeing that nothing is needed is itself a majority. It is only a
        # real disagreement, with no majority at all, when fewer than two runs agree on
        # emptiness and no single action reached two votes either.
        no_majority.append("no majority on acceptable")

    escalate_values = [run.get("escalate_by") for run in present if run.get("escalate_by") is not None]
    escalate_by = _lower_median(escalate_values) if len(escalate_values) >= 2 else None

    trigger = None
    if escalate_by is not None:
        contributing = [run for run in present if run.get("escalate_by") is not None]
        trigger_counts = Counter(run.get("trigger") for run in contributing)
        top_value, top_count = trigger_counts.most_common(1)[0]
        if top_count >= 2:
            trigger = top_value
        else:
            no_majority.append("no majority on trigger")

    first = present[0]
    record = {
        "acceptable_votes": {f"{k}: {v}": a_count[(k, v)] for k, v in sorted(actions) if a_count[(k, v)]},
        "must_not_votes": {f"{k}: {v}": m_count[(k, v)] for k, v in sorted(actions) if m_count[(k, v)]},
        "escalate_by_set_by": [
            RUN_NAMES[i] for i, run in enumerate(runs) if run is not None and run.get("escalate_by") is not None
        ],
    }

    return Vote(
        acceptable=[_action_dict(a) for a in acceptable],
        must_not=[_action_dict(a) for a in must_not],
        doubtful_acceptable=[_action_dict(a) for a in doubtful_acceptable],
        doubtful_must_not=[_action_dict(a) for a in doubtful_must_not],
        escalate_by=escalate_by,
        trigger=trigger,
        rationale=str(first.get("rationale") or ""),
        cites=list(first.get("cites") or []),
        record=record,
        no_majority=no_majority,
    )


def vote_scenario(model_doc: Mapping[str, object]) -> dict[str, Vote]:
    """Vote every checkpoint in a model annotation document that has a
    `third_opinion`, using `(annotator, second_opinion, third_opinion)`."""
    first = {str(item["id"]): item for item in model_doc.get("checkpoints") or ()}
    second_doc = model_doc.get("second_opinion") or {}
    second = {
        str(item["id"]): item
        for item in (second_doc.get("checkpoints") if isinstance(second_doc, Mapping) else ()) or ()
    }
    third_doc = model_doc.get("third_opinion") or {}
    third = {
        str(item["id"]): item
        for item in (third_doc.get("checkpoints") if isinstance(third_doc, Mapping) else ()) or ()
    }
    return {
        checkpoint_id: vote_checkpoint((run, second.get(checkpoint_id), third.get(checkpoint_id)))
        for checkpoint_id, run in first.items()
    }
