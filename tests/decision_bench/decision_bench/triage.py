"""Decide which annotator labels need a human before they are used.

A checkpoint is flagged when an independent second annotator run disagrees
on something that can make the checkpoint critical: the `must_not` set,
whether there is an escalation deadline at all, or `acceptable` sets with
nothing in common. Differences in the deadline's number are not flagged: it
is a caregiver placeholder either way. Self-flags (either run's own
`needs_review`) no longer block by themselves: since 2026-09-21 a checkpoint
with only a self-flag and no content disagreement is accepted (see
`annotate.py`'s two-agreeing-runs rule); its `uncertain` note is kept in the
model file for a human to read later, it just does not stop application.

Triage accepts only what it can check: a draft without the self-flag or
without a second opinion is always flagged, and `scenario_blocked` marks
these scenarios as unable to be tie-broken by a third run at all - they go
straight to a human. A checkpoint flagged for anything else has a content
disagreement and can be tie-broken by a third independent run and a
2-of-3 vote (see `vote.py`). Unflagged checkpoints are accepted as labelled
by the model.
"""

from __future__ import annotations

from collections.abc import Mapping


def _actions(checkpoint: Mapping[str, object], field: str) -> set[tuple[str, str]]:
    return {
        (str(kind), str(value))
        for action in checkpoint.get(field) or ()
        if isinstance(action, Mapping)
        for kind, value in action.items()
    }


def _describe(actions: set[tuple[str, str]]) -> str:
    return ", ".join(f"{kind}: {value}" for kind, value in sorted(actions)) or "none"


def checkpoint_blocked(first: Mapping[str, object], second: Mapping[str, object] | None) -> bool:
    """True when triage cannot check this checkpoint at all: no vote is possible,
    and it must go to a human directly rather than through a third run."""
    return "needs_review" not in first or second is None


def checkpoint_flags(first: Mapping[str, object], second: Mapping[str, object] | None) -> list[str]:
    """Reasons a human should look at this checkpoint; empty when none.

    Self-flags are not reasons: they no longer block by themselves.
    """
    reasons = []
    if "needs_review" not in first:
        reasons.append("the draft predates self-flagging, so triage cannot accept it")
    if second is None:
        reasons.append("there is no second opinion to compare with")
        return reasons
    first_not, second_not = _actions(first, "must_not"), _actions(second, "must_not")
    if first_not != second_not:
        reasons.append(
            "must_not differs: only first "
            f"[{_describe(first_not - second_not)}], only second "
            f"[{_describe(second_not - first_not)}]"
        )
    if (first.get("escalate_by") is None) != (second.get("escalate_by") is None):
        reasons.append("the runs disagree on whether the caregiver must be notified by a deadline")
    first_ok, second_ok = _actions(first, "acceptable"), _actions(second, "acceptable")
    if (first_ok or second_ok) and not first_ok & second_ok:
        reasons.append(
            f"acceptable sets share nothing: [{_describe(first_ok)}] vs [{_describe(second_ok)}]"
        )
    return reasons


def _second_by_id(model_doc: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    second_doc = model_doc.get("second_opinion") or {}
    return {
        str(item["id"]): item
        for item in (second_doc.get("checkpoints") if isinstance(second_doc, Mapping) else ()) or ()
    }


def scenario_flags(model_doc: Mapping[str, object]) -> dict[str, list[str]]:
    """Flag reasons per checkpoint id, from a model annotation document."""
    second = _second_by_id(model_doc)
    flags = {}
    for item in model_doc.get("checkpoints") or ():
        reasons = checkpoint_flags(item, second.get(str(item["id"])))
        if reasons:
            flags[str(item["id"])] = reasons
    return flags


def scenario_blocked(model_doc: Mapping[str, object]) -> bool:
    """True when some checkpoint cannot be tie-broken by a third run: a human
    must look regardless of what a vote would say."""
    second = _second_by_id(model_doc)
    return any(
        checkpoint_blocked(item, second.get(str(item["id"])))
        for item in model_doc.get("checkpoints") or ()
    )
