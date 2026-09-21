"""Decide which annotator labels need a human before they are used.

A checkpoint is flagged when the annotator itself says `needs_review`, or when
an independent second annotator run disagrees on something that can make the
checkpoint critical: the `must_not` set, whether there is an escalation
deadline at all, or `acceptable` sets with nothing in common. Differences in
the deadline's number are not flagged: it is a caregiver placeholder either
way. Triage accepts only what it can check: a draft without the self-flag
or without a second opinion is always flagged. Unflagged checkpoints are
accepted as labelled by the model.
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


def checkpoint_flags(first: Mapping[str, object], second: Mapping[str, object] | None) -> list[str]:
    """Reasons a human should look at this checkpoint; empty when none."""
    reasons = []
    if "needs_review" not in first:
        reasons.append("the draft predates self-flagging, so triage cannot accept it")
    elif first.get("needs_review"):
        reasons.append("the annotator asked for review")
    if second is None:
        reasons.append("there is no second opinion to compare with")
        return reasons
    if second.get("needs_review"):
        reasons.append("the second opinion asked for review")
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


def scenario_flags(model_doc: Mapping[str, object]) -> dict[str, list[str]]:
    """Flag reasons per checkpoint id, from a model annotation document."""
    second_doc = model_doc.get("second_opinion") or {}
    second = {
        str(item["id"]): item
        for item in (second_doc.get("checkpoints") if isinstance(second_doc, Mapping) else ()) or ()
    }
    flags = {}
    for item in model_doc.get("checkpoints") or ():
        reasons = checkpoint_flags(item, second.get(str(item["id"])))
        if reasons:
            flags[str(item["id"])] = reasons
    return flags
