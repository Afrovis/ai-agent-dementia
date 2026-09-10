"""The deterministic rule layer for `agent`'s session state machine (issue #12).

This is the layer HANDOFF.md rule 1 is about: "The LLM never owns safety.
State transitions, escalation timers, and hard limits live in deterministic
code in the `agent` service. The LLM interprets, composes, and proposes.
Every LLM proposal passes through `rules.validate()` before it has any
effect." Nothing in this module calls an LLM, and nothing here is aware
that one exists. `interpret`, `compose`, and `plan` (issue #15) will each
eventually produce a *proposed* phase (via `plan`) or strategy choice; this
module is what stands between that proposal and anything actually
happening. `agent.session.Session` already routes every transition it
computes itself through `validate()` too, so this is not a dead file
waiting for #15 -- it is load-bearing today.

`ALLOWED_TRANSITIONS` is the transition table HANDOFF.md section 6
describes -- `IDLE -> OBSERVING -> ENGAGED -> COOLDOWN -> IDLE`, with
`ENGAGED -> ESCALATED -> COOLDOWN` and `OBSERVING -> IDLE` ("if back
`in_bed`, return to `IDLE`") -- plus every edge rule 5 needs. Rule 5
("`on_floor` or `absent` from the room beyond the configured limit skips
every strategy and goes straight to `escalate_phone`") has no "only if a
session is already live" qualifier and outranks section 6's phase summary,
so it can fire from *any* phase, including ones with no nudging session
running yet:

- `OBSERVING -> ESCALATED` and `ENGAGED -> ESCALATED`: rule 5 firing while
  a nudging session is already under way.
- `IDLE -> ESCALATED`: rule 5 firing with no session running at all, e.g.
  a fall straight from `in_bed` to `on_floor` with no intervening
  `sitting_up`/`standing` reading, or `on_floor`/`absent` outside the
  night window (rule 5 is not gated by the night window; see
  `agent.session.Session._rule5_transition`).
- `COOLDOWN -> ESCALATED`: rule 5 firing during cooldown, when no new
  *nudging* session may start but a fall still must not be ignored.

Beyond that, nothing else. In particular:

- `ESCALATED -> COOLDOWN` is allowed; `ESCALATED -> ENGAGED` is not. Once a
  session has escalated, the rule layer never lets it quietly go back to
  ordinary strategy-running.
- `COOLDOWN` only exits to `IDLE` or, via rule 5, to `ESCALATED`. No new
  *nudging* session may start during cooldown, so
  `COOLDOWN -> OBSERVING`/`ENGAGED` are absent from the table on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Phase(StrEnum):
    """The five session phases from HANDOFF.md section 6."""

    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    ENGAGED = "ENGAGED"
    COOLDOWN = "COOLDOWN"
    ESCALATED = "ESCALATED"


ALLOWED_TRANSITIONS: dict[Phase, frozenset[Phase]] = {
    # `ESCALATED` here is rule 5 firing with no nudging session running.
    Phase.IDLE: frozenset({Phase.OBSERVING, Phase.ESCALATED}),
    Phase.OBSERVING: frozenset({Phase.IDLE, Phase.ENGAGED, Phase.ESCALATED}),
    Phase.ENGAGED: frozenset({Phase.COOLDOWN, Phase.ESCALATED}),
    Phase.ESCALATED: frozenset({Phase.COOLDOWN}),
    # `ESCALATED` here is rule 5 firing during cooldown.
    Phase.COOLDOWN: frozenset({Phase.IDLE, Phase.ESCALATED}),
}


@dataclass(frozen=True)
class RuleResult:
    """The outcome of `validate()`: distinguishes an ordinary rejection
    (`accepted=False`, `reason` explaining why) from acceptance, without
    raising -- an LLM proposing an illegal transition, or a disabled
    strategy, is an everyday event this layer exists to handle quietly,
    not an exceptional one."""

    accepted: bool
    reason: str | None = None


def validate(
    current: Phase,
    proposed: Phase,
    *,
    strategy: str | None = None,
    strategy_disabled: bool = False,
) -> RuleResult:
    """Decide whether `current -> proposed` may happen at all.

    Rejects any transition not present in `ALLOWED_TRANSITIONS`, regardless
    of who is proposing it -- `agent.session.Session`'s own logic today, an
    LLM's `plan` output from issue #15 tomorrow.

    `strategy`/`strategy_disabled` are a seam for issue #14's strategy
    engine: once strategies have configured cooldowns and an enabled flag,
    a caller will pass the strategy a proposed transition would select and
    this function will reject the transition if that strategy is disabled
    or still on cooldown, exactly like an illegal phase transition. Neither
    parameter has any effect yet beyond the explicit check below, because
    issue #12 has no strategy engine to ask.
    """
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if proposed not in allowed:
        return RuleResult(
            accepted=False,
            reason=f"{current.value} -> {proposed.value} is not an allowed transition",
        )

    if strategy_disabled:
        # TODO(#14): the strategy engine will pass real cooldown/enabled
        # state here instead of a caller-supplied bool.
        return RuleResult(
            accepted=False,
            reason=f"strategy {strategy!r} is disabled or on cooldown",
        )

    return RuleResult(accepted=True, reason=None)
