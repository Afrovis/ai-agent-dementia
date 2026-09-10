"""The goal tree for `agent`'s session state machine (issue #13).

PLAN.md section 5.2 defines five goals as a small tree: `return_to_bed` is
the root and default, and `restroom`, `drink_water`, and `comfort` each
return to it once satisfied. `wait_for_caregiver` sits beside them with no
sub-goal of its own.

This module is deliberately just data: the set of valid goals, which goal
is the root, each non-root goal's parent (`PARENT_OF`), and the explicit
table of goal-to-goal changes the rule layer permits (`ALLOWED_GOAL_CHANGES`,
consumed by `agent.rules.validate_goal`). Modelling "return to parent goal"
as an explicit `PARENT_OF` mapping -- rather than hardcoding `return_to_bed`
at every call site that needs to switch back -- means a future issue can
give the tree real depth (e.g. a sub-goal under `restroom`) by editing this
table, not by rewriting `agent.session`.

## Success conditions, and what issue #13 can and cannot observe

PLAN.md section 5.2's "success condition" column is the target end state
for each goal. This issue implements the deterministic parts only; the
rest are explicitly left as seams for issue #15 (LLM `interpret`/`plan`)
or for hardware this system does not have. Per goal:

- `return_to_bed` (root, default): success is `in_bed` for
  `AgentConfig.in_bed_stable_seconds`. **Observable today** --
  `agent.session.Session._track_in_bed_stability` already implements
  exactly this (issue #12), independent of which goal is active.
- `restroom`: success is "went to the bathroom and came back, then
  `return_to_bed`". **Partially observable today.** Entry is observable:
  `PersonState.zone == "bathroom_path"` is PLAN.md's second documented
  trigger ("perception sees the person heading to the bathroom door").
  This module treats "came back" as observable too, approximated as
  "the person is later back at the bed" (`zone == "bed"` or
  `state == "in_bed"`) -- see `agent.session.Session` for exactly how.
  What is NOT observed, and not faked: whether the person actually used
  the bathroom while there. A configured timeout
  (`AgentConfig.restroom_timeout_seconds`) exists only so the goal cannot
  stick forever if the person never returns to the bed zone; it is a
  safety valve, not a claim of detecting bathroom use.
- `drink_water`: success is "drank, then `return_to_bed`". **Not
  observable today.** Nothing in `perceive`'s `PersonState` or `listen`'s
  bare `Utterance` arrival distinguishes "asked for water" or "drank" from
  any other moment. Entry requires STT plus intent classification
  (`interpret`, issue #15) or the LLM `plan` proposing it, which is why
  this goal exists in the tree and is a legal `validate_goal` target, but
  nothing in `agent.session` enters it deterministically.
- `comfort`: success is "distress reduced (calm voice, sitting still)".
  **Not observable today**, for the same reason issue #12's module
  docstring already gives for "distress detected twice": distress
  detection depends on `interpret` (issue #15), an LLM call this package
  must never depend on. Legal as a `validate_goal` target; never entered
  by deterministic code here.
- `wait_for_caregiver`: success is "caregiver present". **Not observable
  today, and will not be faked as observable.** The only signal available
  after an escalation is `Ack` (stream `ack`, produced by `dashboard`/
  `notify` when *a phone is acknowledged*), which this module and
  `agent.session` deliberately never consume for this purpose: someone
  tapping "acknowledge" on a phone is not the same fact as a caregiver
  being physically present in the room, and treating them as equivalent
  could resume nudging a person who is still on the floor. So entry is
  observable and implemented (rule 5 escalating sets this goal), but exit
  is not: the session ends the ordinary way instead, via `in_bed`
  stability into `COOLDOWN` (unchanged from issue #12), and the goal reset
  to root that happens on any return to `IDLE` is what eventually clears
  it, not a satisfied `wait_for_caregiver`.
"""

from __future__ import annotations

ROOT_GOAL = "return_to_bed"
"""The default goal, and the root of the tree. `agent.session.Session`
starts here and resets here on every return to `IDLE`."""

RESTROOM_GOAL = "restroom"
DRINK_WATER_GOAL = "drink_water"
COMFORT_GOAL = "comfort"
WAIT_FOR_CAREGIVER_GOAL = "wait_for_caregiver"

GOALS: frozenset[str] = frozenset(
    {ROOT_GOAL, RESTROOM_GOAL, DRINK_WATER_GOAL, COMFORT_GOAL, WAIT_FOR_CAREGIVER_GOAL}
)
"""Every goal PLAN.md section 5.2 defines. `agent.rules.validate_goal`
rejects anything outside this set as a matter of course (it will simply
never appear as a key in `ALLOWED_GOAL_CHANGES`)."""

PARENT_OF: dict[str, str] = {
    RESTROOM_GOAL: ROOT_GOAL,
    DRINK_WATER_GOAL: ROOT_GOAL,
    COMFORT_GOAL: ROOT_GOAL,
    WAIT_FOR_CAREGIVER_GOAL: ROOT_GOAL,
}
"""Each non-root goal's parent -- "return to parent goal" made explicit as
data instead of a hardcoded `ROOT_GOAL` at every switch-back call site.
Every current goal in this tree happens to have the same parent today, but
call sites should still look it up here (`PARENT_OF[goal]`), not assume
it, so a deeper tree later is a data change, not a rewrite.

`ROOT_GOAL` has no entry, being the root. `WAIT_FOR_CAREGIVER_GOAL`'s
entry exists so a goal reset on return to `IDLE` (see `agent.session`) has
somewhere defined to reset *to*, even though, per the module docstring,
nothing in this codebase currently drives a satisfied return from it --
the session instead ends the ordinary way, through `in_bed` stability."""

ALLOWED_GOAL_CHANGES: dict[str, frozenset[str]] = {
    ROOT_GOAL: frozenset({RESTROOM_GOAL, DRINK_WATER_GOAL, COMFORT_GOAL, WAIT_FOR_CAREGIVER_GOAL}),
    RESTROOM_GOAL: frozenset({ROOT_GOAL, WAIT_FOR_CAREGIVER_GOAL}),
    DRINK_WATER_GOAL: frozenset({ROOT_GOAL, WAIT_FOR_CAREGIVER_GOAL}),
    COMFORT_GOAL: frozenset({ROOT_GOAL, WAIT_FOR_CAREGIVER_GOAL}),
    WAIT_FOR_CAREGIVER_GOAL: frozenset({ROOT_GOAL}),
}
"""The goal-change equivalent of `agent.rules.ALLOWED_TRANSITIONS`: every
edge `agent.rules.validate_goal` accepts, keyed by current goal.

Shape, by row:

- From `ROOT_GOAL`: to any of the four non-root goals. This is how every
  sub-goal starts -- PLAN.md's three documented triggers (stated need,
  perception, LLM proposal) all switch away from the default.
- From each of `restroom`/`drink_water`/`comfort`: back to `ROOT_GOAL`
  (the ordinary "satisfied, return to parent" edge), or to
  `WAIT_FOR_CAREGIVER_GOAL`. The latter exists because HANDOFF.md rule 5
  can escalate from *any* phase regardless of which sub-goal is active --
  a person who said they needed the toilet and then fell still needs the
  caregiver called, goal notwithstanding.
- From `WAIT_FOR_CAREGIVER_GOAL`: only back to `ROOT_GOAL`. This is the
  edge the `IDLE` goal-reset uses; nothing in `agent.session` drives it
  for any other reason (see the module docstring on why "caregiver
  present" is not observed).
- No goal ever changes to itself: self-transitions are absent from every
  row on purpose, the same convention `agent.rules.ALLOWED_TRANSITIONS`
  uses for phases.
- No direct edges between the three non-root return-to-bed sub-goals
  (e.g. `restroom` -> `drink_water`): a switch between them goes through
  `ROOT_GOAL`, keeping "return to parent, then to a new sub-goal" as the
  one path, rather than an ad hoc shortcut per pair.
"""
