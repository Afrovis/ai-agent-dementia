"""The session phase state machine for `agent` (issue #12, extended in #13).

Pure and time-injectable: every method takes `now` -- a local, naive
`datetime`, since `AgentConfig`'s night window is expressed in local
`HH:MM` -- as an explicit parameter instead of reading the wall clock
itself, matching `capture.gate.MotionGate`/`perceive.classify.StateTracker`'s
injected-clock convention. Tests build `datetime`s directly, so they are
deterministic and never sleep for real.

`Session` consumes `PersonState.state`/`.zone` and bare `Utterance`
arrivals (call sites just need to know "an utterance happened", not its
text or confidence -- interpreting *what* was said is `interpret`, issue
#15) and produces `Transition`s: phase changes for `agent.main` to publish
as `SessionState`, an optional `Notify` alongside a move into `ESCALATED`,
and now (issue #13) an optional goal change for `agent.main` to publish as
`GoalChanged`, with or without an accompanying phase change.

Every phase change this class computes is run through `agent.rules.validate`
before being applied, and every goal change through `agent.rules.
validate_goal`, including changes this file itself decides on -- see the
module docstring in `agent.rules` for why that is not redundant.

Issue #13's goal tree (`agent.goals`) and what it can and cannot observe:

- Entering `restroom`: `PersonState.zone == "bathroom_path"` while
  `ENGAGED` with the root goal active. PLAN.md's second documented
  trigger ("perception sees the person heading to the bathroom door").
- Leaving `restroom`: the person is later seen back at the bed
  (`zone == "bed"` or `state == "in_bed"`), or `config.
  restroom_timeout_seconds` elapses with no such reading -- a safety
  valve, not a claim of detecting bathroom use. Either way, back to the
  parent goal (`return_to_bed`).
- Entering `wait_for_caregiver`: on rule 5 escalating into `ESCALATED`,
  from any prior goal. PLAN.md's success condition for this goal --
  "caregiver present" -- is NOT observable by this system today (see
  `agent.goals`'s module docstring for why `Ack` is deliberately not used
  as a stand-in), so this goal is entered but never satisfied by code
  here; the session ends the ordinary way, via `in_bed` stability into
  `COOLDOWN`, same as issue #12.
- `drink_water` and `comfort`: legal `validate_goal` targets, defined in
  the tree, but nothing in this file enters them. Both need `interpret`/
  `plan` (issue #15), an LLM call this module must never depend on.
- A stated need ("I need the toilet"): needs STT plus intent
  classification (issue #15). `on_utterance` still only knows that an
  utterance happened, not what was said, exactly as issue #12 left it.
- `propose_goal()` is the one public entry point for an LLM's `plan`
  proposal (issue #15) to go through: it runs the same
  `agent.rules.validate_goal` check as every deterministic switch above,
  and returns `None` -- not an exception -- on an ordinary rejection.

One remaining fixed seam for a later issue:

- `strategy_index` never leaves `0`. The strategy engine -- ordering,
  cooldowns, dwell times, and "strategies exhausted" as an escalation
  trigger distinct from rule 5 -- is issue #14. "Strategies exhausted" is
  therefore not implemented anywhere in this module; the only escalation
  trigger implemented is rule 5.

Also out of scope, and not faked: distress detection ("distress detected
twice" in HANDOFF.md section 6) does not exist here at all, because it
depends on `interpret` (issue #15), which makes an LLM call this module
must never depend on.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime

from agent.config import AgentConfig
from agent.goals import RESTROOM_GOAL, ROOT_GOAL, WAIT_FOR_CAREGIVER_GOAL
from agent.rules import Phase, validate, validate_goal

DEFAULT_GOAL = ROOT_GOAL


def _new_session_id() -> str:
    """A short, opaque session id. Not a UUID's full length: `SessionState`
    is logged and displayed, and a short id is friendlier for both."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class NotifySpec:
    """What `agent.main` should publish as `Notify` alongside a `Transition`
    into `ESCALATED`. `None` on every other transition."""

    level: str
    title: str
    body: str


@dataclass(frozen=True)
class GoalChangeResult:
    """One goal change `Session` wants published as `GoalChanged`. Carries
    exactly the fields that event needs (`from_goal`, `to_goal`, `reason`)
    plus the `session_id` it happened under, so `agent.main` does not have
    to reach back into `Session` for it."""

    session_id: str | None
    from_goal: str
    to_goal: str
    reason: str


@dataclass(frozen=True)
class Transition:
    """One update `Session` wants published: the current phase and goal
    (`SessionState` is republished whenever either changes), the `Notify`
    to publish alongside it if any, an optional `GoalChangeResult` if the
    goal changed on this update, and a short `reason` for logging (never
    published; not part of any event schema).

    `phase` is always the phase *after* this update, whether or not it
    changed on this call -- a goal-only change (e.g. entering `restroom`
    from `bathroom_path`) still needs a `Transition` to carry the goal
    change, and reusing the same shape (rather than a second, parallel
    return type) is what lets `agent.main` publish both a `SessionState`
    and a `GoalChanged` from one result without a second code path."""

    phase: Phase
    session_id: str | None
    goal: str
    strategy_index: int
    reason: str
    notify: NotifySpec | None = None
    goal_change: GoalChangeResult | None = None


@dataclass
class Session:
    """One instance tracks one ongoing (or currently absent) session.
    `agent.main` owns a single long-lived instance and feeds it every
    `PersonState` and `Utterance` off the bus, plus a periodic `tick` so
    timers advance even when nothing new has arrived."""

    config: AgentConfig
    id_fn: Callable[[], str] = _new_session_id

    phase: Phase = Phase.IDLE
    session_id: str | None = None
    goal: str = DEFAULT_GOAL
    strategy_index: int = 0

    # Rule 5's timers: when the current unbroken run of `on_floor`/`absent`
    # began, or `None` while the person is in neither state right now.
    _on_floor_since: datetime | None = field(default=None, init=False, repr=False)
    _absent_since: datetime | None = field(default=None, init=False, repr=False)

    # `OBSERVING`'s 20 s timer.
    _observing_since: datetime | None = field(default=None, init=False, repr=False)

    # `in_bed` stability tracking (drives entry into `COOLDOWN`) and when
    # `COOLDOWN` itself started (drives the return to `IDLE`).
    _in_bed_since: datetime | None = field(default=None, init=False, repr=False)
    _cooldown_since: datetime | None = field(default=None, init=False, repr=False)

    # When the `restroom` goal was entered, or `None` while it is not
    # active. Drives `config.restroom_timeout_seconds` (issue #13).
    _restroom_since: datetime | None = field(default=None, init=False, repr=False)

    # Consecutive-reading hysteresis on `PersonState.zone` for zone-driven
    # goal switching, mirroring `perceive.classify.StateTracker`'s
    # `confirm_frames` pattern for `state` -- see `_confirmed_zone`.
    _pending_zone: str | None = field(default=None, init=False, repr=False)
    _pending_zone_count: int = field(default=0, init=False, repr=False)

    def _confirmed_zone(self, zone: str) -> str | None:
        """Consecutive-reading hysteresis for `PersonState.zone`, analogous
        to `perceive.classify.StateTracker.confirm_frames`'s treatment of
        `state`: `perceive` puts no hysteresis on `zone` itself (it comes
        straight from a centroid-to-polygon lookup on whatever frame was
        just classified), and `bathroom_path`/`bed` sit right next to each
        other -- exactly where someone stands at the start of a bathroom
        trip -- so a single noisy reading must not be enough to flip a
        goal and spam the caregiver's timeline with `GoalChanged` events.

        Returns `zone` once it has repeated, unbroken, across
        `config.zone_confirm_readings` consecutive calls (i.e. consecutive
        `PersonState` events while a goal-switch-eligible phase is active),
        or `None` while still waiting. Deliberately not used anywhere near
        rule 5: rule 5 fires on `state`, not `zone`, and must keep firing
        on the very first reading regardless.
        """
        if zone == self._pending_zone:
            self._pending_zone_count += 1
        else:
            self._pending_zone = zone
            self._pending_zone_count = 1
        if self._pending_zone_count >= self.config.zone_confirm_readings:
            return zone
        return None

    def _apply_goal(
        self, target_goal: str, *, reason: str, must_be_legal: bool = False
    ) -> GoalChangeResult | None:
        """Validate and apply `self.goal -> target_goal`, returning the
        `GoalChangeResult` to attach to a `Transition`, or `None` if the
        goal is already `target_goal` (not a change, so nothing to report)
        or `agent.rules.validate_goal` rejects it.

        `must_be_legal=True` is for this file's own forced switches (the
        `IDLE` goal reset, `wait_for_caregiver` on escalation) -- edges
        that only exist because `agent.goals.ALLOWED_GOAL_CHANGES` was
        built to permit them, so a rejection here would mean the table and
        this file disagree, a bug rather than an ordinary rejection. Every
        other caller (zone-triggered `restroom` switches, `propose_goal`)
        leaves it `False`: rejection there is a routine, expected outcome,
        not a bug, exactly like `agent.rules.validate`'s own convention.
        """
        if target_goal == self.goal:
            return None
        result = validate_goal(self.goal, target_goal)
        if not result.accepted:
            if must_be_legal:
                raise AssertionError(
                    f"session logic proposed an illegal goal change: {result.reason}"
                )
            return None
        old_goal = self.goal
        self.goal = target_goal
        return GoalChangeResult(
            session_id=self.session_id, from_goal=old_goal, to_goal=target_goal, reason=reason
        )

    def _apply(self, target: Phase, *, reason: str, notify: NotifySpec | None = None) -> Transition:
        """Validate and apply `self.phase -> target`, returning the `Transition`
        to publish.

        Every call site below only proposes edges that exist in
        `agent.rules.ALLOWED_TRANSITIONS`, so `validate` rejecting one here
        would mean this file's own logic disagrees with the table it is
        built against -- a bug, not an ordinary rejection, hence the raise
        rather than silently swallowing it. See `agent.rules`'s module
        docstring: routing even our own transitions through `validate` is
        the point, not an accident.
        """
        result = validate(self.phase, target)
        if not result.accepted:
            raise AssertionError(f"session logic proposed an illegal transition: {result.reason}")

        if target == Phase.IDLE:
            session_id = None
        elif self.session_id is None:
            session_id = self.id_fn()
        else:
            session_id = self.session_id

        self.phase = target
        self.session_id = session_id

        goal_change: GoalChangeResult | None = None
        if target == Phase.IDLE:
            # Goal resets to the root on every return to `IDLE`. Routed
            # through `_apply_goal` like any other goal change (issue #13);
            # it is a no-op returning `None` -- not a spurious
            # `GoalChanged` -- for a session that never left the root goal.
            goal_change = self._apply_goal(DEFAULT_GOAL, reason="session_ended", must_be_legal=True)
            self.strategy_index = 0
            self._reset_timers()
        elif target == Phase.ESCALATED:
            # Rule 5 escalating sets `wait_for_caregiver`, from whatever
            # goal was active -- see the module docstring for why this
            # goal's success condition ("caregiver present") is never
            # satisfied by code here.
            goal_change = self._apply_goal(
                WAIT_FOR_CAREGIVER_GOAL, reason=reason, must_be_legal=True
            )

        return Transition(
            phase=self.phase,
            session_id=self.session_id,
            goal=self.goal,
            strategy_index=self.strategy_index,
            reason=reason,
            notify=notify,
            goal_change=goal_change,
        )

    def _reset_timers(self) -> None:
        self._on_floor_since = None
        self._absent_since = None
        self._observing_since = None
        self._in_bed_since = None
        self._cooldown_since = None
        self._restroom_since = None
        self._pending_zone = None
        self._pending_zone_count = 0

    def _update_rule5_timers(self, state: str, now: datetime) -> None:
        if state == "on_floor":
            if self._on_floor_since is None:
                self._on_floor_since = now
            self._absent_since = None
        elif state == "absent":
            if self._absent_since is None:
                self._absent_since = now
            self._on_floor_since = None
        else:
            self._on_floor_since = None
            self._absent_since = None

    def _rule5_transition(self, now: datetime) -> Transition | None:
        """HANDOFF.md rule 5, evaluated before anything else on every
        `on_person_state`/`tick` call, from every phase: `on_floor` or
        `absent` held past its configured limit skips every strategy and
        escalates immediately, overriding anything else this update would
        otherwise decide. Rule 5 has no "only if a session is already
        live" qualifier in HANDOFF.md, and it outranks section 6's phase
        summary: a person on the floor with no prior `sitting_up`/
        `standing` reading (a fall straight out of `in_bed`), during
        `COOLDOWN`, or outside the night window still needs the caregiver
        called, because `perceive` is watching regardless of what `agent`
        is doing. Firing from `IDLE` goes straight to `ESCALATED` with a
        fresh session id (`_apply` mints one whenever `session_id is
        None`, which it is in `IDLE`). Firing from `COOLDOWN` also goes
        straight to `ESCALATED`, but deliberately keeps the id of the
        session that was resolving in that `COOLDOWN` rather than minting
        a new one (`_apply` only mints when `session_id is None`, and
        `COOLDOWN` never clears it -- only `IDLE` does): a fall moments
        after the person settled back into bed is the same night's
        episode, not a new one, and one continuous id keeps that readable
        on the timeline instead of splitting it in two. See
        `agent.rules.ALLOWED_TRANSITIONS` for the `IDLE -> ESCALATED`/
        `COOLDOWN -> ESCALATED` edges this relies on.

        Explicitly a no-op once already `ESCALATED`: there is nowhere
        further to escalate to, not an oversight.

        Deliberately not gated by `AgentConfig.in_night_window`: the night
        window controls whether a *nudging* session may start (talking to
        the person through the embodiment page), not whether a safety
        escalation may fire. Someone on the floor at 3pm still needs help.
        """
        if self.phase == Phase.ESCALATED:
            return None

        if self._on_floor_since is not None:
            elapsed = (now - self._on_floor_since).total_seconds()
            if elapsed >= self.config.floor_limit_seconds:
                return self._apply(
                    Phase.ESCALATED,
                    reason="rule5_on_floor",
                    notify=NotifySpec(
                        level="critical",
                        title="Possible fall",
                        body="Person detected on the floor.",
                    ),
                )

        if self._absent_since is not None:
            elapsed = (now - self._absent_since).total_seconds()
            if elapsed >= self.config.absent_limit_seconds:
                return self._apply(
                    Phase.ESCALATED,
                    reason="rule5_absent",
                    notify=NotifySpec(
                        level="critical",
                        title="Person missing",
                        body=(f"No detection for over {int(self.config.absent_limit_seconds)}s."),
                    ),
                )

        return None

    def on_person_state(self, state: str, zone: str, now: datetime) -> Transition | None:
        """Feed one `PersonState.state`/`.zone` classification through the
        machine.

        `zone` is issue #13's addition to this signature: perception
        seeing the person heading to the bathroom (`zone == "bathroom_path"`)
        is one of PLAN.md's three documented goal-change triggers, and it
        arrives on every `PersonState`, not as a separate event.

        Returns the `Transition` to publish, or `None` if this update did
        not change anything.
        """
        self._update_rule5_timers(state, now)

        rule5 = self._rule5_transition(now)
        if rule5 is not None:
            return rule5

        if self.phase == Phase.IDLE:
            if state in ("sitting_up", "standing") and self.config.in_night_window(now):
                self._observing_since = now
                return self._apply(Phase.OBSERVING, reason=f"person_{state}")
            return None

        if self.phase == Phase.OBSERVING:
            if state == "in_bed":
                return self._apply(Phase.IDLE, reason="returned_to_bed")
            if self._observing_since is not None:
                elapsed = (now - self._observing_since).total_seconds()
                if elapsed >= self.config.observe_seconds:
                    return self._apply(Phase.ENGAGED, reason="observe_timeout")
            return None

        if self.phase in (Phase.ENGAGED, Phase.ESCALATED):
            phase_transition = self._track_in_bed_stability(state, now)
            goal_change = self._update_goal_from_zone(zone, state, now)
            return self._combine(phase_transition, goal_change)

        if self.phase == Phase.COOLDOWN:
            # No new *nudging* session during cooldown (HANDOFF.md section
            # 6): a `PersonState` here can no longer start `OBSERVING`.
            # Rule 5 is the one exception -- already handled above, before
            # this dispatch -- so what's left is only `tick`'s
            # cooldown-elapsed check.
            return self.tick(now)

        return None

    def _update_goal_from_zone(
        self, zone: str, state: str, now: datetime
    ) -> GoalChangeResult | None:
        """Issue #13's two deterministic, zone-driven goal edges, both
        self-guarding on `self.goal` so calling this unconditionally from
        `ENGAGED`/`ESCALATED` is harmless when neither applies (e.g. while
        `wait_for_caregiver` is active).

        `zone` is run through `_confirmed_zone` first: a single reading of
        `bathroom_path` or `bed` must not flip the goal, since the two
        zones sit right next to each other and a person can easily flap
        between them for a few frames right at the moment a bathroom trip
        actually starts. `state == "in_bed"`, by contrast, is not run
        through any extra debounce here -- `perceive` already applies its
        own hysteresis (`StateTracker.confirm_frames`) before a `state`
        value ever reaches `agent` at all, unlike `zone`, which is a bare,
        unconfirmed per-frame lookup.

        - `bathroom_path`, confirmed, while the root goal is active:
          perception has seen the person heading to the bathroom,
          PLAN.md's second documented trigger. Switch to `restroom` and
          start its timeout clock.
        - Back at the bed (`zone == "bed"`, confirmed, or `state ==
          "in_bed"`) while `restroom` is active: the approximation of
          "went to the bathroom and came back" this issue can actually
          observe (see the module docstring). Return to the parent goal
          and clear the clock.
        """
        confirmed_zone = self._confirmed_zone(zone)

        if confirmed_zone == "bathroom_path" and self.goal == DEFAULT_GOAL:
            change = self._apply_goal(RESTROOM_GOAL, reason="perceived_heading_to_bathroom")
            if change is not None:
                self._restroom_since = now
            return change

        if self.goal == RESTROOM_GOAL and (confirmed_zone == "bed" or state == "in_bed"):
            change = self._apply_goal(DEFAULT_GOAL, reason="returned_from_bathroom")
            if change is not None:
                self._restroom_since = None
            return change

        return None

    def _combine(
        self, phase_transition: Transition | None, goal_change: GoalChangeResult | None
    ) -> Transition | None:
        """Merge a possible phase-change `Transition` with a possible
        goal-only `GoalChangeResult` from the same update into the single
        `Transition` this file's public methods return, per the `Transition`
        docstring: a goal change is always publishable, with or without an
        accompanying phase change."""
        if phase_transition is not None:
            if goal_change is not None and phase_transition.goal_change is None:
                return replace(phase_transition, goal_change=goal_change)
            return phase_transition
        if goal_change is not None:
            return Transition(
                phase=self.phase,
                session_id=self.session_id,
                goal=self.goal,
                strategy_index=self.strategy_index,
                reason=goal_change.reason,
                notify=None,
                goal_change=goal_change,
            )
        return None

    def propose_goal(self, goal: str, reason: str, now: datetime) -> Transition | None:
        """The one public entry point for an LLM's `plan` goal-change
        proposal (issue #15) to take effect. Routed through
        `agent.rules.validate_goal` exactly like every deterministic switch
        in this file; an illegal proposal is rejected quietly (`None`), not
        raised -- an ordinary, expected outcome of this layer, matching
        `agent.rules.validate`'s own convention (HANDOFF.md rule 1: every
        LLM proposal passes through the rule layer before it has any
        effect).

        `now` timestamps a resulting `restroom` entry/exit's timeout clock
        the same way a zone-triggered one is timestamped, even though this
        method makes no other use of the clock itself.
        """
        change = self._apply_goal(goal, reason=reason)
        if change is None:
            return None
        if goal == RESTROOM_GOAL:
            self._restroom_since = now
        elif change.from_goal == RESTROOM_GOAL:
            self._restroom_since = None
        return Transition(
            phase=self.phase,
            session_id=self.session_id,
            goal=self.goal,
            strategy_index=self.strategy_index,
            reason=reason,
            notify=None,
            goal_change=change,
        )

    def on_utterance(self, now: datetime) -> Transition | None:
        """Any `Utterance` moves `OBSERVING` straight to `ENGAGED`
        (HANDOFF.md section 6: "Enter `ENGAGED` after 20 s up, or on any
        `Utterance`"), even before the 20 s timer would. No effect in any
        other phase: `IDLE` never starts a session on speech alone (only
        `PersonState` does), and `ENGAGED`/`ESCALATED`/`COOLDOWN` are not
        eligible for this particular entry."""
        if self.phase == Phase.OBSERVING:
            return self._apply(Phase.ENGAGED, reason="utterance")
        return None

    def _track_in_bed_stability(self, state: str, now: datetime) -> Transition | None:
        """From `ENGAGED` or `ESCALATED`, `in_bed` held unbroken for
        `config.in_bed_stable_seconds` moves to `COOLDOWN` -- the crisis
        resolved on its own, whether or not it ever escalated."""
        if state != "in_bed":
            self._in_bed_since = None
            return None
        if self._in_bed_since is None:
            self._in_bed_since = now
        elapsed = (now - self._in_bed_since).total_seconds()
        if elapsed >= self.config.in_bed_stable_seconds:
            self._cooldown_since = now
            return self._apply(Phase.COOLDOWN, reason="in_bed_stable")
        return None

    def tick(self, now: datetime) -> Transition | None:
        """Time-only check: fires whatever `on_person_state`/`on_utterance`
        would eventually fire anyway, but without waiting for a new event
        to arrive. `agent.main.run` calls this every loop iteration in
        addition to feeding in new messages, the same way `capture`/
        `perceive`'s `run()` loops call their `maybe_emit_*` helpers on
        every iteration regardless of whether a frame was read."""
        rule5 = self._rule5_transition(now)
        if rule5 is not None:
            return rule5

        if self.phase == Phase.OBSERVING and self._observing_since is not None:
            elapsed = (now - self._observing_since).total_seconds()
            if elapsed >= self.config.observe_seconds:
                return self._apply(Phase.ENGAGED, reason="observe_timeout")

        if self.phase == Phase.COOLDOWN and self._cooldown_since is not None:
            elapsed = (now - self._cooldown_since).total_seconds()
            if elapsed >= self.config.cooldown_seconds:
                return self._apply(Phase.IDLE, reason="cooldown_elapsed")

        if self.goal == RESTROOM_GOAL and self._restroom_since is not None:
            elapsed = (now - self._restroom_since).total_seconds()
            if elapsed >= self.config.restroom_timeout_seconds:
                # The safety valve documented in `agent.goals` and
                # `AgentConfig.restroom_timeout_seconds`: give up waiting
                # for a "back at the bed" reading and return to the parent
                # goal anyway, rather than leaving `restroom` active
                # forever.
                change = self._apply_goal(
                    DEFAULT_GOAL, reason="restroom_timeout", must_be_legal=True
                )
                self._restroom_since = None
                if change is not None:
                    return Transition(
                        phase=self.phase,
                        session_id=self.session_id,
                        goal=self.goal,
                        strategy_index=self.strategy_index,
                        reason="restroom_timeout",
                        notify=None,
                        goal_change=change,
                    )

        return None
