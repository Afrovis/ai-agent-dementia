"""The session phase state machine for `agent` (issue #12).

Pure and time-injectable: every method takes `now` -- a local, naive
`datetime`, since `AgentConfig`'s night window is expressed in local
`HH:MM` -- as an explicit parameter instead of reading the wall clock
itself, matching `capture.gate.MotionGate`/`perceive.classify.StateTracker`'s
injected-clock convention. Tests build `datetime`s directly, so they are
deterministic and never sleep for real.

`Session` consumes plain `PersonState.state` strings and bare `Utterance`
arrivals (call sites just need to know "an utterance happened", not its
text or confidence -- interpreting *what* was said is `interpret`, issue
#15) and produces `Transition`s: phase changes for `agent.main` to publish
as `SessionState`, plus an optional `Notify` alongside a move into
`ESCALATED`.

Every phase change this class computes is run through `agent.rules.validate`
before being applied, including transitions this file itself decides on --
see the module docstring in `agent.rules` for why that is not redundant.

Two fixed seams for later issues, both intentionally inert for the
lifetime of issue #12:

- `goal` never changes from `DEFAULT_GOAL` ("return_to_bed"), and
  `GoalChanged` is never emitted. The goal tree and goal switching
  (`restroom`, `drink_water`, `comfort`, `wait_for_caregiver`) are issue
  #13. `Session.goal` exists as a field now specifically so a future
  `Session.set_goal(...)` (or similar) does not require restructuring the
  rest of this class.
- `strategy_index` never leaves `0`. The strategy engine -- ordering,
  cooldowns, dwell times, and "strategies exhausted" as an escalation
  trigger distinct from rule 5 -- is issue #14. "Strategies exhausted" is
  therefore not implemented anywhere in this module; the only escalation
  trigger issue #12 implements is rule 5.

Also out of scope, and not faked: distress detection ("distress detected
twice" in HANDOFF.md section 6) does not exist here at all, because it
depends on `interpret` (issue #15), which makes an LLM call this module
must never depend on.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from agent.config import AgentConfig
from agent.rules import Phase, validate

DEFAULT_GOAL = "return_to_bed"


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
class Transition:
    """One phase change `Session` wants published as `SessionState`, plus the
    `Notify` to publish alongside it, if any, and a short `reason` for
    logging (never published; not part of any event schema)."""

    phase: Phase
    session_id: str | None
    goal: str
    strategy_index: int
    reason: str
    notify: NotifySpec | None = None


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
        if target == Phase.IDLE:
            self.goal = DEFAULT_GOAL
            self.strategy_index = 0
            self._reset_timers()

        return Transition(
            phase=self.phase,
            session_id=self.session_id,
            goal=self.goal,
            strategy_index=self.strategy_index,
            reason=reason,
            notify=notify,
        )

    def _reset_timers(self) -> None:
        self._on_floor_since = None
        self._absent_since = None
        self._observing_since = None
        self._in_bed_since = None
        self._cooldown_since = None

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
        is doing. Firing from `IDLE` or `COOLDOWN` starts a session (a
        fresh id, same as any other entry) and goes straight to
        `ESCALATED`; see `agent.rules.ALLOWED_TRANSITIONS` for the
        `IDLE -> ESCALATED`/`COOLDOWN -> ESCALATED` edges this relies on.

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

    def on_person_state(self, state: str, now: datetime) -> Transition | None:
        """Feed one `PersonState.state` classification through the machine.

        Returns the `Transition` to publish, or `None` if this update did
        not change the phase.
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
            return self._track_in_bed_stability(state, now)

        if self.phase == Phase.COOLDOWN:
            # No new *nudging* session during cooldown (HANDOFF.md section
            # 6): a `PersonState` here can no longer start `OBSERVING`.
            # Rule 5 is the one exception -- already handled above, before
            # this dispatch -- so what's left is only `tick`'s
            # cooldown-elapsed check.
            return self.tick(now)

        return None

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

        return None
