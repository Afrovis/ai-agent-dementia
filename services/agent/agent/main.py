"""Entry point for the `agent` service (issue #12): the real session core.

Reads `PersonState` from the `person` stream and complete `Utterance` events
from `speech_in` (ignoring its early barge-in signal), and drives an
`agent.session.Session` -- the deterministic phase state machine, see that
module and `agent.rules` for what it does and does not decide.

Split the same way `perceive.main`/`capture.main` are: `run_once` reads
whatever is waiting, feeds it through `session`, publishes the resulting
`SessionState`/`Notify`/`Show` events, and returns -- side-effect-light and
directly unit-testable with a `FakeBus`, no Redis, camera, mic, or Ollama.
`run()` is the infinite, real-time loop Docker actually runs, and has
nothing left to unit test directly.

Publishes, per HANDOFF.md section 5 and rule 4 ("fail loud to the
caregiver, fail quiet to the person"):

- `SessionState` on every phase change, plus a heartbeat at
  `HEARTBEAT_INTERVAL_S` otherwise, so a silent `agent` cannot be mistaken
  for a calm night -- the same reasoning as `perceive`'s `PersonState`
  heartbeat and `capture`'s idle fps.
- `Show`: a dim clock at `IDLE`/`COOLDOWN` (`_show_for_phase`, unchanged
  from issue #12), or, whenever `agent.session.Session` selected a
  strategy (`Transition.strategy` is set -- every `ENGAGED`/`ESCALATED`
  update), that strategy's `Show` rendered through `agent.strategies.
  render_template` (issue #14).
- `Say`: only when the selected strategy has a `say_template`, rendered
  the same way, and only after it passes `agent.rules.validate_say`
  (HANDOFF.md rule 3: one sentence, a minimum silence gap, no forbidden
  phrasing, no memory-testing question by form). A `Say` that fails
  validation is logged loudly. A sentence blocked only by the silence gap
  waits briefly for its turn; other rejections remain silent. Issue #15
  composes strategy 4 from the latest utterance
  with the local LLM; a failed call falls back to the caregiver's fixed
  template, and both paths pass through the same deterministic validator.
- `Notify(critical, repeat_until_ack=True, source="agent")` on entering
  `ESCALATED`, built from the `NotifySpec` `agent.session.Session` attaches
  to that `Transition`.
- `Health`, matching `perceive`/`capture`.

The veto checks proposed actions against session and profile facts before
publication; a denied strategy suppresses its `Show` and `Say` together.

Issue #25 wraps the local client with an opt-in Claude fallback. Only a
second consecutive local `unclear` interpretation or a local planner result
below 0.4 confidence may leave the device; composition is always local. Every
outbound structured-text payload is published as `CloudCall` before the SDK
request so `store` retains it and the caregiver History page shows it.

`AGENT_FAKE=true` dispatches to `agent.fake.run` instead -- the M0 demo
fixture (HANDOFF.md section 8, CLAUDE.md) that cycles all `Show` states
with no perception, LLM, or session machine in the loop. Default is the
real agent above.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import redis
from nc_shared.bus import Bus
from nc_shared.events import (
    Activity,
    CloudCall,
    DebugControl,
    GoalChanged,
    Health,
    LightCommand,
    Notify,
    PersonState,
    ResetSession,
    Say,
    SessionState,
    Show,
    Utterance,
)

from agent import veto
from agent.config import AgentConfig
from agent.goals import GOALS
from agent.llm import ClaudeLLM, FallbackLLM, LLMClient
from agent.llm import local_llm as build_local_llm
from agent.profile import DEFAULT_PROFILE, PersonProfile, load_profile
from agent.questions import is_direct_question
from agent.rules import Phase, RuleResult, validate_composition, validate_say
from agent.session import PendingSay, Session, Transition
from agent.strategies import (
    ACKNOWLEDGE_PAIN_ID,
    ACKNOWLEDGE_PROGRESS_ID,
    COMFORT_PAIN_ID,
    ESCALATE_PHONE_ID,
    PATH_LIGHT_ID,
    REASSURE_WAITING_ID,
    StrategyDef,
    load_strategies,
    render_template,
    time_as_words,
)
from agent.veto import Proposal, VetoContext

SERVICE_NAME = "agent"
HEALTH_INTERVAL_S = 30.0
HEARTBEAT_INTERVAL_S = 60.0
PENDING_SAY_MAX_AGE_S = 30.0
MAX_REASSURANCES = 2
REASSURANCE_FALLBACKS = (
    "Help is on the way{name_vocative}; you can rest where you are.",
    "I've let someone know{name_vocative}, and they're coming to you.",
    "You're not alone{name_vocative}; help is coming.",
)

PERSON_STREAM = "person"
PERSON_GROUP = "agent"
UTTERANCE_STREAM = "speech_in"
UTTERANCE_GROUP = "agent"
DEBUG_STREAM = "debug"
DEBUG_GROUP = "agent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _activity(
    bus, kind: str, phase: str, *, session_id=None, ok=True, duration_ms=None, detail=None
):
    event = Activity(
        source=SERVICE_NAME,
        session_id=session_id,
        service=SERVICE_NAME,
        kind=kind,
        phase=phase,
        ok=ok,
        duration_ms=duration_ms,
        detail=detail,
    )
    bus.publish(event, maxlen=200)
    _log("published Activity", event_type="Activity", kind=kind, phase=phase, ok=ok, detail=detail)


def _llm_model_name(llm) -> str:
    local = getattr(llm, "_local", llm)
    return str(getattr(local, "model", type(local).__name__))


def _log(message: str, level: int = logging.INFO, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4)."""
    logger.log(level, json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


def veto_context(session: Session, profile: PersonProfile, now: datetime) -> VetoContext:
    """Snapshot the current person, session, and caregiver facts for one veto check."""
    return VetoContext(
        phase=session.phase.value,
        goal=session.goal,
        person_state=session.last_person_state,
        recent_utterances=session.recent_utterances,
        settled=session.settled,
        in_night_window=session.config.in_night_window(session.wall_clock(now)),
        things_to_avoid=profile.things_to_avoid,
        restroom_need_resolved=session._restroom_need_resolved,
    )


def _decision_activity(bus, session_id: str | None, detail: dict[str, object]) -> None:
    """Best-effort trace record; telemetry must not interrupt agent decisions."""
    try:
        event = Activity(
            source=SERVICE_NAME,
            session_id=session_id,
            service=SERVICE_NAME,
            kind="decision",
            phase="end",
            ok=False,
            duration_ms=None,
            detail=json.dumps(detail, sort_keys=True, separators=(",", ":")),
        )
        bus.publish(event, maxlen=200)
        _log("published Activity", event_type="Activity", kind="decision", phase="end", ok=False)
    except Exception as exc:
        _log("failed to publish decision Activity", level=logging.WARNING, error=type(exc).__name__)


def _reassure_or_stay_silent(
    bus,
    session: Session,
    text: str,
    now: datetime,
    profile: PersonProfile,
    llm,
    *,
    distress: int | None = None,
) -> None:
    """Reserve silence after two replies while still answering questions or maximum distress."""
    if session.reassurance_count < MAX_REASSURANCES or is_direct_question(text) or distress == 3:
        _reply_to_utterance(bus, session, REASSURE_WAITING_ID, now, profile, llm)
        return
    detail = {
        "decision": "no_reply",
        "reason": "reassured_enough",
        "text": text,
        "phase": session.phase.value,
        "goal": session.goal,
        "reassurances": session.reassurance_count,
    }
    _decision_activity(bus, session.session_id, detail)
    _log("no reply after reassurance cap", **detail)


def _intentional_silence(bus, session: Session, text: str, reason: str) -> None:
    _decision_activity(
        bus,
        session.session_id,
        {
            "decision": "no_reply",
            "reason": reason,
            "text": text,
            "phase": session.phase.value,
            "goal": session.goal,
        },
    )


def _vetoed(
    bus, proposal: Proposal, context: VetoContext, *, session_id: str | None = None
) -> bool:
    """Log a denial once without exposing candidate speech in operational logs."""
    verdict = veto.check(proposal, context)
    if verdict.allowed:
        return False
    event_type = {
        "strategy": "Show",
        "say": "Say",
        "notify": "Notify",
        "light": "LightCommand",
    }[proposal.kind]
    _log(
        f"vetoed {proposal.kind}",
        level=logging.WARNING,
        event_type=event_type,
        rule=verdict.rule,
        clause=verdict.clause,
        action=f"{proposal.kind}:{proposal.value}",
        reason=verdict.reason,
    )
    _decision_activity(
        bus,
        session_id,
        {
            "decision": "vetoed",
            "rule": verdict.rule,
            "event_type": event_type,
            "strategy": proposal.value if proposal.kind in ("strategy", "say") else None,
            "text": proposal.text if proposal.kind in ("say", "notify") else None,
            "reason": verdict.reason,
            "phase": context.phase,
            "goal": context.goal,
        },
    )
    return True


def _show_for_phase(phase: Phase, session_id: str | None) -> Show:
    """The minimal, honest `Show` for `phase` when no strategy is selected
    (`IDLE`/`COOLDOWN`, and briefly `OBSERVING` before a strategy exists to
    show). See `_show_for_transition` for the strategy-driven case."""
    if phase == Phase.IDLE:
        return Show(
            source=SERVICE_NAME,
            session_id=session_id,
            face="asleep",
            headline="It is night",
            body="Resting.",
            brightness=0.1,
        )
    if phase == Phase.COOLDOWN:
        return Show(
            source=SERVICE_NAME,
            session_id=session_id,
            face="asleep",
            headline="All calm",
            body="Cooldown.",
            brightness=0.1,
        )
    if phase == Phase.ESCALATED:
        # Safety fallback only: `_apply` always forces `escalate_phone`
        # selected on entering `ESCALATED` (issue #14), so
        # `_show_for_transition` normally never reaches this branch.
        return Show(
            source=SERVICE_NAME,
            session_id=session_id,
            face="awake",
            headline="Someone is coming to help",
            body="",
            brightness=0.2,
        )
    # OBSERVING: awake, before a strategy is selected.
    return Show(
        source=SERVICE_NAME,
        session_id=session_id,
        face="awake",
        headline="",
        body="",
        brightness=0.4,
    )


def _show_for_strategy(
    strategy: StrategyDef, session_id: str | None, now: datetime, profile: PersonProfile
) -> Show:
    """Render one strategy's `Show` fields (issue #14): its fixed `face`/
    `brightness`/`photo_id`, and its headline/body templates interpolated
    through `agent.strategies.render_template` with `profile` and the
    current time-as-words."""
    extra = {"time_words": time_as_words(now)}
    return Show(
        source=SERVICE_NAME,
        session_id=session_id,
        face=strategy.face,
        headline=render_template(strategy.headline_template, profile, **extra),
        body=render_template(strategy.body_template, profile, **extra),
        photo_id=strategy.photo_id,
        brightness=strategy.brightness,
    )


def _show_for_transition(transition: Transition, now: datetime, profile: PersonProfile) -> Show:
    """The `Show` to publish for `transition`: strategy-driven whenever one
    was selected, `_show_for_phase`'s minimal fallback otherwise."""
    if transition.strategy is not None:
        return _show_for_strategy(transition.strategy, transition.session_id, now, profile)
    return _show_for_phase(transition.phase, transition.session_id)


def _profile_for_llm(profile: PersonProfile) -> dict[str, object]:
    """Return every PLAN.md profile field as structured, JSON-safe prompt data."""
    return profile.prompt_data()


def _session_state_for_llm(session: Session) -> dict[str, object]:
    """Expose only advisory planner context, never executable state changes."""
    ordered_strategies = sorted(session.strategies, key=lambda strategy: strategy.order)
    current_strategy = next(
        (
            strategy.id
            for index, strategy in enumerate(ordered_strategies)
            if index == session.strategy_index and session.phase == Phase.ENGAGED
        ),
        None,
    )
    return {
        "phase": session.phase.value,
        "session_id": session.session_id,
        "goal": session.goal,
        "strategy_index": session.strategy_index,
        "current_strategy": current_strategy,
        "strategy_order": [strategy.id for strategy in ordered_strategies],
        "allowed_goals": sorted(GOALS),
        "recent_utterances": list(session.recent_utterances),
        "scene_note": session.last_scene_note,
    }


def _publish_cloud_call(
    bus,
    session: Session,
    task: str,
    model: str,
    payload: dict[str, object],
) -> None:
    """Persist the exact text-only payload before an opted-in cloud request."""
    bus.publish(
        CloudCall(
            source=SERVICE_NAME,
            session_id=session.session_id,
            task=task,
            model=model,
            payload=payload,
        )
    )
    _log("published CloudCall", event_type="CloudCall", task=task, model=model)


def _maybe_publish_say(
    bus,
    transition: Transition,
    session: Session,
    now: datetime,
    profile: PersonProfile,
    llm: LLMClient | None = None,
    *,
    context: VetoContext | None = None,
    direct: bool = False,
) -> bool:
    """Publish a `Say` for `transition.strategy`, if it has a
    `say_template`, after passing it through `agent.rules.validate_say`
    (HANDOFF.md rule 3). A min-gap-only rejection is queued until the gap
    elapses; all other rejections remain silent. The minimum-gap clock moves
    only when speech is actually published.
    """
    strategy = transition.strategy
    if strategy is None or strategy.say_template is None:
        return False
    terminal_reply = strategy.terminal or transition.reason == "pain_reported"
    if session._pending_say is not None:
        _drop_pending_say(bus, session, "superseded", now)

    # Occupancy suppresses only ordinary bedside speech. The Show still
    # updates, the ladder and its dwell timers continue unchanged, and the
    # terminal escalation sentence remains audible in case a person fell
    # just outside the camera's view. The latest raw reading is used rather
    # than waiting for rule 5's absence limit: speaking to a room currently
    # believed empty has no benefit during that grace period.
    if not session.person_present(now) and not terminal_reply:
        _log(
            "suppressed ordinary Say while person is absent",
            event_type="Say",
            strategy=strategy.id,
        )
        return False

    time_words = time_as_words(session.wall_clock(now))
    text = render_template(strategy.say_template, profile, time_words=time_words)
    compose = llm is not None and (
        strategy.id == "validate_and_redirect"
        or (direct and strategy.id in (REASSURE_WAITING_ID, "orient_time_place"))
    )
    if compose:
        model = _llm_model_name(llm)
        _activity(bus, "compose", "start", session_id=session.session_id, detail=model)
        started = time.perf_counter()
        outcome = "unavailable"
        composition_result = None
        try:
            composition = llm.compose(
                strategy.id,
                # The rendered phrase, so the model never sees a raw placeholder.
                text,
                _profile_for_llm(profile),
                time_words,
                session.last_scene_note,
                session.recent_utterances[-1] if session.recent_utterances else None,
                session.goal,
            )
            if composition is not None:
                composition_result = validate_composition(composition.text, profile)
                if (
                    composition_result.accepted
                    and strategy.id == "orient_time_place"
                    and (
                        time_words.lower() not in composition.text.lower()
                        or not any(word in composition.text.lower() for word in ("home", "bedroom"))
                    )
                ):
                    composition_result = RuleResult(False, "orientation lacks local time or place")
                outcome = "ok" if composition_result.accepted else "rejected"
        except Exception:
            outcome = "error"
            raise
        finally:
            _activity(
                bus,
                "compose",
                "end",
                session_id=session.session_id,
                ok=outcome == "ok",
                duration_ms=(time.perf_counter() - started) * 1000,
                detail=f"{model} {outcome}",
            )
        # A model failure cannot replace the caregiver's known-safe phrase.
        # The rendered template still passes the same deterministic Say gate.
        if composition is None:
            _log("LLM composition unavailable; using caregiver fallback", level=logging.WARNING)
        else:
            if not composition_result.accepted:
                # Log the deterministic reason only. The candidate may
                # paraphrase a private utterance and must never be copied
                # into operational logs merely because it was unsafe.
                _log(
                    "rejected LLM composition; using caregiver fallback",
                    level=logging.WARNING,
                    reason=composition_result.reason,
                )
            else:
                text = composition.text
    if strategy.id == REASSURE_WAITING_ID and session.phase == Phase.ESCALATED:
        # Prefer the caregiver's template, then approved alternatives when
        # composition or a fixed fallback repeats something already spoken.
        used = {" ".join(s.lower().split()) for s in session.reassurance_texts}
        if " ".join(text.lower().split()) in used:
            for fallback in (strategy.say_template, *REASSURANCE_FALLBACKS):
                candidate = render_template(fallback, profile, time_words=time_words)
                if " ".join(candidate.lower().split()) not in used:
                    text = candidate
                    break
    # The minimum-silence gap paces ordinary strategy speech, one sentence
    # then quiet. A terminal strategy (`escalate_phone`) is not ordinary
    # speech: it is the single sentence telling a person who may be on the
    # floor that help is coming, published in the same breath as the
    # critical `Notify` that summons it. If an ordinary strategy happened
    # to speak in the seconds before the escalation, the gap check would
    # drop that sentence and leave them with a silent screen, so the
    # terminal strategy is exempt from this one check. Every other rule 3
    # check -- one sentence, no forbidden phrasing, no question -- still
    # applies to it exactly as before.
    seconds_since_last_say = None if terminal_reply else session.seconds_since_last_say(now)
    result = validate_say(
        text,
        seconds_since_last_say=seconds_since_last_say,
        min_gap_seconds=session.config.say_min_gap_seconds,
    )
    gap_only = (
        not result.accepted
        and validate_say(
            text, seconds_since_last_say=None, min_gap_seconds=session.config.say_min_gap_seconds
        ).accepted
    )
    if not result.accepted and not gap_only:
        _log(
            "rejected Say, falling back to silence",
            level=logging.WARNING,
            event_type="Say",
            strategy=strategy.id,
            reason=result.reason,
        )
        return False

    if context is None:
        context = veto_context(session, profile, now)
    if _vetoed(
        bus,
        Proposal("say", strategy.id, text=text, terminal=terminal_reply),
        context,
        session_id=session.session_id,
    ):
        return False

    say_event = Say(
        source=SERVICE_NAME,
        session_id=transition.session_id,
        text=text,
        strategy=strategy.id,
        # `escalate_phone` must not be interrupted by barge-in the way an
        # ordinary strategy's speech can be (HANDOFF.md section 7:
        # `listen`'s barge-in) -- there is nothing left to redirect to.
        interruptible=not terminal_reply,
        clip_id=strategy.clip_id,
    )
    if gap_only:
        session._pending_say = PendingSay(
            say_event,
            transition.goal,
            transition.strategy_index,
            now,
            direct=direct,
            trigger=transition.reason,
        )
        _log(
            "deferred Say",
            event_type="Say",
            strategy=strategy.id,
            session_id=transition.session_id,
            goal=transition.goal,
            reason=result.reason,
        )
        return False

    bus.publish(say_event)
    session.record_say(now, strategy.id, say_event.text)
    _log("published Say", event_type="Say", strategy=strategy.id)
    _said_decision(bus, session, say_event, trigger=transition.reason, direct=direct)
    return True


# Transition reasons produced while handling the person's speech; any `interpreted_*`
# reason is one too. A Say from any other transition (a dwell timer, a zone change) is a
# scheduled step, even when it happens to follow an utterance.
_SPEECH_TRIGGERS = frozenset(
    {
        "utterance",
        "utterance_reply",
        "llm_plan_strategy",
        "distress_detected_twice",
        "pain_reported",
    }
)


def _said_decision(
    bus, session: Session, say: Say, *, trigger: str | None, direct: bool, age_s: float = 0.0
) -> None:
    """Why a Say went out: the transition that caused it (a dwell timer or the person's
    speech), so a trace can tell a reply from a ladder step that happened to follow."""
    reply = direct or (
        trigger is not None and (trigger in _SPEECH_TRIGGERS or trigger.startswith("interpreted_"))
    )
    _decision_activity(
        bus,
        say.session_id,
        {
            "decision": "said",
            "strategy": say.strategy,
            "trigger": trigger,
            "direct": direct,
            "reply": reply,
            "deferred_s": round(age_s, 3),
            "phase": session.phase.value,
            "goal": session.goal,
        },
    )


def _drop_pending_say(bus, session: Session, reason: str, now: datetime) -> None:
    pending = session._pending_say
    if pending is None:
        return
    session._pending_say = None
    _log(
        "dropped deferred Say",
        event_type="Say",
        strategy=pending.event.strategy,
        session_id=pending.event.session_id,
        reason=reason,
    )
    _decision_activity(
        bus,
        pending.event.session_id,
        {
            "decision": "pending_say_dropped",
            "reason": reason,
            "text": pending.event.text,
            "strategy": pending.event.strategy,
            "direct": pending.direct,
            "age_s": (now - pending.queued_at).total_seconds(),
            "phase": session.phase.value,
            "goal": session.goal,
        },
    )


def _flush_pending_say(bus, session: Session, now: datetime, profile: PersonProfile) -> None:
    pending = session._pending_say
    if pending is None:
        return
    age = (now - pending.queued_at).total_seconds()
    current = session._engine.current()
    seconds_since_last_say = session.seconds_since_last_say(now)
    if age > PENDING_SAY_MAX_AGE_S:
        _drop_pending_say(bus, session, "max_age", now)
    elif session.session_id != pending.event.session_id:
        _drop_pending_say(bus, session, "session_changed", now)
    elif session.phase not in (Phase.OBSERVING, Phase.ENGAGED, Phase.ESCALATED):
        _drop_pending_say(bus, session, "session_inactive", now)
    elif session.goal != pending.goal:
        _drop_pending_say(bus, session, "goal_changed", now)
    elif not pending.direct and (
        current is None
        or current.id != pending.event.strategy
        or session.strategy_index != pending.strategy_index
    ):
        _drop_pending_say(bus, session, "strategy_changed", now)
    elif not session.person_present(now) and pending.event.strategy != ESCALATE_PHONE_ID:
        _drop_pending_say(bus, session, "person_absent", now)
    elif (
        seconds_since_last_say is not None
        and seconds_since_last_say < session.config.say_min_gap_seconds
    ):
        return
    else:
        context = veto_context(session, profile, now)
        strategy_id = pending.event.strategy
        terminal = strategy_id == ESCALATE_PHONE_ID
        if _vetoed(
            bus,
            Proposal("strategy", strategy_id, terminal=terminal),
            context,
            session_id=session.session_id,
        ) or _vetoed(
            bus,
            Proposal("say", strategy_id, text=pending.event.text, terminal=terminal),
            context,
            session_id=session.session_id,
        ):
            _drop_pending_say(bus, session, "vetoed", now)
            return
        if pending.show is not None:
            bus.publish(pending.show.model_copy(update={"ts": datetime.now(UTC)}))
        bus.publish(pending.event.model_copy(update={"ts": datetime.now(UTC)}))
        session.record_say(now, pending.event.strategy, pending.event.text)
        session._pending_say = None
        _said_decision(
            bus,
            session,
            pending.event,
            trigger=pending.trigger,
            direct=pending.direct,
            age_s=(now - pending.queued_at).total_seconds(),
        )
        _log(
            "published deferred Say",
            event_type="Say",
            strategy=pending.event.strategy,
            session_id=pending.event.session_id,
        )


def _publish_transition(
    bus,
    transition: Transition,
    session: Session,
    now: datetime,
    *,
    profile: PersonProfile = DEFAULT_PROFILE,
    llm: LLMClient | None = None,
) -> SessionState:
    """Publish everything one `Transition` implies: `SessionState`, an
    optional `GoalChanged` (issue #13), an optional `Notify`, the `Show`
    (phase-driven or strategy-driven, issue #14), and, if the selected
    strategy has one, a `Say` that has passed `agent.rules.validate_say`.
    Returns the `SessionState` published, for callers that just want to
    know what happened.

    `SessionState` is always republished here, even for a goal-only or
    strategy-only update with no phase change, so `SessionState.goal`/
    `.strategy_index` -- shown on the dashboard timeline -- never lag
    behind. `profile` is the caregiver-authored profile loaded by `run`;
    direct callers default to a safe generic profile."""
    if transition.reason == "pain_reported":
        pain_strategy = next(s for s in session.strategies if s.id == ACKNOWLEDGE_PAIN_ID)
        transition = replace(transition, strategy=pain_strategy)
    context = veto_context(session, profile, now)
    event = SessionState(
        source=SERVICE_NAME,
        session_id=transition.session_id,
        phase=transition.phase.value,
        goal=transition.goal,
        strategy_index=transition.strategy_index,
    )
    bus.publish(event)
    _log(
        "published SessionState",
        event_type="SessionState",
        phase=event.phase,
        reason=transition.reason,
    )

    if transition.goal_change is not None:
        goal_change = transition.goal_change
        goal_changed_event = GoalChanged(
            source=SERVICE_NAME,
            session_id=goal_change.session_id,
            from_goal=goal_change.from_goal,
            to_goal=goal_change.to_goal,
            reason=goal_change.reason,
        )
        bus.publish(goal_changed_event)
        _log(
            "published GoalChanged",
            event_type="GoalChanged",
            from_goal=goal_changed_event.from_goal,
            to_goal=goal_changed_event.to_goal,
        )

        if (
            goal_changed_event.to_goal == "restroom"
            and transition.strategy is not None
            and transition.strategy.id == PATH_LIGHT_ID
        ):
            light_event = LightCommand(
                source=SERVICE_NAME,
                session_id=transition.session_id,
                light="hallway",
                state="on",
                reason="restroom_goal_started",
            )
            if not _vetoed(
                bus, Proposal("light", light_event.state), context, session_id=session.session_id
            ):
                bus.publish(light_event)
                _log("published LightCommand", event_type="LightCommand", state="on")
        elif (
            goal_changed_event.from_goal == "restroom"
            and goal_changed_event.to_goal == "return_to_bed"
        ):
            light_event = LightCommand(
                source=SERVICE_NAME,
                session_id=transition.session_id,
                light="hallway",
                state="off",
                reason="restroom_goal_ended",
            )
            if not _vetoed(
                bus, Proposal("light", light_event.state), context, session_id=session.session_id
            ):
                bus.publish(light_event)
                _log("published LightCommand", event_type="LightCommand", state="off")

    # A phase resolution is a final idempotent backstop for a light that
    # remained on through an escalation or an unusual goal transition.
    if transition.phase in (Phase.COOLDOWN, Phase.IDLE) and transition.goal_change is None:
        light_event = LightCommand(
            source=SERVICE_NAME,
            session_id=transition.session_id,
            light="hallway",
            state="off",
            reason=f"session_{transition.phase.value.lower()}",
        )
        if not _vetoed(
            bus, Proposal("light", light_event.state), context, session_id=session.session_id
        ):
            bus.publish(light_event)
            _log("published LightCommand", event_type="LightCommand", state="off")

    if transition.notify is not None:
        notify_event = Notify(
            source=SERVICE_NAME,
            session_id=transition.session_id,
            level=transition.notify.level,
            title=transition.notify.title,
            body=transition.notify.body,
            repeat_until_ack=True,
        )
        if not _vetoed(
            bus,
            Proposal("notify", notify_event.level, text=notify_event.title),
            context,
            session_id=session.session_id,
        ):
            bus.publish(notify_event)
            _log("published Notify", event_type="Notify", notify_level=notify_event.level)

    strategy = transition.strategy
    if strategy is None or not _vetoed(
        bus,
        Proposal(
            "strategy",
            strategy.id,
            terminal=strategy.terminal or transition.reason == "pain_reported",
        ),
        context,
        session_id=session.session_id,
    ):
        show_event = _show_for_transition(transition, session.wall_clock(now), profile)
        spoke = _maybe_publish_say(bus, transition, session, now, profile, llm, context=context)
        if show_event.face == "speaking" and not spoke:
            if (
                session._pending_say is not None
                and session._pending_say.event.strategy == strategy.id
            ):
                session._pending_say = replace(session._pending_say, show=show_event)
            show_event = show_event.model_copy(update={"face": "awake"})
        bus.publish(show_event)
        _log("published Show", event_type="Show", face=show_event.face)

    return event


def _reply_to_utterance(
    bus, session: Session, strategy_id: str, now: datetime, profile: PersonProfile, llm
) -> None:
    """Answer once without selecting a ladder rung or changing the session goal."""
    strategy = next((item for item in session.strategies if item.id == strategy_id), None)
    if strategy is None:
        return
    context = veto_context(session, profile, now)
    if _vetoed(bus, Proposal("strategy", strategy.id), context, session_id=session.session_id):
        return
    if strategy_id == PATH_LIGHT_ID and session.goal != "restroom":
        # Direct guidance does not change the goal. Keep any caregiver alert
        # active while still meeting the stated toilet need.
        light = LightCommand(
            source=SERVICE_NAME,
            session_id=session.session_id,
            light="hallway",
            state="on",
            reason="escalated_restroom_need",
        )
        if not _vetoed(bus, Proposal("light", "on"), context, session_id=session.session_id):
            bus.publish(light)
    transition = Transition(
        phase=session.phase,
        session_id=session.session_id,
        goal=session.goal,
        strategy_index=session.strategy_index,
        reason="utterance_reply",
        strategy=strategy,
    )
    show = _show_for_strategy(strategy, session.session_id, session.wall_clock(now), profile)
    spoke = _maybe_publish_say(
        bus, transition, session, now, profile, llm, context=context, direct=True
    )
    if show.face == "speaking" and not spoke:
        if session._pending_say is not None and session._pending_say.event.strategy == strategy.id:
            session._pending_say = replace(session._pending_say, show=show)
        show = show.model_copy(update={"face": "awake"})
    bus.publish(show)


def _publish_debug_state(bus, session: Session) -> None:
    overrides = session.debug_overrides
    bus.publish(
        DebugControl(
            source=SERVICE_NAME,
            time_offset_hours=overrides.time_offset_hours,
            force_in_bed=overrides.force_in_bed,
        ),
        maxlen=100,
    )
    _log(
        "published DebugControl",
        event_type="DebugControl",
        time_offset_hours=overrides.time_offset_hours,
        force_in_bed=overrides.force_in_bed,
    )


def _handle_person_state(
    bus,
    session: Session,
    state: str,
    zone: str,
    now: datetime,
    published: list[SessionState],
    profile: PersonProfile,
    llm: LLMClient | None,
) -> None:
    transition = session.on_person_state(state, zone, now)
    if transition is not None:
        published.append(
            _publish_transition(bus, transition, session, now, profile=profile, llm=llm)
        )


def run_once(
    bus,
    session: Session,
    *,
    consumer: str = "agent-1",
    count: int = 10,
    block_ms: int = 200,
    now_fn: Callable[[], datetime] = datetime.now,
    profile: PersonProfile = DEFAULT_PROFILE,
    llm: LLMClient | None = None,
) -> list[SessionState]:
    """Read whatever `PersonState`/`Utterance` messages are waiting, feed
    them through `session` in arrival order, and publish one `SessionState`
    (plus any `Notify`/`Show`/`Say`) per resulting update. Also calls
    `session.tick` once, so timers -- including a strategy's dwell -- advance
    even when nothing new arrived.

    Returns the `SessionState`s published, most recent last. Side-effect-
    free beyond bus reads/acks/publishes, so tests can call it directly and
    in a loop with a `FakeBus`, instead of going through the infinite,
    real-time `run()` loop. `profile` defaults to `DEFAULT_PROFILE` for
    tests and other direct callers.
    """
    published: list[SessionState] = []

    for msg_id, event in bus.read(DEBUG_STREAM, DEBUG_GROUP, consumer, count=count, block_ms=None):
        bus.ack(DEBUG_STREAM, DEBUG_GROUP, msg_id)
        if not isinstance(event, (DebugControl, ResetSession)) or event.source == SERVICE_NAME:
            continue
        now = now_fn()
        utc_now = now.astimezone(UTC)
        if (utc_now - event.ts).total_seconds() > 60:
            continue
        if isinstance(event, ResetSession):
            transition = session.reset()
            published.append(
                _publish_transition(bus, transition, session, now, profile=profile, llm=llm)
            )
            _log("session reset by operator", level=logging.WARNING, event_type="ResetSession")
            effective = (
                ("in_bed", "bed")
                if session.debug_overrides.force_in_bed
                else session._last_real_person
            )
            if effective is not None:
                _handle_person_state(bus, session, *effective, now, published, profile, llm)
            continue
        overrides = session.debug_overrides
        was_forced = overrides.force_in_bed
        overrides.time_offset_hours = event.time_offset_hours
        overrides.force_in_bed = event.force_in_bed
        if was_forced != overrides.force_in_bed:
            effective = ("in_bed", "bed") if overrides.force_in_bed else session._last_real_person
            if effective is not None:
                _handle_person_state(bus, session, *effective, now, published, profile, llm)
        _publish_debug_state(bus, session)
        _log(
            "debug override applied",
            level=logging.WARNING,
            event_type="DebugControl",
            time_offset_hours=overrides.time_offset_hours,
            force_in_bed=overrides.force_in_bed,
        )

    person_messages = bus.read(
        PERSON_STREAM, PERSON_GROUP, consumer, count=count, block_ms=block_ms
    )
    for msg_id, event in person_messages:
        bus.ack(PERSON_STREAM, PERSON_GROUP, msg_id)
        assert isinstance(event, PersonState)
        now = now_fn()
        session.record_scene_note(event.scene_note)
        session._last_real_person = (event.state, event.zone)
        state, zone = (
            ("in_bed", "bed") if session.debug_overrides.force_in_bed else session._last_real_person
        )
        _handle_person_state(bus, session, state, zone, now, published, profile, llm)

    utterance_messages = bus.read(
        UTTERANCE_STREAM, UTTERANCE_GROUP, consumer, count=count, block_ms=None
    )
    for msg_id, event in utterance_messages:
        bus.ack(UTTERANCE_STREAM, UTTERANCE_GROUP, msg_id)
        # `listen` also puts the early, transcript-free `SpeechStarted`
        # barge-in signal on this stream.  Embodiment consumes that signal;
        # the session core acts only on complete transcribed utterances.
        if not isinstance(event, Utterance):
            continue
        now = now_fn()
        if session.is_self_echo(event.text, now):
            _log(
                "ignored self-echo Utterance", event_type="Utterance", session_id=session.session_id
            )
            continue
        prior_turns = session.recent_utterances
        transition = session.on_utterance(now)
        cooldown_reply = (
            session.phase == Phase.COOLDOWN
            and session.last_person_state is not None
            and session.last_person_state != "in_bed"
            and not session.debug_overrides.force_in_bed
        )
        llm_active = (
            session.phase in (Phase.OBSERVING, Phase.ENGAGED, Phase.ESCALATED) or cooldown_reply
        )
        if llm_active:
            # Record only after snapshotting the prior turns used by
            # ``interpret``. Composition triggered by this same update can
            # still see the newest utterance.
            session.record_utterance(event.text)
        if transition is not None:
            published.append(
                _publish_transition(bus, transition, session, now, profile=profile, llm=llm)
            )

        if llm is not None and llm_active:
            model = _llm_model_name(llm)
            _activity(bus, "interpret", "start", session_id=session.session_id, detail=model)
            started = time.perf_counter()
            interpretation = None
            outcome = "unavailable"
            try:
                interpretation = llm.interpret(event.text, prior_turns, _profile_for_llm(profile))
                outcome = "ok" if interpretation is not None else "unavailable"
            except Exception:
                outcome = "error"
                raise
            finally:
                _activity(
                    bus,
                    "interpret",
                    "end",
                    session_id=session.session_id,
                    ok=outcome == "ok",
                    duration_ms=(time.perf_counter() - started) * 1000,
                    detail=f"{model} {outcome}",
                )
            if interpretation is not None:
                _decision_activity(
                    bus,
                    session.session_id,
                    {
                        "decision": "interpreted",
                        "intent": interpretation.intent.value,
                        "distress": interpretation.distress,
                        "text": event.text,
                        "phase": session.phase.value,
                        "goal": session.goal,
                    },
                )
                goal_before_interpretation = session.goal
                if not cooldown_reply:
                    interpreted = session.on_interpretation(
                        interpretation.intent.value, interpretation.distress, now
                    )
                    if interpreted is not None:
                        published.append(
                            _publish_transition(
                                bus, interpreted, session, now, profile=profile, llm=llm
                            )
                        )

                intent = interpretation.intent.value
                intentional_silence = False
                if session.phase in (Phase.ENGAGED, Phase.ESCALATED) or cooldown_reply:
                    reply_id = None
                    if cooldown_reply and interpretation.distress >= 2:
                        reply_id = REASSURE_WAITING_ID
                    elif intent == "wants_bed" and (
                        goal_before_interpretation == "return_to_bed"
                        or session.phase == Phase.ESCALATED
                    ):
                        reply_id = "acknowledge_return"
                    elif (
                        session.phase == Phase.ENGAGED
                        and session.goal == "restroom"
                        and interpretation.distress < 2
                        and intent in {"need_restroom", "unclear", "fine"}
                        and is_direct_question(event.text)
                    ):
                        # No directions from the floor, and no "someone is on
                        # their way" either: nobody has been alerted yet in
                        # ENGAGED. The on-floor rule escalates shortly.
                        reply_id = (
                            None if session.last_person_state == "on_floor" else PATH_LIGHT_ID
                        )
                    elif intent == "confused_time":
                        explicit_question = is_direct_question(event.text)
                        if explicit_question or not session.recently_said("orient_time_place", now):
                            reply_id = "orient_time_place"
                    elif intent == "need_restroom" and (
                        session.phase == Phase.ESCALATED or cooldown_reply
                    ):
                        reply_id = (
                            REASSURE_WAITING_ID
                            if session.last_person_state == "on_floor"
                            else PATH_LIGHT_ID
                        )
                    elif session.phase == Phase.ENGAGED and intent == "pain":
                        if not session._pain_acknowledged or is_direct_question(event.text):
                            reply_id = COMFORT_PAIN_ID
                        else:
                            _intentional_silence(bus, session, event.text, "pain_acknowledged")
                            intentional_silence = True
                    elif session.phase == Phase.ENGAGED and session.goal == "restroom":
                        if intent in {"fine", "unclear"} and interpretation.distress < 2:
                            if not session._progress_acknowledged:
                                reply_id = ACKNOWLEDGE_PROGRESS_ID
                            else:
                                _intentional_silence(
                                    bus, session, event.text, "progress_acknowledged"
                                )
                                intentional_silence = True
                    elif cooldown_reply and intent in {"looking_for_person", "pain"}:
                        reply_id = REASSURE_WAITING_ID
                    elif session.phase == Phase.ESCALATED:
                        reply_id = REASSURE_WAITING_ID
                    # A goal or phase transition may already have said or queued
                    # a sentence this tick. A reply must not add a second one.
                    reply_scheduled = session._last_say_at == now or (
                        session._pending_say is not None and session._pending_say.queued_at == now
                    )
                    if reply_id is not None and not reply_scheduled:
                        if reply_id == REASSURE_WAITING_ID and session.phase == Phase.ESCALATED:
                            _reassure_or_stay_silent(
                                bus,
                                session,
                                event.text,
                                now,
                                profile,
                                llm,
                                distress=interpretation.distress,
                            )
                        else:
                            _reply_to_utterance(bus, session, reply_id, now, profile, llm)
            elif (
                session.phase == Phase.ESCALATED or cooldown_reply
            ) and session._last_say_at != now:
                if session.phase == Phase.ESCALATED:
                    _reassure_or_stay_silent(bus, session, event.text, now, profile, llm)
                else:
                    _reply_to_utterance(bus, session, REASSURE_WAITING_ID, now, profile, llm)

            # Interpretation may just have escalated. A plan must never run
            # afterward and use an otherwise legal goal-reset edge to undo
            # ``wait_for_caregiver``.
            plan = (
                llm.plan(_session_state_for_llm(session), _profile_for_llm(profile))
                if session.phase == Phase.ENGAGED
                and session.last_person_state != "in_bed"
                and not session.compliance_hold(now)
                and (interpretation is None or interpretation.intent.value != "confused_time")
                else None
            )
            reply_scheduled = session._last_say_at == now or (
                session._pending_say is not None and session._pending_say.queued_at == now
            )
            if (
                plan is not None
                and not reply_scheduled
                and not (interpretation is not None and intentional_silence)
            ):
                if plan.goal_change is not None:
                    proposed = session.propose_goal(plan.goal_change, "llm_plan", now)
                    if proposed is not None:
                        published.append(
                            _publish_transition(
                                bus, proposed, session, now, profile=profile, llm=llm
                            )
                        )
                    else:
                        _log(
                            "rejected LLM goal proposal",
                            level=logging.WARNING,
                            goal=plan.goal_change,
                        )
                elif plan.next_strategy is not None:
                    proposed = session.propose_strategy(plan.next_strategy, now)
                    if proposed is not None:
                        published.append(
                            _publish_transition(
                                bus, proposed, session, now, profile=profile, llm=llm
                            )
                        )
                    else:
                        _log(
                            "rejected LLM strategy proposal",
                            level=logging.WARNING,
                            strategy=plan.next_strategy,
                        )
        elif (session.phase == Phase.ESCALATED or cooldown_reply) and session._last_say_at != now:
            if session.phase == Phase.ESCALATED:
                _reassure_or_stay_silent(bus, session, event.text, now, profile, None)
            else:
                _reply_to_utterance(bus, session, REASSURE_WAITING_ID, now, profile, None)

    now = now_fn()
    transition = session.tick(now)
    if transition is not None:
        published.append(
            _publish_transition(bus, transition, session, now, profile=profile, llm=llm)
        )

    _flush_pending_say(bus, session, now, profile)

    return published


def maybe_emit_session_heartbeat(
    bus,
    session: Session,
    last_emitted_at: datetime | None,
    now: datetime,
    *,
    interval: float,
) -> datetime | None:
    """Publish the current phase as `SessionState` if `interval` seconds have
    passed with no phase change to report -- the same reasoning, and the
    same shape, as `perceive.main.maybe_emit_person_heartbeat` (HANDOFF.md
    rule 4). Returns the (possibly updated) `last_emitted_at`.
    """
    if last_emitted_at is not None and (now - last_emitted_at).total_seconds() < interval:
        return last_emitted_at

    event = SessionState(
        source=SERVICE_NAME,
        session_id=session.session_id,
        phase=session.phase.value,
        goal=session.goal,
        strategy_index=session.strategy_index,
    )
    bus.publish(event)
    _log("published SessionState heartbeat", event_type="SessionState", phase=event.phase)
    return now


def maybe_emit_health(
    bus,
    last_emitted_at: float | None,
    now: float,
    *,
    interval: float = HEALTH_INTERVAL_S,
    ok: bool = True,
    detail: str = "running",
) -> float | None:
    """Publish a `Health` heartbeat if `interval` seconds have passed since
    the last one. Same shape as `capture.main.maybe_emit_health`."""
    if last_emitted_at is not None and now - last_emitted_at < interval:
        return last_emitted_at
    bus.publish(Health(source=SERVICE_NAME, service=SERVICE_NAME, ok=ok, detail=detail))
    _log("published Health", event_type="Health", ok=ok, detail=detail)
    return now


def _log_startup_timezone_check(config: AgentConfig) -> None:
    """Log one structured line naming the configured night window alongside
    the process's actually-resolved local time and zone.

    `AGENT_NIGHT_START`/`AGENT_NIGHT_END` are local `HH:MM`, but Docker
    containers default to UTC regardless of the host clock, so a missing
    or wrong `TZ` silently shifts the whole night window -- the agent
    would start and stop nudging at the wrong hour with nothing else
    indicating why. Logging `configured_tz`, the process's resolved
    `time.tzname`, and the current local time it resolves to right here at
    startup makes that mismatch visible in `docker compose logs agent`
    immediately, rather than discovered at 3am.
    """
    time.tzset()  # refresh time.tzname from the TZ env var, per the stdlib docs
    _log(
        "agent starting",
        night_start=config.night_start.isoformat(),
        night_end=config.night_end.isoformat(),
        configured_tz=os.environ.get("TZ", "UTC"),
        resolved_tzname=time.tzname,
        resolved_local_time=datetime.now().isoformat(),
    )


def run() -> None:
    """Connect to Redis and loop forever, driving the real session machine --
    unless `AGENT_FAKE=true`, in which case dispatch to `agent.fake.run`
    instead (the M0 demo fixture; see the module docstring).

    Reads `AGENT_NIGHT_START`, `AGENT_NIGHT_END`, `AGENT_OBSERVE_SECONDS`,
    `AGENT_COOLDOWN_SECONDS`, `AGENT_IN_BED_STABLE_SECONDS`,
    `AGENT_FLOOR_LIMIT_SECONDS`, `AGENT_ABSENT_LIMIT_SECONDS`,
    `AGENT_SAY_MIN_GAP_SECONDS`, `STRATEGIES_PATH`, `PERSON_PATH`,
    `REDIS_URL`, `ANTHROPIC_API_KEY`, and `TZ`
    from the environment (defaults documented in `.env.example`; `TZ`
    matters because `AGENT_NIGHT_START`/`AGENT_NIGHT_END` are local time
    and containers default to UTC otherwise). A short sleep between
    iterations when nothing was published avoids a busy loop.
    """
    if os.environ.get("AGENT_FAKE", "false").strip().lower() == "true":
        from agent.fake import run as run_fake

        run_fake()
        return

    config = AgentConfig.from_env()
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    _log_startup_timezone_check(config)

    strategies = load_strategies(config.strategies_path)
    profile = load_profile(config.person_path)
    bus = Bus(redis.Redis.from_url(redis_url))
    bus.ensure_group(PERSON_STREAM, PERSON_GROUP)
    bus.ensure_group(UTTERANCE_STREAM, UTTERANCE_GROUP)
    bus.ensure_group(DEBUG_STREAM, DEBUG_GROUP)
    session = Session(config=config, strategies=strategies)
    _publish_debug_state(bus, session)
    local_llm = build_local_llm(
        config.llm_backend,
        url=(
            os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
            if config.llm_backend == "ollama"
            else config.llm_url
        ),
        model=config.llm_model,
        timeout_seconds=config.llm_timeout_seconds,
    )
    warm_up = getattr(local_llm, "warm_up", None)
    if warm_up is not None:
        _log("local llm warm-up", model=config.llm_model, ok=warm_up())
    cloud_llm = None
    if profile.enable_cloud_fallback:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            _log(
                "cloud fallback enabled but ANTHROPIC_API_KEY is missing; using local model only",
                level=logging.WARNING,
            )
        else:

            def record_cloud_call(task: str, model: str, payload: dict[str, object]) -> None:
                _publish_cloud_call(bus, session, task, model, payload)

            cloud_llm = ClaudeLLM(api_key=api_key, on_call=record_cloud_call)
    llm = FallbackLLM(local_llm, cloud_llm)

    last_health_at: float | None = None
    last_heartbeat_at: datetime | None = None
    while True:
        published = run_once(bus, session, profile=profile, llm=llm)
        now = datetime.now()
        if published:
            last_heartbeat_at = now
        else:
            last_heartbeat_at = maybe_emit_session_heartbeat(
                bus, session, last_heartbeat_at, now, interval=HEARTBEAT_INTERVAL_S
            )
        last_health_at = maybe_emit_health(bus, last_health_at, time.time())
        if not published:
            time.sleep(0.05)


if __name__ == "__main__":
    run()
