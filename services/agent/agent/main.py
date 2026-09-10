"""Entry point for the `agent` service (issue #12): the real session core.

Reads `PersonState` from the `person` stream and `Utterance` from
`speech_in`, both through Redis consumer groups, and drives an
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
  validation is logged loudly and never published -- silence, not a wrong
  sentence at 3am. Composing strategy 4's text from what the person
  actually said is `compose` (issue #15); every strategy here uses a
  fixed, caregiver-editable template instead (see `agent.strategies`'s
  module docstring), so nothing here fabricates dialogue ahead of that.
- `Notify(critical, repeat_until_ack=True, source="agent")` on entering
  `ESCALATED`, built from the `NotifySpec` `agent.session.Session` attaches
  to that `Transition`.
- `Health`, matching `perceive`/`capture`.

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
from datetime import datetime

import redis
from nc_shared.bus import Bus
from nc_shared.events import (
    GoalChanged,
    Health,
    Notify,
    PersonState,
    Say,
    SessionState,
    Show,
    Utterance,
)

from agent.config import AgentConfig
from agent.rules import Phase, validate_say
from agent.session import Session, Transition
from agent.strategies import (
    DEFAULT_PROFILE,
    ESCALATE_PHONE_ID,
    PersonProfile,
    StrategyDef,
    load_strategies,
    render_template,
    time_as_words,
)

SERVICE_NAME = "agent"
HEALTH_INTERVAL_S = 30.0
HEARTBEAT_INTERVAL_S = 60.0

PERSON_STREAM = "person"
PERSON_GROUP = "agent"
UTTERANCE_STREAM = "speech_in"
UTTERANCE_GROUP = "agent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, level: int = logging.INFO, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4)."""
    logger.log(level, json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


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


def _maybe_publish_say(
    bus,
    transition: Transition,
    session: Session,
    now: datetime,
    profile: PersonProfile,
) -> None:
    """Publish a `Say` for `transition.strategy`, if it has a
    `say_template`, after passing it through `agent.rules.validate_say`
    (HANDOFF.md rule 3). A rejection is logged loudly and nothing is
    published -- silence, never a wrong sentence at 3am -- and does not
    move `session`'s minimum-gap clock, since nothing was actually said.
    """
    strategy = transition.strategy
    if strategy is None or strategy.say_template is None:
        return

    text = render_template(strategy.say_template, profile, time_words=time_as_words(now))
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
    seconds_since_last_say = None if strategy.terminal else session.seconds_since_last_say(now)
    result = validate_say(
        text,
        seconds_since_last_say=seconds_since_last_say,
        min_gap_seconds=session.config.say_min_gap_seconds,
    )
    if not result.accepted:
        _log(
            "rejected Say, falling back to silence",
            level=logging.WARNING,
            event_type="Say",
            strategy=strategy.id,
            reason=result.reason,
        )
        return

    say_event = Say(
        source=SERVICE_NAME,
        session_id=transition.session_id,
        text=text,
        strategy=strategy.id,
        # `escalate_phone` must not be interrupted by barge-in the way an
        # ordinary strategy's speech can be (HANDOFF.md section 7:
        # `listen`'s barge-in) -- there is nothing left to redirect to.
        interruptible=strategy.id != ESCALATE_PHONE_ID,
    )
    bus.publish(say_event)
    session.record_say(now)
    _log("published Say", event_type="Say", strategy=strategy.id)


def _publish_transition(
    bus,
    transition: Transition,
    session: Session,
    now: datetime,
    *,
    profile: PersonProfile = DEFAULT_PROFILE,
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
    behind. `profile` is issue #16's seam: always `DEFAULT_PROFILE` until
    a real person profile exists to load (see `agent.strategies`'s module
    docstring)."""
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

    if transition.notify is not None:
        notify_event = Notify(
            source=SERVICE_NAME,
            session_id=transition.session_id,
            level=transition.notify.level,
            title=transition.notify.title,
            body=transition.notify.body,
            repeat_until_ack=True,
        )
        bus.publish(notify_event)
        _log("published Notify", event_type="Notify", notify_level=notify_event.level)

    show_event = _show_for_transition(transition, now, profile)
    bus.publish(show_event)
    _log("published Show", event_type="Show", face=show_event.face)

    _maybe_publish_say(bus, transition, session, now, profile)

    return event


def run_once(
    bus,
    session: Session,
    *,
    consumer: str = "agent-1",
    count: int = 10,
    block_ms: int = 200,
    now_fn: Callable[[], datetime] = datetime.now,
    profile: PersonProfile = DEFAULT_PROFILE,
) -> list[SessionState]:
    """Read whatever `PersonState`/`Utterance` messages are waiting, feed
    them through `session` in arrival order, and publish one `SessionState`
    (plus any `Notify`/`Show`/`Say`) per resulting update. Also calls
    `session.tick` once, so timers -- including a strategy's dwell -- advance
    even when nothing new arrived.

    Returns the `SessionState`s published, most recent last. Side-effect-
    free beyond bus reads/acks/publishes, so tests can call it directly and
    in a loop with a `FakeBus`, instead of going through the infinite,
    real-time `run()` loop. `profile` defaults to `DEFAULT_PROFILE`
    (issue #16's seam, see `agent.strategies`).
    """
    published: list[SessionState] = []

    person_messages = bus.read(
        PERSON_STREAM, PERSON_GROUP, consumer, count=count, block_ms=block_ms
    )
    for msg_id, event in person_messages:
        bus.ack(PERSON_STREAM, PERSON_GROUP, msg_id)
        assert isinstance(event, PersonState)
        now = now_fn()
        transition = session.on_person_state(event.state, event.zone, now)
        if transition is not None:
            published.append(_publish_transition(bus, transition, session, now, profile=profile))

    utterance_messages = bus.read(
        UTTERANCE_STREAM, UTTERANCE_GROUP, consumer, count=count, block_ms=0
    )
    for msg_id, event in utterance_messages:
        bus.ack(UTTERANCE_STREAM, UTTERANCE_GROUP, msg_id)
        assert isinstance(event, Utterance)
        now = now_fn()
        transition = session.on_utterance(now)
        if transition is not None:
            published.append(_publish_transition(bus, transition, session, now, profile=profile))

    now = now_fn()
    transition = session.tick(now)
    if transition is not None:
        published.append(_publish_transition(bus, transition, session, now, profile=profile))

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
    `AGENT_SAY_MIN_GAP_SECONDS`, `STRATEGIES_PATH`, `REDIS_URL`, and `TZ`
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
    bus = Bus(redis.Redis.from_url(redis_url))
    bus.ensure_group(PERSON_STREAM, PERSON_GROUP)
    bus.ensure_group(UTTERANCE_STREAM, UTTERANCE_GROUP)
    session = Session(config=config, strategies=strategies)

    last_health_at: float | None = None
    last_heartbeat_at: datetime | None = None
    while True:
        published = run_once(bus, session)
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
