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
- `Show`, minimally: a dim clock at `IDLE`/`COOLDOWN`, the `awake` face at
  `OBSERVING`/`ENGAGED`/`ESCALATED`. HANDOFF.md section 7's strategy
  catalogue -- a distinct `Show` per strategy -- is issue #14; this issue
  owns only the phase machine, so this is deliberately just enough to make
  the phase visible on the embodiment page, not a strategy.
- No `Say`, ever. HANDOFF.md rule 3 governs spoken text precisely (one
  sentence, no memory-testing questions, never "no"/"you can't"/"you're
  wrong"), and composing that text is `compose` (issue #15), chosen by a
  strategy (issue #14). Emitting anything here would be inventing dialogue
  ahead of both strategy selection and composition, which HANDOFF.md
  section 11 rules out ("no fabricated behaviour"). The safest reading of
  issue #12's scope is silence until #14/#15 exist to fill it in properly.
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
from nc_shared.events import GoalChanged, Health, Notify, PersonState, SessionState, Show, Utterance

from agent.config import AgentConfig
from agent.rules import Phase
from agent.session import Session, Transition

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
    """The minimal, honest `Show` for `phase`. See the module docstring for
    why this is intentionally not a strategy: issue #14 owns those."""
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
        return Show(
            source=SERVICE_NAME,
            session_id=session_id,
            face="awake",
            headline="Someone is coming to help",
            body="",
            brightness=0.2,
        )
    # OBSERVING or ENGAGED: awake, no strategy text yet (issue #14).
    return Show(
        source=SERVICE_NAME,
        session_id=session_id,
        face="awake",
        headline="",
        body="",
        brightness=0.4,
    )


def _publish_transition(bus, transition: Transition) -> SessionState:
    """Publish everything one `Transition` implies: `SessionState`, an
    optional `GoalChanged` (issue #13), an optional `Notify`, and the
    phase's `Show`. Returns the `SessionState` published, for callers that
    just want to know what happened.

    `SessionState` is always republished here, even for a goal-only
    update with no phase change, so `SessionState.goal` -- shown on the
    dashboard timeline alongside `GoalChanged` -- never lags behind."""
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

    show_event = _show_for_phase(transition.phase, transition.session_id)
    bus.publish(show_event)
    _log("published Show", event_type="Show", face=show_event.face)

    return event


def run_once(
    bus,
    session: Session,
    *,
    consumer: str = "agent-1",
    count: int = 10,
    block_ms: int = 200,
    now_fn: Callable[[], datetime] = datetime.now,
) -> list[SessionState]:
    """Read whatever `PersonState`/`Utterance` messages are waiting, feed
    them through `session` in arrival order, and publish one `SessionState`
    (plus any `Notify`/`Show`) per resulting phase change. Also calls
    `session.tick` once, so timers advance even when nothing new arrived.

    Returns the `SessionState`s published, most recent last. Side-effect-
    free beyond bus reads/acks/publishes, so tests can call it directly and
    in a loop with a `FakeBus`, instead of going through the infinite,
    real-time `run()` loop.
    """
    published: list[SessionState] = []

    person_messages = bus.read(
        PERSON_STREAM, PERSON_GROUP, consumer, count=count, block_ms=block_ms
    )
    for msg_id, event in person_messages:
        bus.ack(PERSON_STREAM, PERSON_GROUP, msg_id)
        assert isinstance(event, PersonState)
        transition = session.on_person_state(event.state, event.zone, now_fn())
        if transition is not None:
            published.append(_publish_transition(bus, transition))

    utterance_messages = bus.read(
        UTTERANCE_STREAM, UTTERANCE_GROUP, consumer, count=count, block_ms=0
    )
    for msg_id, event in utterance_messages:
        bus.ack(UTTERANCE_STREAM, UTTERANCE_GROUP, msg_id)
        assert isinstance(event, Utterance)
        transition = session.on_utterance(now_fn())
        if transition is not None:
            published.append(_publish_transition(bus, transition))

    transition = session.tick(now_fn())
    if transition is not None:
        published.append(_publish_transition(bus, transition))

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
    `AGENT_FLOOR_LIMIT_SECONDS`, `AGENT_ABSENT_LIMIT_SECONDS`, `REDIS_URL`,
    and `TZ` from the environment (defaults documented in `.env.example`;
    `TZ` matters because `AGENT_NIGHT_START`/`AGENT_NIGHT_END` are local
    time and containers default to UTC otherwise). A short sleep between
    iterations when nothing was published avoids a busy loop.
    """
    if os.environ.get("AGENT_FAKE", "false").strip().lower() == "true":
        from agent.fake import run as run_fake

        run_fake()
        return

    config = AgentConfig.from_env()
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    _log_startup_timezone_check(config)

    bus = Bus(redis.Redis.from_url(redis_url))
    bus.ensure_group(PERSON_STREAM, PERSON_GROUP)
    bus.ensure_group(UTTERANCE_STREAM, UTTERANCE_GROUP)
    session = Session(config=config)

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
