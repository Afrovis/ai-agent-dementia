"""Entry point for the `agent` service (issue #4): the fake agent.

This is a dev-only publisher, not the real session/strategy core (that is
M2, issues 12 to 17). It walks the embodiment face through all four `Show`
states and a representative set of strategies from the goal tree
(`return_to_bed`, `restroom`, `drink_water`, `comfort`,
`wait_for_caregiver`; see HANDOFF.md sections 6 and 7, and PLAN.md section
5.2/5.3) so the embodiment page and dashboard can be built and demoed
without perception, an LLM, or a real session state machine
(HANDOFF.md section 8, M0 done criteria: "the fake agent cycles all
states").

`STEPS` is the ordered, looping fixture. `run_once` publishes exactly one
step's events and is the unit tested surface; `run()` is the infinite,
real-time loop that Docker actually runs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

import redis
from nc_shared.bus import Bus
from nc_shared.events import Say, Show

SERVICE_NAME = "agent"
FAKE_SESSION_ID = "fake-session"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4)."""
    logger.info(json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


@dataclass(frozen=True)
class Step:
    """One tick of the fake agent cycle: a `Show`, and optionally a `Say`."""

    face: str
    headline: str
    body: str
    brightness: float
    dwell_s: float
    photo_id: str | None = None
    say_text: str | None = None
    say_strategy: str | None = None
    interruptible: bool = True


# Ordered fixture walking through the four `Show.face` states and the goal
# tree's illustrative strategies (HANDOFF.md sections 6 to 7). The real
# strategy engine does not exist yet; text here is a dev-only stand-in so
# the face and dashboard have something to render.
STEPS: list[Step] = [
    Step(
        face="asleep",
        headline="It is night",
        body="Goal: none. Resting.",
        brightness=0.1,
        dwell_s=3,
    ),
    Step(
        face="awake",
        headline="Good, you're up",
        body="Goal: return_to_bed. Strategy: ambient_orient.",
        brightness=0.3,
        dwell_s=3,
    ),
    Step(
        face="awake",
        headline="Hello, Jean",
        body="Goal: return_to_bed. Strategy: soft_greeting.",
        brightness=0.5,
        dwell_s=3,
        say_text="Hello Jean, it's a quiet night.",
        say_strategy="soft_greeting",
    ),
    Step(
        face="awake",
        headline="It's just past 2 in the morning",
        body="Goal: return_to_bed. Strategy: orient_time_place, in your bedroom.",
        brightness=0.6,
        dwell_s=3,
        photo_id="room_familiar",
    ),
    Step(
        face="listening",
        headline="I'm listening",
        body="Goal: return_to_bed. Strategy: validate_and_redirect.",
        brightness=0.5,
        dwell_s=3,
    ),
    Step(
        face="speaking",
        headline="Let's head back to bed",
        body="Goal: return_to_bed. Strategy: guided_return.",
        brightness=0.7,
        dwell_s=3,
        say_text="Your bed is just a few steps this way.",
        say_strategy="guided_return",
    ),
    Step(
        face="awake",
        headline="Need the bathroom?",
        body="Goal: restroom. Strategy: path_light.",
        brightness=0.4,
        dwell_s=3,
    ),
    Step(
        face="speaking",
        headline="Here's some water",
        body="Goal: drink_water. Strategy: guided_return, then back to bed.",
        brightness=0.6,
        dwell_s=3,
        say_text="There's water on the nightstand.",
        say_strategy="guided_return",
    ),
    Step(
        face="awake",
        headline="A familiar voice",
        body="Goal: comfort. Strategy: familiar_voice.",
        brightness=0.5,
        dwell_s=3,
        photo_id="family_photo",
    ),
    Step(
        face="awake",
        headline="Someone is coming to help",
        body="Goal: wait_for_caregiver. Strategy: escalate_phone.",
        brightness=0.2,
        dwell_s=3,
        say_text="Someone is coming to help.",
        say_strategy="escalate_phone",
        interruptible=False,
    ),
    Step(
        face="asleep",
        headline="All calm",
        body="Cooldown. Back to rest.",
        brightness=0.05,
        dwell_s=3,
    ),
]


def run_once(bus, step: Step, session_id: str = FAKE_SESSION_ID) -> None:
    """Publish exactly one step's `Show` (and `Say`, if set) events on `bus`.

    Deterministic and side-effect-free beyond the publish itself, so tests
    can call it directly with a `FakeBus` instead of going through the
    infinite, real-time `run()` loop.
    """
    show = Show(
        source=SERVICE_NAME,
        session_id=session_id,
        face=step.face,
        headline=step.headline,
        body=step.body,
        photo_id=step.photo_id,
        brightness=step.brightness,
    )
    bus.publish(show)
    _log("published Show", event_type="Show", face=step.face, headline=step.headline)

    if step.say_text is not None:
        say = Say(
            source=SERVICE_NAME,
            session_id=session_id,
            text=step.say_text,
            strategy=step.say_strategy or "",
            interruptible=step.interruptible,
        )
        bus.publish(say)
        _log("published Say", event_type="Say", strategy=say.strategy)


def run() -> None:
    """Loop `STEPS` forever in real time, publishing each step then sleeping.

    This is the fake agent required for M0 (HANDOFF.md section 8): it
    cycles all `Show.face` states and a representative strategy per goal so
    `docker compose up` shows the face changing with no perception, LLM, or
    real session state machine in the loop.
    """
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")
    bus = Bus(redis.Redis.from_url(redis_url))
    _log("fake agent starting, cycling steps", step_count=len(STEPS))

    while True:
        for step in STEPS:
            run_once(bus, step)
            time.sleep(step.dwell_s)


if __name__ == "__main__":
    run()
