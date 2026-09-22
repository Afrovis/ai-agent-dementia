"""`agent`'s env-driven configuration (issue #12).

Same shape as `perceive.main.PerceiveConfig` and `capture.main.CaptureConfig`:
a frozen dataclass, a `from_env` classmethod (env, then defaults in code --
the profile and strategy catalogues have their own YAML loaders), and every field documented in
`.env.example`.

`REDIS_URL` is read directly in `agent.main.run`, the same way `perceive`
and `capture` keep it out of their config dataclasses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dt_time

from agent.llm import LOCAL_BACKENDS


def _parse_hhmm(value: str) -> dt_time:
    """Parse a local `HH:MM` string (`AGENT_NIGHT_START`/`AGENT_NIGHT_END`)."""
    hours, minutes = value.strip().split(":")
    return dt_time(int(hours), int(minutes))


def _parse_llm_backend(value: str) -> str:
    """Validate `AGENT_LLM_BACKEND` at startup so a typo fails loudly."""
    backend = value.strip().lower()
    if backend not in LOCAL_BACKENDS:
        raise ValueError(f"AGENT_LLM_BACKEND must be one of {LOCAL_BACKENDS}, got {value!r}")
    return backend


@dataclass(frozen=True)
class AgentConfig:
    """`agent`'s tunables (HANDOFF.md section 6 and the issue #12 brief)."""

    night_start: dt_time = dt_time(21, 0)
    """Local time the night window opens. A session may only *start* inside
    the window (HANDOFF.md: "Enter OBSERVING ... inside the night window").
    `AGENT_NIGHT_START`."""

    night_end: dt_time = dt_time(7, 0)
    """Local time the night window closes. `AGENT_NIGHT_END`."""

    observe_seconds: float = 20.0
    """How long `OBSERVING` waits for the person to either settle back
    `in_bed` (-> `IDLE`) or stay up (-> `ENGAGED`). HANDOFF.md section 11
    is explicit that this delay must stay configurable rather than be
    skipped for a faster demo. `AGENT_OBSERVE_SECONDS`."""

    cooldown_seconds: float = 300.0
    """How long `COOLDOWN` holds before returning to `IDLE`, during which
    no new session may start. `AGENT_COOLDOWN_SECONDS`."""

    in_bed_stable_seconds: float = 120.0
    """How long `in_bed` must hold, unbroken, before the machine leaves
    `ENGAGED`/`ESCALATED` for `COOLDOWN`. `AGENT_IN_BED_STABLE_SECONDS`."""

    floor_limit_seconds: float = 0.0
    """How long `on_floor` is tolerated, from any phase with a live session,
    before HANDOFF.md rule 5 fires and skips straight to `ESCALATED`.
    Defaults to `0`, i.e. escalate on the very first classified frame:
    `on_floor` is the single highest-risk state this system observes, and
    unlike `absent` there is no benign everyday reason for it, so there is
    no grace period to justify. `AGENT_FLOOR_LIMIT_SECONDS`."""

    absent_limit_seconds: float = 600.0
    """How long `absent` is tolerated before rule 5 fires the same way.
    Ten minutes, not zero, because someone out of camera view at night is
    ordinarily just using the bathroom -- normal and expected -- whereas
    lying on the floor never is; that asymmetry is why `absent` gets a
    grace period and `on_floor` does not. `AGENT_ABSENT_LIMIT_SECONDS`."""

    utterance_presence_seconds: float = 30.0
    """A complete utterance establishes presence for this long despite an absent camera reading."""

    restroom_timeout_seconds: float = 900.0
    """How long the `restroom` goal (issue #13) is allowed to sit unresolved
    before `agent.session.Session` gives up waiting for the person to be
    seen back at the bed and returns the goal to `return_to_bed` on its
    own. This is a safety valve against the goal sticking forever, not a
    claim that 15 minutes has any significance for actual bathroom use --
    see `agent.goals`'s module docstring. `AGENT_RESTROOM_TIMEOUT_SECONDS`.
    """

    strategies_path: str | None = None
    """Path to the caregiver-editable strategy catalogue yaml (issue #14),
    passed straight to `agent.strategies.load_strategies` (which applies
    the `STRATEGIES_PATH` env var itself if this is left `None` -- kept
    out of the `from_env`-computed default below the same way `perceive.
    main.PerceiveConfig.zones_path` keeps `ZONES_PATH` out of its own
    `from_env`, so a caller passing an explicit path in a test does not
    also have to fight this field's own env lookup). `STRATEGIES_PATH`."""

    person_path: str | None = None
    """Path to the caregiver-authored person profile (issue #16). If unset,
    `agent.profile.load_profile` resolves `PERSON_PATH`, then
    `config/person.yaml`, then `config/person.example.yaml`. `PERSON_PATH`."""

    say_min_gap_seconds: float = 8.0
    compliance_grace_seconds: float = 120.0
    repeat_window_seconds: float = 120.0
    """The minimum silence, in seconds, `agent.rules.validate_say` requires
    between one published `Say` and the next (HANDOFF.md rule 3: "Spoken
    output is one sentence, then silence for at least 8 seconds").
    `AGENT_SAY_MIN_GAP_SECONDS`."""

    zone_confirm_readings: int = 3
    """How many consecutive `PersonState` readings must agree on a zone
    before that zone may drive a goal change (issue #13). `perceive`
    computes `PersonState.zone` from a bare centroid-to-polygon lookup with
    no hysteresis of its own -- only `state` gets that treatment, via
    `perceive.classify.StateTracker.confirm_frames` -- and `bathroom_path`
    and `bed` sit right next to each other, exactly where someone stands
    at the start of a bathroom trip, so a single noisy reading must not be
    enough to flip the goal and spam the caregiver's timeline with
    `GoalChanged` events. `3`, the same default `PERCEIVE_CONFIRM_FRAMES`
    uses for the analogous reason on `state`. Deliberately not applied to
    HANDOFF.md rule 5, which must keep firing on the very first reading.
    `AGENT_ZONE_CONFIRM_READINGS`."""

    llm_model: str = "gemma4:e4b-mlx"
    """Local text model used for interpret/compose/plan (issue #15): an
    Ollama tag, or the served model id for the `openai` backend. The default
    is the model `perceive`'s floor check already loads, so the two share one
    copy in memory; it was chosen on the dialogue bench (`tools/llm_speedtest`).
    `AGENT_LLM_MODEL`."""

    llm_timeout_seconds: float = 10.0
    """Hard timeout for one local text-model request. The latency budgets
    remain under one second for interpret and under two seconds to first
    token for compose/plan; this larger ceiling prevents a cold model load
    from crashing the agent while still bounding a failed request.
    `AGENT_LLM_TIMEOUT_SECONDS`."""

    llm_backend: str = "ollama"
    """Local text-model transport: `ollama` (reads `OLLAMA_URL`) or `openai`,
    an OpenAI-compatible server on the host such as `mlx_lm.server` (reads
    `AGENT_LLM_URL`). `AGENT_LLM_BACKEND`."""

    llm_url: str = "http://host.docker.internal:11435"
    """Base URL of the `openai` backend; unused for `ollama`. `AGENT_LLM_URL`."""

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AgentConfig:
        """Build an `AgentConfig` from environment variables, defaults otherwise."""
        env = os.environ if env is None else env
        return cls(
            night_start=_parse_hhmm(env.get("AGENT_NIGHT_START", "21:00")),
            night_end=_parse_hhmm(env.get("AGENT_NIGHT_END", "07:00")),
            observe_seconds=float(env.get("AGENT_OBSERVE_SECONDS", "20")),
            cooldown_seconds=float(env.get("AGENT_COOLDOWN_SECONDS", "300")),
            in_bed_stable_seconds=float(env.get("AGENT_IN_BED_STABLE_SECONDS", "120")),
            floor_limit_seconds=float(env.get("AGENT_FLOOR_LIMIT_SECONDS", "0")),
            absent_limit_seconds=float(env.get("AGENT_ABSENT_LIMIT_SECONDS", "600")),
            restroom_timeout_seconds=float(env.get("AGENT_RESTROOM_TIMEOUT_SECONDS", "900")),
            strategies_path=env.get("STRATEGIES_PATH"),
            person_path=env.get("PERSON_PATH") or None,
            say_min_gap_seconds=float(env.get("AGENT_SAY_MIN_GAP_SECONDS", "8")),
            compliance_grace_seconds=float(env.get("AGENT_COMPLIANCE_GRACE_SECONDS", "120")),
            repeat_window_seconds=float(env.get("AGENT_REPEAT_WINDOW_SECONDS", "120")),
            zone_confirm_readings=int(env.get("AGENT_ZONE_CONFIRM_READINGS", "3")),
            llm_model=env.get("AGENT_LLM_MODEL") or "gemma4:e4b-mlx",
            llm_timeout_seconds=float(env.get("AGENT_LLM_TIMEOUT_SECONDS") or "10"),
            llm_backend=_parse_llm_backend(env.get("AGENT_LLM_BACKEND") or "ollama"),
            llm_url=env.get("AGENT_LLM_URL") or "http://host.docker.internal:11435",
        )

    def in_night_window(self, when: datetime) -> bool:
        """Whether `when`'s local time of day falls inside the night window.

        The window wraps midnight whenever `night_end` is earlier in the
        day than `night_start` (the default `21:00`-`07:00` always does):
        in that case "inside the window" means "at or after `night_start`,
        *or* before `night_end`", not a simple `start <= t < end` range.

        `night_start == night_end` is treated as an always-on 24-hour
        window rather than the empty one a literal `start <= t < end`
        would produce. A zero-width window is not a state anyone would
        deliberately configure -- it silently disables session start
        entirely, forever -- so the safe reading of "someone set both the
        same" is "no window restriction", not "never".
        """
        if self.night_start == self.night_end:
            return True
        t = when.time()
        if self.night_start < self.night_end:
            return self.night_start <= t < self.night_end
        return t >= self.night_start or t < self.night_end
