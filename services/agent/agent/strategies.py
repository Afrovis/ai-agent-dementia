"""The strategy catalogue and selection engine for `agent` (issue #14).

HANDOFF.md section 7 and PLAN.md section 5.3 list ten strategies. This
module implements strategies 1 to 6, 8, and 10:

- `ambient_orient` (1), `soft_greeting` (2), `orient_time_place` (3),
  `validate_and_redirect` (4), `guided_return` (5), `familiar_voice` (6):
  the ordinary, escalating-intrusiveness ladder HANDOFF.md's phase summary
  means by "run strategies in configured order" while `ENGAGED`.
- `path_light` (8): selected specifically when the `restroom` goal starts,
  and excluded from the ordinary return-to-bed ladder.
- `escalate_phone` (10): the terminal strategy selected on `ESCALATED`,
  see `StrategyEngine.force` and its docstring for why it is forced rather
  than chosen through the ordinary ladder.

Deliberately absent, and not faked:

- `music_or_story` (7) needs a caregiver track-upload path that does not
  exist yet. Out of scope for every current issue, not just this one.
- `escalate_gentle` (9) needs in-home chime hardware PLAN.md marks v2.
  Out of scope for every current issue.

## Where the LLM does and does not sit

Nothing in this module calls an LLM (HANDOFF.md rule 1). It exposes bounded
selection seams to callers that handle model proposals:

- `validate_and_redirect`'s caregiver template is the safe fallback for
  issue #15's local composition; `agent.main` performs that call and runs
  either result through `validate_say`.
- `StrategyEngine.propose_next` accepts a planner suggestion only when it
  is the exact next configured, enabled, off-cooldown, non-terminal rung.
- Every strategy's `say_template`/`headline_template`/`body_template` is
  caregiver-editable text (`config/strategies.example.yaml`), rendered
  through `render_template` below with the `PersonProfile` loaded by
  `agent.profile` (issue #16). Missing profile fields degrade to a generic
  phrase, never to a literal `{name}` shown to the person; see `render_template`.

## The engine

`StrategyEngine` owns exactly the state a caregiver's config cannot: which
strategy is current, when it started (for its dwell timer), and each
strategy's cooldown clock. Ordering, enable/disable, cooldown seconds,
dwell seconds, and every phrase come from `load_strategies` (config), not
from constants baked into the engine.

Every candidate the engine considers, whether starting a fresh session or
advancing past a dwell-elapsed one, is run through `agent.rules.
validate_strategy` before it is selected -- HANDOFF.md rule 1's "every
proposal passes through the rule layer" applied to strategy choice, not
just phase transitions. A disabled or on-cooldown strategy is rejected
there and skipped, an ordinary outcome, not an exception.

`escalate_phone` is the one exception to that flow: `StrategyEngine.force`
selects it directly, bypassing enabled/cooldown checks entirely, when
`agent.session.Session` enters `ESCALATED`. See `force`'s docstring for
why -- the reasoning is the same one `agent.goals` already gives for
`wait_for_caregiver` staying active rather than being resolved by code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from agent.profile import DEFAULT_PROFILE as DEFAULT_PROFILE
from agent.profile import PersonProfile
from agent.rules import validate_strategy

SERVICE_NAME = "agent"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)

ESCALATE_PHONE_ID = "escalate_phone"
PATH_LIGHT_ID = "path_light"
GUIDED_RETURN_ID = "guided_return"
FAMILIAR_VOICE_ID = "familiar_voice"

DEFAULT_VOICE_CLIP_DIR = Path("/app/data/voice-clips")
VOICE_CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

MAX_PROGRESS_DWELL_MULTIPLIER = 3.0
"""How far "progress" may stretch one strategy's dwell before the ladder
moves on regardless (`StrategyEngine.note_progress`).

`agent.session.Session._is_progress` counts standing in the bed zone as
progress, and re-arms the dwell on every `PersonState` that says so. Some
of those readings are a person genuinely climbing back into bed, and
interrupting them would be the wrong thing. But a person who stands at the
bedside without getting in produces exactly the same readings forever, and
without a cap the ladder pins on its current strategy for the rest of the
night -- silently, if that strategy is `ambient_orient` -- and never
reaches escalation either, since rule 5 sees neither `on_floor` nor
`absent`. Three dwells is long enough to let a slow, real return finish
and short enough that a stalled one still moves.
"""

DEFAULT_STRATEGIES_DIR = Path("config")
DEFAULT_STRATEGIES_FILENAME = "strategies.yaml"
DEFAULT_STRATEGIES_EXAMPLE_FILENAME = "strategies.example.yaml"


@dataclass(frozen=True)
class StrategyDef:
    """One strategy: everything config-driven about it in one place.

    `headline_template`/`body_template`/`say_template` are plain strings
    with `{name}`/`{caregiver_name}`/`{time_words}` placeholders, rendered
    by `render_template` at publish time (`agent.main`), never baked in
    ahead of time -- a caregiver edits these in
    `config/strategies.yaml` without touching code. `say_template=None`
    means the strategy never speaks (`ambient_orient`, per HANDOFF.md
    section 7's table: "Say: none").
    """

    id: str
    order: int
    enabled: bool
    intrusiveness: int
    dwell_seconds: float
    cooldown_seconds: float
    face: str
    brightness: float
    headline_template: str
    body_template: str
    say_template: str | None = None
    photo_id: str | None = None
    clip_id: str | None = None
    goal_only: bool = False
    """Exclude this strategy from the ordinary ladder.

    Goal-only strategies are selected through `select_for_goal`, which
    still enforces caregiver enablement and cooldown through the rule layer.
    """
    terminal: bool = False
    """Whether this strategy may only ever be selected through
    `StrategyEngine.force`, never by the ordinary ladder.

    `escalate_phone` is the only one (issue #14). It is the sentence that
    tells the person help is on the way, and saying that is only honest
    once `agent.session.Session` has actually entered `ESCALATED` and
    published the `Notify` that summons a caregiver. Reaching it by simply
    running out of ordinary strategies would speak those words with nobody
    told, so `StrategyEngine._first_available_id` skips every terminal
    strategy and "strategies exhausted" fires instead. Deliberately not
    overridable from yaml: this is a safety property of the ladder, not a
    caregiver preference.
    """


# The ladder, HANDOFF.md section 7 / PLAN.md section 5.3, in catalogue
# order. Cooldown values are this module's own sensible defaults -- the
# catalogue only specifies dwell and intrusiveness -- chosen so a strategy
# does not come back around again within the same short session (roughly
# 2x its dwell for the low-intrusiveness ladder entries, longer for the
# ones a caregiver would want to see used sparingly) while staying
# reselectable in a later session; a caregiver can change every one of
# these in `config/strategies.yaml`.
#
# Every `say_template` below is deliberately one sentence, per HANDOFF.md
# rule 3 ("Spoken output is one sentence, then silence for at least 8
# seconds"). PLAN.md section 5.3's illustrative phrasing for
# `orient_time_place` and `validate_and_redirect` predates that rule and is
# two sentences ("You are home, in your bedroom. Everyone is sleeping.");
# rule 3 is non-negotiable and wins, so do not "restore" that phrasing --
# `tests/test_strategies.py::test_every_default_strategy_template_passes_validate_say`
# fails immediately if a future edit reintroduces a second sentence.
DEFAULT_STRATEGIES: tuple[StrategyDef, ...] = (
    StrategyDef(
        id="ambient_orient",
        order=1,
        enabled=True,
        intrusiveness=1,
        dwell_seconds=30.0,
        cooldown_seconds=120.0,
        face="awake",
        brightness=0.3,
        headline_template="It is night",
        # `{name_vocative}` (see `render_template`) is the mechanism that
        # keeps this natural whether or not a name is configured: ", Jean"
        # when it is, nothing at all -- not a dangling ", there" -- when
        # it is not.
        body_template="It is {time_words}{name_vocative}.",
        say_template=None,
    ),
    StrategyDef(
        id="soft_greeting",
        order=2,
        enabled=True,
        intrusiveness=2,
        dwell_seconds=20.0,
        cooldown_seconds=120.0,
        face="awake",
        brightness=0.5,
        # `{name}` (not `{name_vocative}`) is deliberate here: "Hello,
        # there" and "Hello there, it's ..." are both natural English on
        # their own, unlike a name dropped into the middle of a sentence,
        # so the plain, always-present fallback ("there") works as-is.
        headline_template="Hello, {name}",
        body_template="It is {time_words}.",
        say_template="Hello {name}, it's {time_words}.",
    ),
    StrategyDef(
        id="orient_time_place",
        order=3,
        enabled=True,
        intrusiveness=2,
        dwell_seconds=30.0,
        cooldown_seconds=180.0,
        face="awake",
        brightness=0.5,
        headline_template="You are home",
        # One sentence (rule 3) -- see the module-level comment above this
        # tuple for why this is not PLAN.md's original two-sentence text.
        body_template="You are in your bedroom, and it is {time_words}.",
        say_template="You are home in your bedroom, and it is {time_words}.",
        photo_id="demo_room",
    ),
    StrategyDef(
        id="validate_and_redirect",
        order=4,
        enabled=True,
        intrusiveness=3,
        dwell_seconds=20.0,
        cooldown_seconds=180.0,
        face="listening",
        brightness=0.5,
        headline_template="I'm listening",
        body_template="",
        # HANDOFF.md's catalogue description, "composed from utterance",
        # is `interpret`/`compose` (issue #15). This is the caregiver's
        # fixed fallback template: it validates a feeling in general terms
        # and redirects, without inventing any acknowledgement of what the
        # person actually said (HANDOFF.md section 11: no fabricated
        # behaviour). One sentence (rule 3) -- see the module-level
        # comment above this tuple. `{name_vocative}` degrades to nothing
        # rather than a bare ", there" when no name is configured.
        say_template="It's alright{name_vocative}, let's rest now and talk more in the morning.",
    ),
    StrategyDef(
        id=GUIDED_RETURN_ID,
        order=5,
        enabled=True,
        intrusiveness=3,
        dwell_seconds=30.0,
        cooldown_seconds=180.0,
        face="speaking",
        brightness=0.7,
        headline_template="Let's head back to bed",
        body_template="Your bed is behind you.",
        say_template="Let's go back to bed now{name_vocative}.",
    ),
    StrategyDef(
        id=FAMILIAR_VOICE_ID,
        order=6,
        enabled=False,
        intrusiveness=3,
        # Replaced with the WAV duration plus 15 seconds when selected.
        dwell_seconds=15.0,
        cooldown_seconds=300.0,
        face="speaking",
        brightness=0.6,
        headline_template="A message from your family",
        body_template="Here is a familiar voice for you.",
        say_template="Here is a familiar voice for you.",
        photo_id="demo_family",
    ),
    StrategyDef(
        id=PATH_LIGHT_ID,
        order=8,
        enabled=True,
        intrusiveness=2,
        # The physical light remains on for the whole restroom goal. This
        # dwell only controls the screen/speech strategy if session updates
        # continue before the person returns.
        dwell_seconds=900.0,
        cooldown_seconds=0.0,
        face="speaking",
        brightness=0.6,
        headline_template="The restroom path is lit",
        body_template="{restroom_direction}.",
        say_template="{restroom_direction}{name_vocative}.",
        goal_only=True,
    ),
    StrategyDef(
        id=ESCALATE_PHONE_ID,
        order=10,
        enabled=True,
        intrusiveness=5,
        # "Until Ack" (HANDOFF.md section 7): the engine never advances
        # past this strategy on a dwell timer at all (see
        # `agent.session.Session._run_strategy_engine`), so this value is
        # documentation, not a timer anything reads.
        dwell_seconds=float("inf"),
        cooldown_seconds=0.0,
        # `speaking` at the same brightness as `guided_return`, the most
        # intrusive ordinary rung. The earlier `asleep`/0.15 pairing was
        # dimmer and less awake than every strategy preceding it, and
        # dimmer than the phase-level `ESCALATED` fallback in
        # `agent.main._show_for_phase` (`awake`/0.2) that it replaced --
        # the wrong direction for the one moment the person most needs to
        # see and hear the screen. Not raised past `guided_return` either:
        # this fires at 3am, possibly at someone on the floor.
        face="speaking",
        brightness=0.7,
        headline_template="Someone is coming to help",
        body_template="",
        say_template="Someone is coming to help.",
        # Force-only: see `StrategyDef.terminal`.
        terminal=True,
    ),
)


_UNSET_NAME = "there"
"""`PersonProfile.name`'s default. Deliberately not empty: "Hello, there"
and "Hello there, it's ..." both read as natural, friendly English on
their own (see `soft_greeting` in `DEFAULT_STRATEGIES`), so a template that
uses the raw `{name}` placeholder can keep doing so. It is a reserved
sentinel for "no name configured", not a literal name -- `render_template`
compares against it to compute `{name_vocative}` (see there), so a
caregiver should not configure an actual person's name as the string
"there"; this is a known, accepted limitation of the sentinel approach."""


class _SafeFormatDict(dict):
    """`dict` subclass for `str.format_map` that degrades a missing key to
    an empty string instead of raising `KeyError` -- the mechanism behind
    `render_template`'s "never a literal `{name}`" guarantee."""

    def __missing__(self, key: str) -> str:
        return ""


def render_template(template: str, profile: PersonProfile, **extra: str) -> str:
    """Fill `template`'s `{name}`/`{name_vocative}`/`{caregiver_name}`/...
    placeholders from `profile` and `extra` (e.g. `time_words=...`).

    Every field `PersonProfile` defines always has a value -- its own
    dataclass defaults are the "safe default" issue #14 requires, so a
    template referencing `{name}` when no real profile is available
    still renders "there" or similar, never a literal `{name}` shown to
    the person. A placeholder this function does not recognise at all
    (a typo in a caregiver's edited template, or a field not listed here)
    degrades to an empty string via `_SafeFormatDict` rather than raising,
    for the same reason: a slightly odd sentence is safe, a crash mid-
    strategy is not. Collapses any resulting double space from a dropped
    placeholder, so "Hello , it's..." still reads as "Hello, it's...".

    `{name_vocative}` is the mechanism for a name dropped into the middle
    or end of a sentence rather than used as a plain word (contrast
    `soft_greeting`'s `{name}` usage, which reads fine either way): it
    renders as `", <name>"` when a name is actually configured, and as an
    empty string -- not `", there"` -- when it is still `PersonProfile`'s
    default (`_UNSET_NAME`). This is a property of rendering, not of any
    one template's word order, so every template that wants an optional
    "by name" aside uses this placeholder instead of writing `, {name}`
    itself; the caregiver never sees a mistake like "It's alright, there."
    """
    spoken_name = profile.preferred_address or profile.name
    name_vocative = f", {spoken_name}" if spoken_name != _UNSET_NAME else ""
    fields: dict[str, str] = {
        "name": spoken_name,
        "preferred_address": profile.preferred_address,
        "caregiver_name": profile.caregiver_name,
        "caregiver_relationship": profile.caregiver_relationship,
        "restroom_location": profile.restroom_location,
        "restroom_direction": (
            profile.restroom_location or "The restroom is just outside the bedroom"
        ).rstrip(".!?"),
        "name_vocative": name_vocative,
    }
    fields.update(extra)
    try:
        rendered = template.format_map(_SafeFormatDict(fields))
    except (ValueError, IndexError, KeyError) as exc:
        # `_SafeFormatDict` only rescues an *unknown* placeholder name.
        # A template that is malformed as a format string at all -- an
        # unclosed "{name", a positional "{0}" -- still raises out of
        # `format_map`, and `agent.main`'s loop has no handler, so an
        # uncaught one here would take the whole service down mid-session
        # at 3am. `load_strategies` rejects such a template at load time
        # (see `_check_template`) so this should be unreachable in
        # practice; it is the backstop for the code defaults themselves
        # and for any future caller that builds a `StrategyDef` directly.
        # An empty string is the safe degradation: a blank headline or
        # body shows nothing, and a blank `say` is rejected by
        # `agent.rules.validate_say` and becomes silence.
        logger.warning(
            json.dumps(
                {
                    "service": SERVICE_NAME,
                    "message": "malformed strategy template, rendering as empty",
                    "template": template,
                    "error": str(exc),
                }
            )
        )
        return ""
    return " ".join(rendered.split())


def time_as_words(now: datetime) -> str:
    """A simple, fixed phrase for the current hour, e.g. "3 o'clock at
    night" (PLAN.md section 8: "time shown as words and a clock"). Deliberately
    minimal -- a fixed template, not natural-language generation -- and
    always "at night", since this system only ever speaks inside a
    configured night window (`AgentConfig.in_night_window`)."""
    hour = now.hour % 12
    if hour == 0:
        hour = 12
    return f"{hour} o'clock at night"


def _log_fallback(reason: str, path: str) -> None:
    logger.warning(
        json.dumps(
            {
                "service": SERVICE_NAME,
                "message": "strategies config unavailable, using code defaults",
                "reason": reason,
                "path": path,
            }
        )
    )


def load_strategies(
    path: str | Path | None = None, *, env: dict[str, str] | None = None
) -> list[StrategyDef]:
    """Load the strategy catalogue, caregiver overrides layered on top of
    `DEFAULT_STRATEGIES`.

    Resolution order, matching `perceive.zones.load_zones` exactly:
    explicit `path` argument, then `STRATEGIES_PATH` from `env` (defaults
    to `os.environ`), then `config/strategies.yaml`. Falls back to
    `strategies.example.yaml` in the same directory if that is missing,
    and to `DEFAULT_STRATEGIES` unchanged if neither exists, the file
    fails to parse, or it parses to something unusable -- logged, never
    fatal, so a fresh checkout with no yaml still runs (HANDOFF.md
    section 4: env, then yaml, then code defaults).

    The yaml only ever *overrides* fields on a known strategy id
    (`order`, `enabled`, `cooldown_seconds`, `dwell_seconds`,
    `intrusiveness`, `headline`, `body`, `say`, `photo_id`, `brightness`,
    `face`, `clip_id`); an unknown id is logged and ignored rather than
    accepted as a new, code-unaware strategy, and a strategy the yaml does
    not mention at all keeps its code default untouched.
    """
    env = os.environ if env is None else env
    primary = (
        Path(path)
        if path is not None
        else Path(env.get("STRATEGIES_PATH", DEFAULT_STRATEGIES_DIR / DEFAULT_STRATEGIES_FILENAME))
    )
    candidate = (
        primary if primary.exists() else primary.parent / DEFAULT_STRATEGIES_EXAMPLE_FILENAME
    )

    if not candidate.exists():
        _log_fallback("no strategies.yaml or strategies.example.yaml found", str(candidate))
        return list(DEFAULT_STRATEGIES)

    try:
        with candidate.open() as handle:
            raw = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        _log_fallback(f"failed to parse {candidate}: {exc}", str(candidate))
        return list(DEFAULT_STRATEGIES)

    if not isinstance(raw, dict) or not isinstance(raw.get("strategies"), list):
        _log_fallback(f"{candidate} did not contain a 'strategies' list", str(candidate))
        return list(DEFAULT_STRATEGIES)

    by_id = {s.id: s for s in DEFAULT_STRATEGIES}
    merged: dict[str, StrategyDef] = dict(by_id)

    for entry in raw["strategies"]:
        try:
            merged_entry = _apply_override(by_id, entry)
        except (TypeError, ValueError, KeyError) as exc:
            _log_fallback(
                f"malformed strategy entry {entry!r} in {candidate}: {exc}", str(candidate)
            )
            continue
        if merged_entry is None:
            _log_fallback(
                f"unknown strategy id {entry.get('id')!r} in {candidate}, ignoring", str(candidate)
            )
            continue
        merged[merged_entry.id] = merged_entry

    return sorted(merged.values(), key=lambda s: s.order)


def _apply_override(by_id: Mapping[str, StrategyDef], entry: object) -> StrategyDef | None:
    """Merge one yaml strategy entry onto its code default, returning the
    updated `StrategyDef`, or `None` if `entry["id"]` names a strategy this
    module does not know about."""
    if not isinstance(entry, dict) or "id" not in entry:
        raise KeyError("id")
    base = by_id.get(entry["id"])
    if base is None:
        return None

    overrides: dict[str, object] = {}
    if "order" in entry:
        overrides["order"] = int(entry["order"])
    if "enabled" in entry:
        overrides["enabled"] = bool(entry["enabled"])
    if "cooldown_seconds" in entry:
        overrides["cooldown_seconds"] = float(entry["cooldown_seconds"])
    if "dwell_seconds" in entry:
        overrides["dwell_seconds"] = float(entry["dwell_seconds"])
    if "intrusiveness" in entry:
        overrides["intrusiveness"] = int(entry["intrusiveness"])
    if "headline" in entry:
        overrides["headline_template"] = str(entry["headline"])
    if "body" in entry:
        overrides["body_template"] = str(entry["body"])
    if "say" in entry:
        overrides["say_template"] = None if entry["say"] is None else str(entry["say"])
    if "photo_id" in entry:
        overrides["photo_id"] = None if entry["photo_id"] is None else str(entry["photo_id"])
    if "clip_id" in entry:
        clip_id = "" if entry["clip_id"] is None else str(entry["clip_id"]).strip()
        overrides["clip_id"] = clip_id or None
    if "brightness" in entry:
        overrides["brightness"] = float(entry["brightness"])
    if "face" in entry:
        overrides["face"] = str(entry["face"])

    merged = replace(base, **overrides)
    # Validate the caregiver's templates here, while there is still a code
    # default to fall back to. Raising sends the caller down its
    # "malformed strategy entry" path, which logs and keeps `base`
    # untouched -- the same shape as every other bad-config outcome in
    # this function. The alternative is discovering the typo when the
    # strategy is selected, mid-session, in the middle of the night.
    for label in ("headline_template", "body_template", "say_template"):
        _check_template(label, getattr(merged, label))
    return merged


def _check_template(label: str, template: str | None) -> None:
    """Raise `ValueError` if `template` is not a usable format string.

    Only catches templates that are malformed *as format strings*; an
    unknown placeholder name is fine and degrades to an empty string at
    render time (`render_template`), which is a caregiver's typo showing
    up as a slightly odd sentence rather than as a rejected config.
    """
    if template is None:
        return
    try:
        template.format_map(_SafeFormatDict({}))
    except (ValueError, IndexError, KeyError) as exc:
        raise ValueError(f"{label} {template!r} is not a valid template: {exc}") from exc


@dataclass
class StrategyEngine:
    """Ordering, cooldowns, and dwell for one session's strategy selection.

    Holds exactly the state config cannot: which strategy is current
    (`current_id`), when it started (for its dwell timer), and each
    strategy's cooldown clock (keyed by id). `agent.session.Session` owns
    one instance per session and resets it (`reset`) on every return to
    `IDLE`, so cooldowns do not carry across separate nights -- see
    `Session._reset_timers`.
    """

    strategies: list[StrategyDef]
    voice_clip_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("VOICE_CLIP_DIR") or DEFAULT_VOICE_CLIP_DIR)
    )

    _by_id: dict[str, StrategyDef] = field(init=False, repr=False)
    _cooldown_until: dict[str, datetime] = field(default_factory=dict, init=False, repr=False)
    current_id: str | None = field(default=None, init=False)
    _started_at: datetime | None = field(default=None, init=False, repr=False)
    # When the current strategy was first selected. Unlike `_started_at`,
    # `note_progress` never moves this, so it is the fixed reference the
    # `MAX_PROGRESS_DWELL_MULTIPLIER` cap is measured against.
    _selected_at: datetime | None = field(default=None, init=False, repr=False)
    _clip_unavailable_logged: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {s.id: s for s in self.strategies}

    def _ordered_ids(self) -> list[str]:
        return [s.id for s in sorted(self.strategies, key=lambda s: s.order)]

    def index_of(self, strategy_id: str | None) -> int:
        """Position of `strategy_id` in the full configured order
        (including disabled strategies, since a caregiver reordering the
        list in the dashboard should still see a stable position),
        published as `SessionState.strategy_index`. `0` for `None` (no
        strategy selected -- `IDLE`/`OBSERVING`) or an id this engine does
        not know about."""
        if strategy_id is None:
            return 0
        for i, sid in enumerate(self._ordered_ids()):
            if sid == strategy_id:
                return i
        return 0

    def current(self) -> StrategyDef | None:
        return self._by_id.get(self.current_id) if self.current_id is not None else None

    def _is_available(self, strategy_id: str, now: datetime) -> bool:
        strategy = self._by_id[strategy_id]
        on_cooldown = now < self._cooldown_until.get(strategy_id, now)
        disabled = (not strategy.enabled) or on_cooldown
        if not validate_strategy(strategy_id, disabled=disabled).accepted:
            return False
        if strategy.id == FAMILIAR_VOICE_ID:
            dwell_seconds = self._familiar_voice_dwell(strategy)
            if dwell_seconds is None:
                return False
            self._by_id[strategy_id] = replace(strategy, dwell_seconds=dwell_seconds)
        return True

    def _familiar_voice_dwell(self, strategy: StrategyDef) -> float | None:
        """Return clip duration plus its 15-second visual dwell, if usable."""
        clip_id = strategy.clip_id
        if not clip_id:
            self._log_clip_unavailable("no clip configured", clip_id)
            return None
        if (
            "/" in clip_id
            or "\\" in clip_id
            or ".." in clip_id
            or VOICE_CLIP_ID_RE.fullmatch(clip_id) is None
        ):
            self._log_clip_unavailable("unsafe clip id", clip_id)
            return None
        path = self.voice_clip_dir / f"{clip_id}.wav"
        if not path.is_file():
            self._log_clip_unavailable("clip missing", clip_id)
            return None
        try:
            with wave.open(str(path), "rb") as clip:
                duration = clip.getnframes() / clip.getframerate()
        except (EOFError, OSError, wave.Error, ZeroDivisionError) as exc:
            self._log_clip_unavailable("clip WAV header unreadable", clip_id, error=str(exc))
            return None
        return duration + 15.0

    def _log_clip_unavailable(
        self, reason: str, clip_id: str | None, **fields: object
    ) -> None:
        """Log each familiar-voice rejection reason once per engine."""
        if reason in self._clip_unavailable_logged:
            return
        self._clip_unavailable_logged.add(reason)
        logger.warning(
            json.dumps(
                {
                    "service": SERVICE_NAME,
                    "message": "familiar_voice unavailable; skipping strategy",
                    "reason": reason,
                    "clip_id": clip_id,
                    **fields,
                }
            )
        )

    def _first_available_id(self, now: datetime, *, after_id: str | None = None) -> str | None:
        ids = self._ordered_ids()
        start = 0
        if after_id is not None and after_id in ids:
            start = ids.index(after_id) + 1
        for sid in ids[start:]:
            # A terminal strategy is force-only and must never be reached
            # by simply running out of ordinary ones -- see
            # `StrategyDef.terminal`. Skipping it here is what makes
            # "strategies exhausted" reachable at all with the real
            # catalogue: `escalate_phone` is enabled with a zero cooldown,
            # so it would otherwise always be "available" and the
            # exhaustion branch of `maybe_advance` would be dead code.
            if self._by_id[sid].terminal or self._by_id[sid].goal_only:
                continue
            if self._is_available(sid, now):
                return sid
        return None

    def has_available(self, now: datetime) -> bool:
        """Whether any strategy could be selected right now -- used by
        `agent.session.Session` to decide, *before* committing to
        `ENGAGED`, whether entering it has anything to show at all or
        should escalate immediately instead (HANDOFF.md section 6:
        "strategies exhausted" as an escalation trigger)."""
        return self._first_available_id(now) is not None

    def start(self, now: datetime) -> StrategyDef | None:
        """Select the first available strategy in configured order.
        Returns `None` (and leaves `current_id` unset) if none is --
        callers should check `has_available` first if that would be a
        surprise; `agent.session.Session` always does."""
        sid = self._first_available_id(now)
        self.current_id = sid
        self._started_at = now if sid is not None else None
        self._selected_at = self._started_at
        return self.current()

    def force(self, strategy_id: str, now: datetime) -> StrategyDef | None:
        """Select `strategy_id` unconditionally, bypassing enabled/cooldown
        entirely. The only caller is `agent.session.Session` entering
        `ESCALATED`, forcing `escalate_phone`: PLAN.md's success condition
        for that phase is a caregiver responding, which this system
        cannot observe (the same reasoning `agent.goals`'s module
        docstring gives for `wait_for_caregiver` never being resolved by
        code -- an `Ack` on a phone is not a caregiver in the room), so
        there is nothing to "advance" to and no sense in which a cooldown
        or a disabled flag should be able to leave `ESCALATED` showing
        nothing at all. `escalate_phone` stays selected, by construction,
        until the session ends the ordinary way through `in_bed`
        stability into `COOLDOWN` (`Session._track_in_bed_stability`).
        """
        self.current_id = strategy_id
        self._started_at = now
        self._selected_at = now
        return self._by_id.get(strategy_id)

    def select_for_goal(self, strategy_id: str, now: datetime) -> StrategyDef | None:
        """Select a deterministic goal-specific strategy when available.

        Unlike `force`, this obeys caregiver enablement and cooldown. Unlike
        `propose_next`, it may jump outside the ordinary ladder because the
        active goal, not an LLM proposal, determines the strategy.
        """
        strategy = self._by_id.get(strategy_id)
        if strategy is None or strategy.terminal or not self._is_available(strategy_id, now):
            return None
        current = self.current()
        if current is not None and current.id != strategy_id:
            self._cooldown_until[current.id] = now + timedelta(seconds=current.cooldown_seconds)
        self.current_id = strategy_id
        self._started_at = now
        self._selected_at = now
        return strategy

    def propose_next(self, strategy_id: str, now: datetime) -> StrategyDef | None:
        """Accept one planner suggestion only when it is the deterministic
        next available ordinary strategy.

        The planner may shorten a dwell, but it may not skip the configured
        order, reselect a cooled-down/disabled strategy, or select a terminal
        escalation action.  ``_first_available_id`` is the same route normal
        advancement uses, including ``validate_strategy`` for each candidate.
        ``None`` is an ordinary rejected advisory proposal.
        """
        current = self.current()
        if current is None or current.terminal:
            return None
        next_id = self._first_available_id(now, after_id=current.id)
        if next_id != strategy_id:
            return None

        self._cooldown_until[current.id] = now + timedelta(seconds=current.cooldown_seconds)
        self.current_id = next_id
        self._started_at = now
        self._selected_at = now
        return self.current()

    def note_progress(self, now: datetime) -> None:
        """Extend the current strategy's dwell window: called whenever
        `agent.session.Session` sees "progress" (heading to or reaching
        the bed) while a strategy is showing, so a person who is
        responding to it is not interrupted by an unrelated timeout.

        Bounded by `MAX_PROGRESS_DWELL_MULTIPLIER` (see there): once this
        strategy has been current for that multiple of its own dwell, the
        extension stops and the next `maybe_advance` moves the ladder on,
        however much progress is still being reported. A caller may keep
        calling this every reading; past the cap it simply does nothing.
        """
        current = self.current()
        if current is None or self._selected_at is None:
            return
        held_for = (now - self._selected_at).total_seconds()
        if held_for >= current.dwell_seconds * MAX_PROGRESS_DWELL_MULTIPLIER:
            return
        self._started_at = now

    def maybe_advance(self, now: datetime) -> tuple[StrategyDef | None, bool, bool]:
        """If the current strategy's dwell has elapsed, put it on cooldown
        and move to the next available one in order.

        Returns `(new_current, changed, exhausted)`: `changed` is `True`
        only if an advance actually happened (dwell not yet elapsed, or no
        strategy selected at all, returns the unchanged current strategy
        and `False`). `exhausted` is `True` when the dwell elapsed but no
        further strategy is available -- every remaining one disabled or
        on cooldown -- the "strategies exhausted" trigger; `new_current`
        is `None` in that case and `current_id` is cleared.
        """
        current = self.current()
        if current is None or self._started_at is None:
            return current, False, False
        # A goal-specific strategy is bounded by the goal lifecycle, not by
        # the ordinary ladder. In particular, `path_light` must not exhaust
        # and escalate on the same tick that the restroom timeout safely
        # returns the session to `return_to_bed`.
        if current.goal_only:
            return current, False, False

        elapsed = (now - self._started_at).total_seconds()
        if elapsed < current.dwell_seconds:
            return current, False, False

        self._cooldown_until[current.id] = now + timedelta(seconds=current.cooldown_seconds)
        next_id = self._first_available_id(now, after_id=current.id)
        self.current_id = next_id
        self._started_at = now if next_id is not None else None
        self._selected_at = self._started_at
        if next_id is None:
            return None, True, True
        return self.current(), True, False

    def reset(self) -> None:
        """Clear all per-session state: current strategy, its start time,
        and every cooldown clock. Called on every return to `IDLE`
        (`Session._reset_timers`) so a new night starts with a clean
        ladder."""
        self.current_id = None
        self._started_at = None
        self._selected_at = None
        self._cooldown_until.clear()
        self._clip_unavailable_logged.clear()
