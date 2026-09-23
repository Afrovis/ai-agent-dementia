"""Scenario schema for decision_bench.

A scenario is one scripted night: a timeline of `PersonState` readings and
`Utterance`s, and the checkpoints at which the agent's actions are judged.
Each scenario lives in its own YAML file under `fixtures/scenarios/`, named
after its id.

A checkpoint starts out unlabelled: it carries a `question` for the
annotator and a `window`, and nothing else. Labels (`acceptable`,
`must_not`, `escalate_by`) are added by the annotation workflow in PLAN.md,
never by hand-copying what the agent currently does.

The action vocabulary here is deliberately written out rather than imported
from the agent, so the annotator can be shown this list without the agent's
code or strategy config. `tests/test_schema.py` checks that it still matches
the agent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PACKAGE_ROOT / "fixtures"
SCENARIOS_DIR = FIXTURES_DIR / "scenarios"
DEFAULT_PROFILE_PATH = FIXTURES_DIR / "profile.yaml"
GUIDELINES_PATH = PACKAGE_ROOT / "guidelines.md"

Category = Literal[
    "conversation",
    "restroom",
    "disorientation",
    "distress_pain",
    "fall",
    "false_alarm",
    "silent_wander",
]
CATEGORIES: tuple[str, ...] = Category.__args__

PersonStateName = Literal["in_bed", "sitting_up", "standing", "walking", "on_floor", "absent"]
Zone = Literal["bed", "door", "bathroom_path", "other"]

PHASES = frozenset({"IDLE", "OBSERVING", "ENGAGED", "COOLDOWN", "ESCALATED"})
GOALS = frozenset({"bed", "restroom", "comfort"})
STRATEGIES = frozenset(
    {
        "ambient_orient",
        "soft_greeting",
        "orient_time_place",
        "validate_and_redirect",
        "guided_return",
        "familiar_voice",
        "path_light",
        "reassure_waiting",
        "acknowledge_return",
        "acknowledge_pain",
        "comfort_pain",
        "acknowledge_progress",
        "acknowledge_feeling",
        "ask_need",
        "caregiver_alerted",
        "escalate_phone",
    }
)
NOTIFY_LEVELS = frozenset({"any", "info", "attention", "critical"})

# The first five are the `dialogue_bench` checks; the rest are named
# wording patterns defined in guidelines.md. `any` means any `Say` at all.
SAY_PATTERNS = frozenset(
    {
        "any",
        "conjunction_but",
        "avoid_terms",
        "states_clock_time",
        "invents_proper_noun",
        "addresses_by_name",
        "correction_of_reality",
        "memory_question",
        "blunt_refusal",
        "infantilising",
        "invents_directions",
        "unsupported_claim",
    }
)

ACTION_VALUES: dict[str, frozenset[str]] = {
    "phase": PHASES,
    "goal": GOALS,
    "strategy": STRATEGIES,
    "notify": NOTIFY_LEVELS,
    "say": SAY_PATTERNS,
}

# Mirrors `agent.profile.PersonProfile`; the test suite keeps them in step.
PROFILE_FIELDS = frozenset(
    {
        "name",
        "preferred_address",
        "caregiver_name",
        "caregiver_relationship",
        "night_themes",
        "calming_things",
        "things_to_avoid",
        "physical_notes",
        "restroom_location",
    }
)

CLAUSE_PREFIXES = ("NICE", "AA", "VAL", "PCC", "DICE", "FALL", "TOIL")
CLAUSE_ID_RE = re.compile(rf"^(?:{'|'.join(CLAUSE_PREFIXES)})-\d{{2}}$")
_CLAUSE_HEADING_RE = re.compile(rf"^### ((?:{'|'.join(CLAUSE_PREFIXES)})-\d{{2}}) ·", re.M)
_CLAUSE_CHECKED_RE = re.compile(r"^- \*\*Checked:\*\* \[([ xX])\]", re.M)
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PersonReading(_Strict):
    """One `PersonState` the harness feeds the agent."""

    state: PersonStateName
    zone: Zone
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    scene_note: str | None = None


class UtteranceInput(_Strict):
    """One `Utterance` the harness feeds the agent."""

    text: str = Field(min_length=1)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    duration_s: float = Field(default=2.0, gt=0.0)


class TimelineEvent(_Strict):
    """One input at `t` seconds after the scenario's start."""

    t: float = Field(ge=0.0)
    person: PersonReading | None = None
    utterance: UtteranceInput | None = None

    @model_validator(mode="after")
    def _exactly_one_input(self) -> TimelineEvent:
        if (self.person is None) == (self.utterance is None):
            raise ValueError("a timeline event needs exactly one of 'person' or 'utterance'")
        return self


class Action(_Strict):
    """One thing the agent can do, written as a one-key mapping such as
    `{strategy: path_light}` or `{notify: critical}`."""

    phase: str | None = None
    goal: str | None = None
    strategy: str | None = None
    notify: str | None = None
    say: str | None = None

    @model_validator(mode="after")
    def _exactly_one_known_value(self) -> Action:
        set_fields = [(k, v) for k, v in self.model_dump().items() if v is not None]
        if len(set_fields) != 1:
            raise ValueError(f"an action names exactly one of {sorted(ACTION_VALUES)}")
        kind, value = set_fields[0]
        if value not in ACTION_VALUES[kind]:
            allowed = ", ".join(sorted(ACTION_VALUES[kind]))
            raise ValueError(f"unknown {kind} {value!r}; expected one of: {allowed}")
        return self

    @property
    def kind(self) -> str:
        return next(k for k, v in self.model_dump().items() if v is not None)

    @property
    def value(self) -> str:
        return getattr(self, self.kind)


class Checkpoint(_Strict):
    """A moment at which the agent's actions are judged.

    `window` is `[start, end]` in scenario seconds. An `escalate_by`
    deadline counts from `trigger` (default: the window start, else 0) and
    must say where its number comes from, which is always the caregiver.

    `doubtful_acceptable` and `doubtful_must_not` hold actions that only one
    of three independent annotator runs set, with no majority either way
    (see `vote.py`). They are never a critical violation and never required
    for a pass; `scoring.py` gives them half credit (`doubt`).
    """

    id: str
    question: str | None = None
    window: tuple[float, float] | None = None
    acceptable: tuple[Action, ...] = ()
    must_not: tuple[Action, ...] = ()
    doubtful_acceptable: tuple[Action, ...] = ()
    doubtful_must_not: tuple[Action, ...] = ()
    escalate_by: float | None = Field(default=None, gt=0.0)
    trigger: float | None = Field(default=None, ge=0.0)
    threshold_source: Literal["caregiver"] | None = None
    rationale: str | None = None
    cites: tuple[str, ...] = ()

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value):
            raise ValueError(f"checkpoint id {value!r} must be lower-case words joined by - or _")
        return value

    @field_validator("cites")
    @classmethod
    def _clause_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for cite in value:
            if not CLAUSE_ID_RE.match(cite):
                raise ValueError(f"{cite!r} is not a guideline clause id such as VAL-01")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Checkpoint:
        if self.window is not None and self.window[0] > self.window[1]:
            raise ValueError(f"checkpoint {self.id!r} window starts after it ends")
        if (
            self.acceptable or self.must_not or self.doubtful_acceptable or self.doubtful_must_not
        ) and self.window is None:
            raise ValueError(f"checkpoint {self.id!r} has action labels but no window")
        if self.window is None and self.escalate_by is None:
            raise ValueError(f"checkpoint {self.id!r} needs a window or an escalate_by deadline")
        if self.escalate_by is not None and self.threshold_source != "caregiver":
            raise ValueError(
                f"checkpoint {self.id!r}: escalate_by is a time threshold and must set "
                "threshold_source: caregiver"
            )
        if self.escalate_by is None and (
            self.threshold_source is not None or self.trigger is not None
        ):
            raise ValueError(
                f"checkpoint {self.id!r}: threshold_source and trigger only go with escalate_by"
            )
        if self.labelled:
            if not self.rationale:
                raise ValueError(f"labelled checkpoint {self.id!r} needs a rationale")
            if (
                self.acceptable
                or self.must_not
                or self.doubtful_acceptable
                or self.doubtful_must_not
            ) and not self.cites:
                raise ValueError(f"labelled checkpoint {self.id!r} must cite guideline clauses")
        elif not self.question:
            raise ValueError(f"unlabelled checkpoint {self.id!r} needs a question")
        return self

    @property
    def labelled(self) -> bool:
        return bool(
            self.acceptable
            or self.must_not
            or self.doubtful_acceptable
            or self.doubtful_must_not
            or self.escalate_by is not None
        )

    @property
    def deadline_from(self) -> float:
        """Scenario second that an `escalate_by` deadline counts from."""
        if self.trigger is not None:
            return self.trigger
        return self.window[0] if self.window is not None else 0.0

    def actions(self) -> tuple[Action, ...]:
        return self.acceptable + self.must_not


class Scenario(_Strict):
    """One scripted night. See README.md, "Scenario format"."""

    id: str
    category: Category
    summary: str = Field(min_length=1)
    start: time
    profile: dict[str, object] = Field(default_factory=dict)
    voice_clip: bool = False
    noise_of: str | None = None
    timeline: tuple[TimelineEvent, ...] = Field(min_length=1)
    checkpoints: tuple[Checkpoint, ...] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value):
            raise ValueError(f"scenario id {value!r} must be lower-case words joined by - or _")
        return value

    @field_validator("start", mode="before")
    @classmethod
    def _parse_start(cls, value: object) -> object:
        # YAML 1.1 reads an unquoted 02:14 as the integer 134 (sexagesimal).
        if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}", value):
            raise ValueError('start must be a quoted local time such as "02:14"')
        return time.fromisoformat(value)

    @field_validator("profile")
    @classmethod
    def _known_profile_fields(cls, value: dict[str, object]) -> dict[str, object]:
        unknown = sorted(set(value) - PROFILE_FIELDS)
        if unknown:
            raise ValueError(f"unknown profile fields: {unknown}")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Scenario:
        times = [event.t for event in self.timeline]
        if times != sorted(times):
            raise ValueError(f"scenario {self.id!r} timeline is not in time order")
        ids = [checkpoint.id for checkpoint in self.checkpoints]
        if len(ids) != len(set(ids)):
            raise ValueError(f"scenario {self.id!r} has duplicate checkpoint ids")
        if self.noise_of == self.id:
            raise ValueError(f"scenario {self.id!r} cannot be a noisy variant of itself")
        if not self.voice_clip:
            for checkpoint in self.checkpoints:
                accepted = {(a.kind, a.value) for a in checkpoint.acceptable} | {
                    (a.kind, a.value) for a in checkpoint.doubtful_acceptable
                }
                if ("strategy", "familiar_voice") in accepted:
                    raise ValueError(
                        f"scenario {self.id!r} checkpoint {checkpoint.id!r} accepts "
                        "familiar_voice, but the scenario sets no voice_clip"
                    )
        return self

    @property
    def labelled(self) -> bool:
        return all(checkpoint.labelled for checkpoint in self.checkpoints)

    @property
    def cites(self) -> frozenset[str]:
        return frozenset(c for checkpoint in self.checkpoints for c in checkpoint.cites)


def load_scenario(path: Path) -> Scenario:
    """Load one scenario file. Its id must match the file name."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    scenario = Scenario.model_validate(raw)
    if scenario.id != path.stem:
        raise ValueError(f"{path}: id {scenario.id!r} does not match the file name")
    return scenario


def load_scenarios(directory: Path = SCENARIOS_DIR) -> list[Scenario]:
    """Load every scenario in `directory`, sorted by id, and check that
    ids are unique and every `noise_of` names a clean scenario in the set."""
    scenarios = sorted(
        (load_scenario(path) for path in directory.glob("*.yaml")), key=lambda s: s.id
    )
    by_id: dict[str, Scenario] = {}
    for scenario in scenarios:
        if scenario.id in by_id:
            raise ValueError(f"duplicate scenario id {scenario.id!r}")
        by_id[scenario.id] = scenario
    for scenario in scenarios:
        if scenario.noise_of is None:
            continue
        parent = by_id.get(scenario.noise_of)
        if parent is None:
            raise ValueError(f"{scenario.id!r} is noise_of unknown scenario {scenario.noise_of!r}")
        if parent.noise_of is not None:
            raise ValueError(f"{scenario.id!r} is noise_of another noisy variant")
    return scenarios


def load_default_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, object]:
    """The profile every scenario starts from before its own overrides."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    unknown = sorted(set(raw) - PROFILE_FIELDS)
    if unknown:
        raise ValueError(f"{path}: unknown profile fields: {unknown}")
    return raw


def guideline_clauses(path: Path = GUIDELINES_PATH) -> dict[str, bool]:
    """Map each clause id in the guideline pack to whether a human has
    ticked its `Checked` box."""
    text = path.read_text(encoding="utf-8")
    headings = list(_CLAUSE_HEADING_RE.finditer(text))
    clauses: dict[str, bool] = {}
    for index, heading in enumerate(headings):
        clause_id = heading.group(1)
        if clause_id in clauses:
            raise ValueError(f"{path}: clause {clause_id} appears twice")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        checked = _CLAUSE_CHECKED_RE.search(text, heading.end(), end)
        if checked is None:
            raise ValueError(f"{path}: clause {clause_id} has no Checked line")
        clauses[clause_id] = checked.group(1) != " "
    return clauses


def citation_problems(scenarios: Iterable[Scenario], clauses: dict[str, bool]) -> list[str]:
    """Cites that name no clause, or a clause no human has checked yet."""
    problems = []
    for scenario in scenarios:
        for checkpoint in scenario.checkpoints:
            for cite in checkpoint.cites:
                where = f"{scenario.id}/{checkpoint.id}"
                if cite not in clauses:
                    problems.append(f"{where} cites unknown clause {cite}")
                elif not clauses[cite]:
                    problems.append(f"{where} cites {cite}, which is not checked yet")
    return problems
