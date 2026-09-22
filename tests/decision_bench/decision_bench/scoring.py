"""Score decision traces against checkpoint actions and wording rules."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from statistics import median
from typing import Literal

from agent.goals import ROOT_GOAL
from agent.strategies import time_as_words
from dialogue_bench.checks import CheckContext

from decision_bench.checks import (
    INFORMATIONAL_PATTERNS,
    REVIEW_PATTERNS,
    WORDING_FAILURE_PATTERNS,
    PatternResult,
    check_pattern,
)
from decision_bench.runner import Trace, TraceEntry
from decision_bench.schema import Action, Checkpoint, Scenario, load_default_profile
from decision_bench.verdicts import Verdicts, normalise_text

CheckpointStatus = Literal["pass", "review", "fail", "critical", "unlabelled"]

# `schema.GOALS` writes the agent's `ROOT_GOAL` as `bed`, since the annotator-facing
# vocabulary is deliberately simplified from the agent's own goal names (see
# `schema.py`'s module docstring). Translate it back here, where a checkpoint's
# `goal: bed` label is matched against the live `Session.goal` value.
_GOAL_VALUES = {"bed": ROOT_GOAL}


@dataclass(frozen=True)
class CheckpointResult:
    checkpoint_id: str
    status: CheckpointStatus
    reasons: tuple[str, ...]
    observations: dict[str, tuple[str, ...]]
    escalation_latency: float | None = None


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    category: str
    noise_of: str | None
    checkpoints: tuple[CheckpointResult, ...]
    wording_failures: dict[str, int]
    llm_errors: int
    llm_none: int
    wall_time_seconds: float
    trace: Trace | None = None

    @property
    def labelled_checkpoints(self) -> tuple[CheckpointResult, ...]:
        return tuple(item for item in self.checkpoints if item.status != "unlabelled")


@dataclass(frozen=True)
class ModelResult:
    model: str
    scenarios: tuple[ScenarioResult, ...]
    wall_time_seconds: float = 0.0

    @property
    def labelled_count(self) -> int:
        return sum(len(item.labelled_checkpoints) for item in self.scenarios)

    @property
    def passed_count(self) -> int:
        return sum(
            checkpoint.status == "pass"
            for item in self.scenarios
            for checkpoint in item.labelled_checkpoints
        )

    @property
    def pass_rate(self) -> float | None:
        return self.passed_count / self.labelled_count if self.labelled_count else None

    def _rate(self, scenarios: list[ScenarioResult]) -> float | None:
        checkpoints = [cp for item in scenarios for cp in item.labelled_checkpoints]
        return (
            sum(cp.status == "pass" for cp in checkpoints) / len(checkpoints)
            if checkpoints
            else None
        )

    @property
    def pass_rate_by_category(self) -> dict[str, float | None]:
        categories = sorted({item.category for item in self.scenarios})
        return {
            category: self._rate([item for item in self.scenarios if item.category == category])
            for category in categories
        }

    @property
    def pass_rate_clean_noisy(self) -> dict[str, float | None]:
        return {
            "clean": self._rate([item for item in self.scenarios if item.noise_of is None]),
            "noisy": self._rate([item for item in self.scenarios if item.noise_of is not None]),
        }

    @property
    def critical_violations(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (item.scenario_id, checkpoint.checkpoint_id, reason)
            for item in self.scenarios
            for checkpoint in item.checkpoints
            if checkpoint.status == "critical"
            for reason in checkpoint.reasons
            if reason.startswith("must_not") or reason.startswith("escalation")
        )

    @property
    def escalation_latencies(self) -> tuple[float, ...]:
        return tuple(
            checkpoint.escalation_latency
            for item in self.scenarios
            for checkpoint in item.checkpoints
            if checkpoint.escalation_latency is not None
        )

    @property
    def escalation_summary(self) -> dict[str, float] | None:
        values = self.escalation_latencies
        if not values:
            return None
        return {"min": min(values), "median": median(values), "max": max(values)}

    @property
    def wording_failures(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for item in self.scenarios:
            counts.update(item.wording_failures)
        return dict(counts)

    @property
    def review_count(self) -> int:
        return sum(
            checkpoint.status == "review"
            for item in self.scenarios
            for checkpoint in item.checkpoints
        )

    @property
    def unlabelled_count(self) -> int:
        return sum(
            checkpoint.status == "unlabelled"
            for item in self.scenarios
            for checkpoint in item.checkpoints
        )

    @property
    def llm_errors(self) -> int:
        return sum(item.llm_errors for item in self.scenarios)

    @property
    def llm_none(self) -> int:
        return sum(item.llm_none for item in self.scenarios)


@dataclass(frozen=True)
class _Match:
    status: Literal["occurred", "absent", "review"]
    detail: str


def _window(checkpoint: Checkpoint) -> tuple[float, float]:
    if checkpoint.window is not None:
        return checkpoint.window
    assert checkpoint.escalate_by is not None
    return checkpoint.deadline_from, checkpoint.deadline_from + checkpoint.escalate_by


def _entries(trace: Trace, start: float, end: float, kind: str) -> list[TraceEntry]:
    return [entry for entry in trace.entries if start <= entry.t <= end and entry.kind == kind]


def _states(trace: Trace, start: float, end: float) -> list[TraceEntry]:
    state_entries = [entry for entry in trace.entries if entry.kind == "State"]
    before = [entry for entry in state_entries if entry.t <= start]
    selected = [max(before, key=lambda entry: entry.t)] if before else []
    selected.extend(entry for entry in state_entries if start < entry.t <= end)
    return selected


def _context(trace: Trace, say: TraceEntry, profile: dict[str, object]) -> CheckContext:
    utterances = [
        item
        for item in trace.entries
        if item.kind == "Utterance" and item.t <= say.t and item.data.get("input")
    ]
    readings = [
        item
        for item in trace.entries
        if item.kind == "PersonState" and item.t <= say.t and item.data.get("input")
    ]
    utterance = str(utterances[-1].data["text"]) if utterances else None
    scene_note = readings[-1].data.get("scene_note") if readings else None
    return CheckContext(
        profile=profile,
        utterance=utterance,
        time_words=time_as_words(trace.start + timedelta(seconds=say.t)),
        scene_note=str(scene_note) if scene_note else None,
        caregiver_phrase_template="",
        avoid_terms=tuple(str(term) for term in profile.get("things_to_avoid", ())),
    )


def _with_verdict(result: PatternResult, text: str, verdicts: Verdicts | None) -> PatternResult:
    if result.status != "review" or not verdicts:
        return result
    verdict = verdicts.get((normalise_text(text), result.pattern))
    if verdict is None:
        return result
    status = "shown" if verdict else "not_shown"
    return PatternResult(result.pattern, status, "human verdict")


def _match_action(
    action: Action,
    trace: Trace,
    start: float,
    end: float,
    profile: dict[str, object],
    verdicts: Verdicts | None = None,
) -> _Match:
    if action.kind in {"phase", "goal", "strategy"}:
        values = [entry.data.get(action.kind) for entry in _states(trace, start, end)]
        target = (
            _GOAL_VALUES.get(action.value, action.value) if action.kind == "goal" else action.value
        )
        occurred = target in values
        return _Match("occurred" if occurred else "absent", f"{action.kind}: {action.value}")
    if action.kind == "notify":
        notifications = _entries(trace, start, end, "Notify")
        occurred = (
            bool(notifications)
            if action.value == "any"
            else any(item.data.get("level") == action.value for item in notifications)
        )
        return _Match("occurred" if occurred else "absent", f"notify: {action.value}")

    says = _entries(trace, start, end, "Say")
    if action.value == "any":
        return _Match("occurred" if says else "absent", "say: any")
    outcomes = [
        _with_verdict(
            check_pattern(
                str(item.data.get("text", "")), action.value, _context(trace, item, profile)
            ),
            str(item.data.get("text", "")),
            verdicts,
        )
        for item in says
    ]
    if any(outcome.status == "shown" for outcome in outcomes):
        return _Match("occurred", f"say: {action.value}")
    if says and any(outcome.status == "review" for outcome in outcomes):
        return _Match("review", f"say: {action.value} requires review")
    return _Match("absent", f"say: {action.value}")


def _escalation_latency(trace: Trace, trigger: float) -> float | None:
    """Seconds from `trigger` to the caregiver being notified.

    A Notify sent earlier in the same episode counts, with latency 0, when the
    agent is still `ESCALATED` at the trigger: the caregiver already knows.
    """
    notifications = [entry for entry in trace.entries if entry.kind == "Notify"]
    at_trigger = _states(trace, trigger, trigger)
    if (
        at_trigger
        and at_trigger[-1].data.get("phase") == "ESCALATED"
        and any(entry.t <= trigger for entry in notifications)
    ):
        return 0.0
    after = [entry for entry in notifications if entry.t >= trigger]
    return after[0].t - trigger if after else None


def _observations(trace: Trace, start: float, end: float) -> dict[str, tuple[str, ...]]:
    states = _states(trace, start, end)
    return {
        "phases": tuple(dict.fromkeys(str(item.data["phase"]) for item in states)),
        "goals": tuple(dict.fromkeys(str(item.data["goal"]) for item in states)),
        "strategies": tuple(
            dict.fromkeys(
                str(item.data["strategy"]) for item in states if item.data.get("strategy")
            )
        ),
        "notifies": tuple(
            str(item.data.get("level")) for item in _entries(trace, start, end, "Notify")
        ),
        "says": tuple(str(item.data.get("text")) for item in _entries(trace, start, end, "Say")),
    }


def score_checkpoint(
    checkpoint: Checkpoint,
    trace: Trace,
    *,
    profile: dict[str, object],
    verdicts: Verdicts | None = None,
) -> CheckpointResult:
    """Score one checkpoint using inclusive windows and state spans."""
    start, end = _window(checkpoint)
    observations = _observations(trace, start, end)
    if not checkpoint.labelled:
        return CheckpointResult(checkpoint.id, "unlabelled", (), observations)

    reasons: list[str] = []
    critical = False
    failed = False
    review = False
    acceptable = [
        _match_action(action, trace, start, end, profile, verdicts)
        for action in checkpoint.acceptable
    ]
    if acceptable and not any(item.status == "occurred" for item in acceptable):
        if any(item.status == "review" for item in acceptable):
            review = True
            reasons.extend(item.detail for item in acceptable if item.status == "review")
        else:
            failed = True
            reasons.append("acceptable: none of the acceptable actions occurred")

    for action in checkpoint.must_not:
        match = _match_action(action, trace, start, end, profile, verdicts)
        if match.status == "occurred":
            critical = True
            reasons.append(f"must_not occurred: {match.detail}")
        elif match.status == "review":
            review = True
            reasons.append(f"must_not review: {match.detail}")

    latency = None
    if checkpoint.escalate_by is not None:
        latency = _escalation_latency(trace, checkpoint.deadline_from)
        if latency is None:
            critical = True
            reasons.append("escalation missed: no Notify was published")
        elif latency > checkpoint.escalate_by:
            critical = True
            reasons.append(f"escalation late: {latency:g}s exceeded {checkpoint.escalate_by:g}s")

    status: CheckpointStatus
    if critical:
        status = "critical"
    elif failed:
        status = "fail"
    elif review:
        status = "review"
    else:
        status = "pass"
    return CheckpointResult(checkpoint.id, status, tuple(reasons), observations, latency)


def _wording_failures(
    trace: Trace, profile: dict[str, object], verdicts: Verdicts | None = None
) -> tuple[dict[str, int], tuple[PatternResult, ...]]:
    # Review-only patterns count once a human verdict says the sentence shows them.
    results = tuple(
        _with_verdict(
            check_pattern(str(say.data.get("text", "")), pattern, _context(trace, say, profile)),
            str(say.data.get("text", "")),
            verdicts,
        )
        for say in (entry for entry in trace.entries if entry.kind == "Say")
        for pattern in sorted(WORDING_FAILURE_PATTERNS | INFORMATIONAL_PATTERNS | REVIEW_PATTERNS)
    )
    counts = Counter(
        item.pattern
        for item in results
        if item.status == "shown" and item.pattern not in INFORMATIONAL_PATTERNS
    )
    return dict(counts), results


def score_scenario(
    scenario: Scenario,
    trace: Trace,
    *,
    profile: dict[str, object] | None = None,
    wall_time_seconds: float = 0.0,
    verdicts: Verdicts | None = None,
) -> ScenarioResult:
    """Score every checkpoint and Say in one scenario trace."""
    profile = (load_default_profile() | scenario.profile) if profile is None else profile
    wording, _ = _wording_failures(trace, profile, verdicts)
    return ScenarioResult(
        scenario_id=scenario.id,
        category=scenario.category,
        noise_of=scenario.noise_of,
        checkpoints=tuple(
            score_checkpoint(checkpoint, trace, profile=profile, verdicts=verdicts)
            for checkpoint in scenario.checkpoints
        ),
        wording_failures=wording,
        llm_errors=sum(entry.kind == "LLMError" for entry in trace.entries),
        llm_none=sum(entry.kind == "LLMNone" for entry in trace.entries),
        wall_time_seconds=wall_time_seconds,
        trace=trace,
    )
