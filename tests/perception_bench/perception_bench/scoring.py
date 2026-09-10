"""Scoring: confusion matrix, per-state precision/recall, and latency stats.

Kept separate from every tier's fixture-generation/loading code, and from
`report.py`'s presentation, the same way `perceive.classify` keeps pure
geometry separate from `StateTracker`'s statefulness: these functions take
plain lists of `(actual, predicted)` pairs or latency floats and return
plain dataclasses, so they are trivially unit-testable with hand-built
inputs and reusable across all three tiers.

A note on what "per-state accuracy" means here, since PLAN.md section 12
and issue #11 say "accuracy" without defining it precisely: this module
reports **recall** per state (of the frames whose true state is X, what
fraction did the pipeline call X) as the number checked against the >95%
targets for `standing` and `on_floor`. Recall is the right reading of "the
system correctly recognises X when X is happening" -- the safety-relevant
question -- as opposed to precision ("when the system says X, how often is
it right"), which is reported alongside it for the confusion matrix to be
useful, but is not itself gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Local import, not `perceive.classify.PersonStateName`, to keep this module
# importable with zero knowledge of where `perceive` lives on disk -- only
# `perception_bench.synthetic`/`daylight`/`infrared`, which actually build
# `PoseResult`s and drive the tracker, need `perceive` on `sys.path`.
STATE_NAMES: tuple[str, ...] = (
    "in_bed",
    "sitting_up",
    "standing",
    "walking",
    "on_floor",
    "absent",
)

TARGET_STATES: tuple[str, ...] = ("standing", "on_floor")
"""The two states PLAN.md section 12 sets an explicit >95% recall target for."""

ACCURACY_TARGET = 0.95
LATENCY_TARGET_S = 2.0


@dataclass(frozen=True)
class PerStateStats:
    """Precision/recall/support for one state, plus raw counts for review."""

    state: str
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int
    """Number of frames whose ground truth is `state`. Zero means this
    state never occurred in the fixtures scored, so precision/recall are
    both `None` rather than a misleading 0.0 or 1.0."""

    @property
    def recall(self) -> float | None:
        if self.support == 0:
            return None
        return self.true_positives / self.support

    @property
    def precision(self) -> float | None:
        predicted = self.true_positives + self.false_positives
        if predicted == 0:
            return None
        return self.true_positives / predicted


@dataclass(frozen=True)
class AccuracyResult:
    """Confusion matrix and per-state stats over one set of scored frames."""

    confusion: dict[str, dict[str, int]]
    """`confusion[actual][predicted] -> count`. Every `STATE_NAMES` value is
    a key on both axes, even at zero, so the printed matrix is always
    square and comparable across runs."""
    per_state: dict[str, PerStateStats]
    frame_count: int

    @property
    def overall_accuracy(self) -> float | None:
        if self.frame_count == 0:
            return None
        correct = sum(self.confusion[s].get(s, 0) for s in STATE_NAMES)
        return correct / self.frame_count


def score_predictions(pairs: list[tuple[str, str]]) -> AccuracyResult:
    """Build an `AccuracyResult` from `(actual, predicted)` state pairs.

    Both `actual` and `predicted` are expected to be members of
    `STATE_NAMES`; an unrecognised value is still tallied (it will show up
    as an extra row/column beyond the six) rather than raising, so a bug
    upstream surfaces as a visibly wrong report instead of a crash.
    """
    confusion: dict[str, dict[str, int]] = {s: dict.fromkeys(STATE_NAMES, 0) for s in STATE_NAMES}
    for actual, predicted in pairs:
        confusion.setdefault(actual, dict.fromkeys(STATE_NAMES, 0))
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1

    per_state: dict[str, PerStateStats] = {}
    for state in STATE_NAMES:
        tp = confusion.get(state, {}).get(state, 0)
        support = sum(confusion.get(state, {}).values())
        fn = support - tp
        fp = sum(confusion.get(a, {}).get(state, 0) for a in confusion if a != state)
        per_state[state] = PerStateStats(
            state=state,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            support=support,
        )

    return AccuracyResult(confusion=confusion, per_state=per_state, frame_count=len(pairs))


@dataclass(frozen=True)
class LatencyResult:
    """Frames-to-detection latency for a set of scripted transitions.

    `samples` is `(transition_label, latency_seconds)` for every measured
    transition -- e.g. `("in_bed -> on_floor", 0.5)`. Only tier 1 (and a
    tier 3 with per-frame timestamps) can produce these: tier 2's
    clip-level labels carry no transition timestamp, so nothing in
    `perception_bench.daylight` calls this.
    """

    samples: list[tuple[str, float]] = field(default_factory=list)

    @property
    def max_latency_s(self) -> float | None:
        return max((s[1] for s in self.samples), default=None)

    @property
    def mean_latency_s(self) -> float | None:
        if not self.samples:
            return None
        return sum(s[1] for s in self.samples) / len(self.samples)

    def within_target(self, target_s: float = LATENCY_TARGET_S) -> bool | None:
        """`True`/`False` if any transitions were measured, `None` if none
        were -- the "not measured" case a caller must not report as a pass."""
        if not self.samples:
            return None
        return all(latency <= target_s for _, latency in self.samples)


def target_states_meet_accuracy(
    result: AccuracyResult,
    *,
    states: tuple[str, ...] = TARGET_STATES,
    target: float = ACCURACY_TARGET,
) -> dict[str, bool | None]:
    """For each of `states`, whether its recall meets `target`.

    `None` means the state has zero support in `result` -- not measured,
    not failed. Callers (the CLI's exit-code logic) must treat `None`
    differently from `False`.
    """
    verdicts: dict[str, bool | None] = {}
    for state in states:
        stats = result.per_state.get(state)
        recall = stats.recall if stats is not None else None
        verdicts[state] = None if recall is None else recall >= target
    return verdicts
