"""Phase 0 probe for docs/CLASSIFIER_BENCH.md: shared state, questions and timing helpers.

Nothing here is a benchmark of accuracy. Zero-shot Laya is expected to sit near chance;
Phase 0 only asks whether the model is fast enough on the M4 and whether the Core ML
conversion computes the same thing as torch.
"""

import resource
import statistics
import sys
import time

# The closed action space, copied from tests/decision_bench/decision_bench/schema.py.
PHASES = ["IDLE", "OBSERVING", "ENGAGED", "COOLDOWN", "ESCALATED"]
GOALS = ["bed", "restroom", "comfort"]
STRATEGIES = [
    "ambient_orient",
    "soft_greeting",
    "orient_time_place",
    "validate_and_redirect",
    "guided_return",
    "familiar_voice",
    "path_light",
    "escalate_phone",
]
NOTIFY_LEVELS = ["any", "info", "attention", "critical"]
SAY_PATTERNS = [
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
]
ACTIONS = (
    [("phase", v) for v in PHASES]
    + [("goal", v) for v in GOALS]
    + [("strategy", v) for v in STRATEGIES]
    + [("notify", v) for v in NOTIFY_LEVELS]
    + [("say", v) for v in SAY_PATTERNS]
)
assert len(ACTIONS) == 32

# restroom-01, checkpoint first-response, at t=35: they said they need the toilet and are
# walking the bathroom path. A deliberately realistic, compact record of the kind the live
# agent could build at 3am. Not the Phase 3 serialization spec, only a stand-in of the
# right length and shape.
STATE = (
    "Night-time bedside monitor. Clock 02:40:35. "
    "Person: walking, zone bathroom_path, detection confidence 0.91. "
    "Left the bed 15 seconds ago. "
    'Last utterance 23 seconds ago: "I need the toilet." (transcription confidence 0.95, '
    "said once). "
    "Session phase OBSERVING, no goal set, no strategy running, nothing said yet. "
    "Profile: Jean, 84, moderate dementia, unsteady on her feet at night, usually uses the "
    "toilet once a night, calmed by a quiet voice, dislikes being told the time is wrong, "
    "caregiver is her daughter, asleep in the next room."
)


def action_question(kind: str, value: str) -> dict:
    """One `bool` question per candidate action (Laya calls the type `noul`)."""
    if kind == "say":
        text = (
            "In this state, is it acceptable for the agent to say something now?"
            if value == "any"
            else f"In this state, would a spoken sentence with the pattern '{value}' be acceptable?"
        )
    elif kind == "notify":
        text = f"In this state, is it acceptable to notify the caregiver at level '{value}'?"
    else:
        text = f"In this state, is it acceptable for the agent to set {kind} to '{value}'?"
    return {"type": "noul", "instructions": text}


def all_action_questions() -> dict:
    return {f"{k}:{v}": action_question(k, v) for k, v in ACTIONS}


def strategy_choice() -> dict:
    return {
        "type": "choice",
        "instructions": "Which strategy fits this state best?",
        "criteria": STRATEGIES,
    }


def action_choice_32() -> dict:
    """The 32-option question that sits exactly at the Core ML bucket capacity."""
    return {
        "type": "choice",
        "instructions": "Which single action fits this state best?",
        "criteria": [f"{k}:{v}" for k, v in ACTIONS],
    }


def time_it(fn, reps: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 2),
        "min_ms": round(samples[0], 2),
        "reps": reps,
    }


def peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
