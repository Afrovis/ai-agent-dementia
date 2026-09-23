"""SM-1 and SM-5 follow the agent's own 'settled' rule (NICE-05)."""

from scene_lab.invariants import check_trace
from scene_lab.thresholds import load
from scene_lab.trace import Trace, TraceEvent


def ev(t, type, **data):
    kind = "input" if type in {"PersonState", "Utterance", "SpeechStarted"} else "output"
    return TraceEvent(t=t, kind=kind, type=type, data=data)


def interpreted(t, text):
    return TraceEvent(
        t=t, kind="decision", type="Activity", data={"decision": "interpreted", "text": text}
    )


def failures(events, rule, end=60):
    trace = Trace(id="t", source="live", events=list(events), end_t=end, meta={})
    return [r for r in check_trace(trace, load()) if r.id == rule and not r.passed]


def test_sm1_speaking_from_bed_restarts_the_settle_clock():
    # She lies down and keeps talking from bed; each reply answers her.
    talking = [
        ev(0, "PersonState", state="in_bed", zone="bed"),
        ev(40, "Utterance", text="Okay, I'll just wait then."),
        interpreted(41.5, "Okay, I'll just wait then."),
        ev(43, "Say", text="Tom will be here soon.", strategy="reassure_waiting"),
    ]
    assert not failures(talking, "SM-1")
    # Silent for 30 s after her last words: a Say after that is still flagged.
    silent_again = [
        ev(0, "PersonState", state="in_bed", zone="bed"),
        ev(10, "Utterance", text="Alright."),
        interpreted(11, "Alright."),
        ev(45, "Say", text="Good night.", strategy="soft_greeting"),
    ]
    hits = failures(silent_again, "SM-1")
    assert hits and hits[0].severity == "critical"


def test_sm5_settled_uses_the_agents_processing_time():
    # Published at 12.8, interpreted (registered) at 14.5, lay down at 14.5: the agent
    # does not count her as settled, so a greeting at 16.5 is allowed.
    events = [
        ev(0, "PersonState", state="sitting_up", zone="bed"),
        ev(0, "SessionState", phase="ENGAGED", goal="comfort", strategy_index=0),
        ev(12.8, "Utterance", text="Oh my hip really hurts."),
        interpreted(14.5, "Oh my hip really hurts."),
        ev(14.5, "PersonState", state="in_bed", zone="bed"),
        ev(16.5, "Say", text="Hello Jean, it's night-time.", strategy="soft_greeting"),
    ]
    assert not failures(events, "SM-5")
    # Without speech since lying down, the same greeting is a veto violation.
    quiet = [e for e in events if e.type not in {"Utterance", "Activity"}]
    hits = failures(quiet, "SM-5")
    assert hits and "silence_when_settled" in hits[0].reason
