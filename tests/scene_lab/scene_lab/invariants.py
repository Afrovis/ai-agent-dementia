"""Deterministic trace invariants. Utterance.t is its end, not its start."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable
from statistics import median
from typing import Literal

from agent.questions import is_direct_question
from pydantic import BaseModel

from .thresholds import Thresholds
from .trace import Trace, TraceEvent, playbacks


class InvariantResult(BaseModel):
    id: str
    severity: Literal["critical", "major", "minor", "review", "info"]
    passed: bool
    window: tuple[float, float]
    evidence: list[str]
    reason: str
    context: dict = {}
    t: float = 0.0


def is_question_or_request(text: str) -> bool:
    """English keyword heuristic; it deliberately favors review over missing a request."""
    lower = text.strip().lower()
    return (
        lower.endswith("?")
        or bool(
            re.match(
                r"^(?:where|when|what|who|how|why|can|could|will|would|is|are|do|does)\b", lower
            )
        )
        or any(
            phrase in lower
            for phrase in (
                "i need",
                "i want",
                "help me",
                "take me",
                "let me",
                "i have to",
                "i must",
            )
        )
    )


def _latest(trace: Trace, name: str, t: float) -> TraceEvent | None:
    return next((e for e in reversed(trace.events) if e.type == name and e.t <= t), None)


def _state(trace: Trace, t: float) -> dict:
    session = _latest(trace, "SessionState", t)
    goal = _latest(trace, "GoalChanged", t)
    say = _latest(trace, "Say", t)
    return {
        "phase": session.data.get("phase") if session else None,
        "goal": (
            goal.data.get("to_goal")
            if goal and (not session or goal.t > session.t)
            else session.data.get("goal")
            if session
            else None
        ),
        # Only a Say names the strategy id; an index alone would split fingerprints.
        "strategy": say.data.get("strategy") if say else None,
        # Set explicitly by checks that attribute a drop (TT-1); never inherited.
        "drop_reason": None,
    }


def _result(
    trace: Trace,
    id: str,
    severity: str,
    passed: bool,
    t: float,
    end: float,
    reason: str,
    evidence: list[str] | None = None,
    context: dict | None = None,
) -> InvariantResult:
    return InvariantResult(
        id=id,
        severity=severity,
        passed=passed,
        t=t,
        window=(t, end),
        reason=reason,
        evidence=evidence or [],
        context=context if context is not None else _state(trace, t),
    )


def _said_record(trace: Trace, say: TraceEvent) -> TraceEvent | None:
    """The agent's `said` decision record for this Say (same strategy, same loop tick).

    The closest record in time wins, so two Says of one strategy in quick succession (a
    deferred flush followed by a fresh Say) cannot swap records.
    """
    candidates = [
        e
        for e in trace.events
        if e.kind == "decision"
        and e.data.get("decision") == "said"
        and e.data.get("strategy") == say.data.get("strategy")
        and abs(e.t - say.t) <= 0.5
    ]
    return min(candidates, key=lambda e: abs(e.t - say.t), default=None)


def _first_reply(trace: Trace, utt: TraceEvent) -> TraceEvent | None:
    """First Say after the utterance and before the next one that answers it.

    When the agent records why it spoke (`said` records), a Say caused by a dwell timer
    or zone change is a scheduled step, not a reply, even if it follows the utterance.
    Traces without those records count any following Say.
    """
    next_utt = next((e.t for e in trace.of_type("Utterance") if e.t > utt.t), float("inf"))
    following = [e for e in trace.of_type("Say") if utt.t <= e.t < next_utt]
    if not any(e.kind == "decision" and e.data.get("decision") == "said" for e in trace.events):
        return following[0] if following else None
    for say in following:
        record = _said_record(trace, say)
        if record is None or record.data.get("reply"):
            return say
    return None


def _scheduled_step_after(trace: Trace, utt: TraceEvent) -> TraceEvent | None:
    next_utt = next((e.t for e in trace.of_type("Utterance") if e.t > utt.t), float("inf"))
    return next(
        (
            e
            for e in trace.of_type("Say")
            if utt.t <= e.t < next_utt
            and (record := _said_record(trace, e)) is not None
            and not record.data.get("reply")
        ),
        None,
    )


def _summary(
    trace: Trace,
    id: str,
    count: int,
    results: list[InvariantResult],
    why: str = "no events to evaluate",
) -> None:
    if not count:
        results.append(_result(trace, id, "info", True, 0, 0, f"not applicable: {why}"))
    elif not any(r.id == id and not r.passed for r in results):
        results.append(
            _result(
                trace,
                id,
                "info",
                True,
                0,
                trace.end_t,
                f"{count} checked, all passed",
                context={"count": count},
            )
        )


def _llm_busy_intervals(trace: Trace) -> list[tuple[float, float, list[str]]]:
    """Merged [start, end) intervals when an agent LLM call blocked the loop."""
    raw: list[tuple[float, float, str]] = []
    open_calls: dict[str, float] = {}
    for event in trace.of_type("Activity"):
        kind = event.data.get("kind")
        if kind not in {"interpret", "compose", "plan"}:
            continue
        duration = (event.data.get("duration_ms") or 0) / 1000
        if event.data.get("phase") == "start":
            open_calls[kind] = event.t
        elif event.data.get("phase") == "end":
            began = open_calls.pop(kind, None)
            if duration:
                began = event.t - duration
            if began is not None and event.t > began:
                raw.append((began, event.t, f"{kind} {event.t - began:.2f}s"))
    merged: list[tuple[float, float, list[str]]] = []
    for a, b, label in sorted(raw):
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2] + [label])
        else:
            merged.append((a, b, [label]))
    return merged


def _spoke_after_toilet_need(trace: Trace, t: float) -> bool:
    """True when the person said anything after their latest toilet request (up to t).

    Only then could an unobserved wants_bed interpretation have resolved the need.
    """
    from agent.veto import _TOILET_RE

    said = [u for u in trace.of_type("Utterance") if u.t <= t]
    last_need = max(
        (u.t for u in said if _TOILET_RE.search(str(u.data.get("text", "")))), default=None
    )
    return last_need is not None and any(u.t > last_need for u in said)


def _speech_end(trace: Trace, utt: TraceEvent) -> float:
    """When the person stopped speaking: listen starts transcribing at end of speech.

    Live traces carry listen's transcribe Activity; offline traces fall back to the
    Utterance publish time.
    """
    start = next(
        (
            e
            for e in reversed(trace.events)
            if e.type == "Activity"
            and e.data.get("kind") == "transcribe"
            and e.data.get("phase") == "start"
            and utt.t - 30 <= e.t <= utt.t
        ),
        None,
    )
    return start.t if start else utt.t


def _utterance_times(trace: Trace) -> list[float]:
    """When the agent registered each utterance.

    `run_once` calls `Session.on_utterance(now)` as soon as it reads the Utterance, before
    the interpret call, so the Utterance time is the right estimate (plus any time the loop
    was blocked, which the trace cannot always see).
    """
    return sorted(u.t for u in trace.of_type("Utterance"))


def _restroom_need_resolved(trace: Trace, t: float) -> bool:
    """Mirror Session._restroom_need_resolved from GoalChanged events up to t."""
    resolved = False
    for event in trace.events:
        if event.t > t:
            break
        if event.type == "SessionState" and event.data.get("phase") == "IDLE":
            resolved = False
        elif (
            event.kind == "decision"
            and event.data.get("decision") == "interpreted"
            and event.data.get("intent") == "wants_bed"
            and event.data.get("goal") == "return_to_bed"
        ):
            resolved = True
        elif event.type == "GoalChanged":
            if event.data.get("to_goal") == "restroom":
                resolved = False
            elif event.data.get("from_goal") == "restroom" and event.data.get("reason") in {
                "returned_from_bathroom",
                # Session.on_interpretation sets the flag after a wants_bed goal change.
                "interpreted_wants_bed",
            }:
                resolved = True
    return resolved


def check_trace(
    trace: Trace, thresholds: Thresholds, judge: Callable[[str, str | None], str] | None = None
) -> list[InvariantResult]:
    out: list[InvariantResult] = []
    pbs = playbacks(trace, thresholds)
    says = trace.of_type("Say")
    utts = trace.of_type("Utterance")
    people = trace.of_type("PersonState")
    sessions = trace.of_type("SessionState")
    replies: list[float] = []
    count = 0
    for i, utt in enumerate(utts):
        if _state(trace, utt.t)["phase"] not in thresholds.active_phases:
            continue
        next_utt = utts[i + 1] if i + 1 < len(utts) else None
        if (
            next_utt
            and next_utt.t - utt.t <= thresholds.reply_deadline_s
            and not any(utt.t <= s.t < next_utt.t for s in says)
        ):
            continue
        count += 1
        say = _first_reply(trace, utt)
        pb = next((p for p in pbs if p.say_t == say.t), None) if say else None
        spoke_end = _speech_end(trace, utt)
        if pb and pb.start - utt.t <= thresholds.direct_reply_window_s:
            lag = pb.start - spoke_end
            replies.append(lag)
            if lag > thresholds.reply_deadline_s:
                out.append(
                    _result(
                        trace,
                        "TT-1",
                        "major",
                        False,
                        utt.t,
                        pb.start,
                        "late reply",
                        [
                            f"Utterance@{utt.t}: {utt.data.get('text')}",
                            f"speech ended@{spoke_end:.2f} (transcription "
                            f"{utt.t - spoke_end:.2f}s)",
                            f"Say playback@{pb.start}: {lag:.2f}s",
                        ],
                    )
                )
        else:
            limit = min(
                next_utt.t if next_utt else float("inf"), utt.t + thresholds.direct_reply_window_s
            )
            decisions = [
                e
                for e in trace.events
                if e.kind == "decision"
                and e.data.get("decision") in {"pending_say_dropped", "vetoed", "no_reply"}
                and utt.t <= e.t <= limit
            ]
            decision = next((e for e in decisions if e.data.get("decision") == "no_reply"), None)
            if decision is None:
                decision = next(iter(decisions), None)
            designed_silence = (
                say is None and decision is not None and decision.data.get("decision") == "no_reply"
            )
            if say is not None and pb is None:
                cause = "reply composed but never played"
            elif designed_silence:
                reason = decision.data.get("reason", "unknown")
                cause = (
                    f"no reply by design to a question: {reason}"
                    if is_direct_question(utt.data.get("text", ""))
                    else f"no reply by design: {reason}"
                )
            elif decision and decision.data.get("decision") == "pending_say_dropped":
                cause = f"dropped pending say: {decision.data.get('reason', 'unknown')}"
            elif decision and decision.data.get("decision") == "vetoed":
                cause = f"vetoed: {decision.data.get('rule', 'unknown')}"
            elif _state(trace, limit)["phase"] not in thresholds.active_phases:
                cause = "phase"
            elif step := _scheduled_step_after(trace, utt):
                cause = (
                    f"only a scheduled {step.data.get('strategy')} step followed "
                    f"({_said_record(trace, step).data.get('trigger')})"
                )
            else:
                cause = "never composed"
            context = _state(trace, utt.t)
            if decision:
                context["drop_reason"] = decision.data.get("reason")
            designed_answer_to_question = designed_silence and is_direct_question(
                utt.data.get("text", "")
            )
            out.append(
                _result(
                    trace,
                    "TT-1",
                    "info" if designed_silence and not designed_answer_to_question else "critical",
                    designed_silence and not designed_answer_to_question,
                    utt.t,
                    limit,
                    cause if designed_silence else f"no reply: {cause}",
                    [f"Utterance@{utt.t}: {utt.data.get('text')}", cause],
                    context,
                )
            )
    _summary(trace, "TT-1", count, out, "no active utterances")

    for utt in utts:
        text = utt.data.get("text", "")
        if is_question_or_request(text):
            say = _first_reply(trace, utt)
            reply = say.data.get("text") if say else None
            rating = judge(text, reply) if judge else "unrated" if reply else "no_reply"
            out.append(
                _result(
                    trace,
                    "TT-2",
                    "review",
                    True,
                    utt.t,
                    say.t if say else utt.t,
                    rating,
                    [f"utterance: {text}", f"reply: {reply or '-'}"],
                    {**_state(trace, utt.t), "rating": rating},
                )
            )
    _summary(
        trace, "TT-2", len([u for u in utts if is_question_or_request(u.data.get("text", ""))]), out
    )

    starts = trace.of_type("SpeechStarted")
    for speech in starts:
        end = (
            next((u.t for u in utts if u.t >= speech.t), speech.t + 30)
            + thresholds.talk_over_grace_s
        )
        for pb in pbs:
            if speech.t <= pb.start <= end:
                out.append(
                    _result(
                        trace,
                        "TT-3",
                        "major",
                        False,
                        speech.t,
                        pb.start,
                        "playback started over speech",
                        [f"SpeechStarted@{speech.t}", f"playback@{pb.start}"],
                    )
                )
    _summary(trace, "TT-3", len(starts), out, "no SpeechStarted")

    real_active = [
        (speech, pb)
        for speech in starts
        for pb in pbs
        if not pb.estimated and pb.interruptible and pb.start <= speech.t < pb.end
    ]
    for speech, pb in real_active:
        if not pb.interrupted or pb.end - speech.t > thresholds.barge_in_s:
            out.append(
                _result(
                    trace,
                    "TT-4",
                    "major",
                    False,
                    speech.t,
                    pb.end,
                    "interruptible playback did not stop within barge-in deadline",
                    [f"playback end@{pb.end}"],
                )
            )
    _summary(
        trace,
        "TT-4",
        len(real_active),
        out,
        "no real interruptible playback active at SpeechStarted",
    )

    for prev, current in zip(pbs, pbs[1:]):
        direct = any(
            prev.start < u.t <= current.say_t
            and current.say_t - u.t <= thresholds.direct_reply_window_s
            and not any(u.t <= s.t < current.say_t for s in says)
            for u in utts
        )
        if current.start - prev.end < thresholds.silence_gap_s and not direct:
            out.append(
                _result(
                    trace,
                    "TT-5",
                    "minor" if current.estimated or prev.estimated else "major",
                    False,
                    current.start,
                    current.end,
                    "silence gap too short",
                    [
                        f"gap={current.start - prev.end:.2f}s",
                        f"estimated={current.estimated or prev.estimated}",
                    ],
                )
            )
    _summary(trace, "TT-5", max(0, len(pbs) - 1), out)

    for say in says:
        person = _latest(trace, "PersonState", say.t)
        spoke = _latest(trace, "Utterance", say.t)
        heard_recently = spoke is not None and say.t - spoke.t <= thresholds.utterance_presence_s
        if person and person.data.get("state") == "absent" and not heard_recently:
            out.append(
                _result(
                    trace,
                    "TT-6",
                    "major",
                    False,
                    say.t,
                    say.t,
                    "Say while person absent",
                    [f"strategy={say.data.get('strategy')}"],
                )
            )
    _summary(trace, "TT-6", len(says), out)

    # A reply less than one second after a newer utterance is likely an old compose completing.
    for first, second in zip(utts, utts[1:]):
        if second.t - first.t <= thresholds.reply_deadline_s:
            continue
        if any(first.t <= s.t < second.t for s in says):
            continue
        say = next((s for s in says if second.t < s.t < second.t + 1), None)
        if say:
            out.append(
                _result(
                    trace,
                    "TT-7",
                    "minor",
                    False,
                    second.t,
                    say.t,
                    "likely stale reply to earlier utterance",
                    [f"earlier utterance@{first.t}", f"new utterance@{second.t}", f"Say@{say.t}"],
                )
            )
    _summary(trace, "TT-7", len(utts) - 1 if len(utts) > 1 else 0, out)

    # Settled, as in the agent's NICE-05 rule: in bed and silent since lying down. Speaking
    # from bed restarts the settle clock, so a reply to the person is not "after settling".
    heard = _utterance_times(trace)
    settled_windows = []
    for i, person in enumerate(people):
        if person.data.get("state") != "in_bed" or (
            i and people[i - 1].data.get("state") == "in_bed"
        ):
            continue
        end = next((p.t for p in people[i + 1 :] if p.data.get("state") != "in_bed"), trace.end_t)
        breaks = [person.t] + [u for u in heard if person.t <= u < end] + [end]
        for begin, stop in zip(breaks, breaks[1:]):
            if stop - begin >= thresholds.settle_s:
                settled_windows.append((begin + thresholds.settle_s, stop))
    for start, end in settled_windows:
        for event in trace.events:
            if not start <= event.t < end:
                continue
            previous_session = next((s for s in reversed(sessions) if s.t < event.t), None)
            strategy_advanced = (
                event.type == "SessionState"
                and previous_session is not None
                and event.data.get("strategy_index") is not None
                and previous_session.data.get("strategy_index") is not None
                and event.data["strategy_index"] > previous_session.data["strategy_index"]
            )
            bad = (
                event.type in {"GoalChanged", "Notify"}
                or strategy_advanced
                or (
                    event.type == "Say"
                    and not str(event.data.get("strategy", "")).startswith("escalate")
                )
            )
            if bad:
                out.append(
                    _result(
                        trace,
                        "SM-1",
                        "critical",
                        False,
                        event.t,
                        event.t,
                        "activity after settling in bed",
                        [f"{event.type}@{event.t}"],
                    )
                )
    _summary(trace, "SM-1", len(settled_windows), out)

    alerts = [
        e for e in trace.of_type("Notify") if e.data.get("level") in {"attention", "critical"}
    ]
    for alert in alerts:
        floor = any(p.t <= alert.t and p.data.get("state") == "on_floor" for p in people)
        distress = any(
            u.t <= alert.t
            and re.search(
                r"\b(pain|hurt|help|fallen|can't breathe|fell|injured)\b",
                u.data.get("text", ""),
                re.I,
            )
            for u in utts
        )
        state = _state(trace, alert.t)
        person = _latest(trace, "PersonState", alert.t)
        ladder = (
            (state["phase"] == "ESCALATED" or state["goal"] == "wait_for_caregiver")
            and person
            and person.data.get("state") != "in_bed"
        )
        if not (floor or distress or ladder):
            out.append(
                _result(
                    trace,
                    "SM-2",
                    "critical",
                    False,
                    alert.t,
                    alert.t,
                    "escalation without floor, distress, or exhausted ladder",
                )
            )
    _summary(trace, "SM-2", len(alerts), out)

    # Merge repeated SessionState events into contiguous ENGAGED/ESCALATED intervals so a
    # re-published phase does not report the same gap twice.
    active: list[tuple[float, float]] = []
    for session in sessions:
        is_active = session.data.get("phase") in {"ENGAGED", "ESCALATED"}
        if is_active and (not active or active[-1][1] != math.inf):
            active.append((session.t, math.inf))
        elif not is_active and active and active[-1][1] == math.inf:
            active[-1] = (active[-1][0], session.t)
    if active and active[-1][1] == math.inf:
        active[-1] = (active[-1][0], trace.end_t)
    for begin, end in active:
        person = _latest(trace, "PersonState", begin)
        gap_start = (
            begin if person and person.data.get("state") not in {"in_bed", "absent"} else None
        )
        changes = [
            event
            for event in trace.events
            if begin < event.t < end and event.type in {"PersonState", "Say"}
        ]
        for event in changes + [None]:
            at = event.t if event else end
            closes_gap = (
                event is None
                or event.type == "Say"
                or event.data.get("state") in {"in_bed", "absent"}
            )
            if (
                closes_gap
                and gap_start is not None
                and at - gap_start > thresholds.silent_session_s
            ):
                state = _state(trace, gap_start)
                last_say = _latest(trace, "Say", gap_start)
                reading = _latest(trace, "PersonState", at)
                out.append(
                    _result(
                        trace,
                        "SM-3",
                        "major",
                        False,
                        gap_start,
                        at,
                        "silent session gap",
                        [
                            f"{state.get('phase')} goal={state.get('goal')} "
                            f"strategy={state.get('strategy')}",
                            f"silence {at - gap_start:.1f}s from {gap_start} to {at}",
                            f"last Say@{last_say.t}: {last_say.data.get('text')}"
                            if last_say
                            else "no Say before the gap",
                            f"person {reading.data.get('state')}/{reading.data.get('zone')}"
                            if reading
                            else "no reading",
                        ],
                    )
                )
            if event is None:
                break
            if event.type == "Say":
                if gap_start is not None:
                    gap_start = event.t
            elif event.data.get("state") in {"in_bed", "absent"}:
                gap_start = None
            elif gap_start is None:
                gap_start = event.t
    # "Did not end" only applies once the person has been back in bed long enough for the
    # session to close (settle + tail); a person still up at the end of a bench timeline is
    # an open session by design.
    readings = trace.of_type("PersonState")
    bed_since = None
    for reading in readings:
        if reading.data.get("state") == "in_bed":
            bed_since = reading.t if bed_since is None else bed_since
        else:
            bed_since = None
    final_phase = sessions[-1].data.get("phase") if sessions else None
    if (
        sessions
        and bed_since is not None
        and trace.end_t - bed_since >= thresholds.settle_s + thresholds.tail_s
        and final_phase not in {"IDLE", "COOLDOWN", "ESCALATED"}
    ):
        out.append(
            _result(
                trace,
                "SM-3",
                "major",
                False,
                bed_since,
                trace.end_t,
                "session did not end",
                [
                    f"in_bed since {bed_since}, trace ends {trace.end_t}",
                    f"final phase {final_phase}",
                ],
            )
        )
    _summary(trace, "SM-3", len(active) + bool(sessions), out)

    lights = trace.of_type("LightCommand")
    for on in (e for e in lights if e.data.get("state") == "on"):
        stable = next((start for start, end in settled_windows if start >= on.t), None)
        if stable is not None and not any(
            e.t >= stable and e.data.get("state") == "off" for e in lights
        ):
            out.append(
                _result(
                    trace,
                    "SM-4",
                    "minor",
                    False,
                    on.t,
                    trace.end_t,
                    "light remained on after settling",
                )
            )
        elif stable is None and not any(
            e.t > on.t and e.data.get("state") == "off" for e in lights
        ):
            out.append(
                _result(
                    trace, "SM-4", "minor", False, on.t, trace.end_t, "trace ended before light off"
                )
            )
    _summary(trace, "SM-4", sum(e.data.get("state") == "on" for e in lights), out)

    _check_veto(trace, out, thresholds)

    changed = [
        p
        for i, p in enumerate(people)
        if i == 0
        or (p.data.get("state"), p.data.get("zone"))
        != (people[i - 1].data.get("state"), people[i - 1].data.get("zone"))
    ]
    # Loop lag: the agent runs LLM calls synchronously inside run_once, so an input that
    # arrives while a call (or a back-to-back chain of calls) is running waits until the chain
    # ends. Delays with no LLM call running are deliberate confirmation/dwell timing, not lag.
    busy = _llm_busy_intervals(trace)
    inputs = sorted(changed + utts + trace.of_type("SpeechStarted"), key=lambda e: e.t)
    counted = len(inputs) if busy else 0
    for item in inputs:
        blocking = next(((a, b, calls) for a, b, calls in busy if a <= item.t < b), None)
        if blocking is None:
            continue
        begin, free_at, calls = blocking
        wait = free_at - item.t
        if wait > thresholds.loop_lag_s:
            out.append(
                _result(
                    trace,
                    "TM-1",
                    "minor",
                    False,
                    item.t,
                    free_at,
                    "loop lag exceeded",
                    [
                        f"input {item.type}@{item.t}",
                        f"loop busy {begin:.2f}-{free_at:.2f}s, waited {wait:.2f}s",
                        "LLM " + ", ".join(calls),
                    ],
                )
            )
    _summary(trace, "TM-1", counted, out, "no LLM activity with durations in trace")

    ordered = sorted(replies)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None
    out.append(
        _result(
            trace,
            "TM-2",
            "info",
            True,
            0,
            trace.end_t,
            "reply latency distribution",
            context={
                "n": len(replies),
                "p50": median(replies) if replies else None,
                "p95": p95,
                "max": max(replies) if replies else None,
            },
        )
    )
    bathroom = next((p for p in people if p.data.get("zone") == "bathroom_path"), None)
    goal = next((g for g in trace.of_type("GoalChanged") if bathroom and g.t >= bathroom.t), None)
    out.append(
        _result(
            trace,
            "TM-3",
            "info",
            True,
            bathroom.t if bathroom else 0,
            goal.t if goal else trace.end_t,
            "bathroom path to goal latency" if bathroom else "not applicable: no bathroom path",
            context={"latency_s": goal.t - bathroom.t if goal and bathroom else None},
        )
    )
    out.extend(wording_checks(trace))
    return out


def _check_veto(trace: Trace, out: list[InvariantResult], thresholds: Thresholds) -> None:
    try:
        from agent.veto import Proposal, VetoContext, check
    except ImportError:
        out.append(
            _result(trace, "SM-5", "info", True, 0, 0, "not applicable: agent.veto unavailable")
        )
        return
    actions = [e for e in trace.events if e.type in {"Say", "Show", "Notify"}]
    checked = 0
    for action in actions:
        state = _state(trace, action.t)
        person = _latest(trace, "PersonState", action.t)
        missing = [
            name
            for name, value in (
                ("phase", state["phase"]),
                ("goal", state["goal"]),
                ("person_state", person),
            )
            if value is None
        ]
        if missing:
            out.append(
                _result(
                    trace,
                    "SM-5",
                    "info",
                    True,
                    action.t,
                    action.t,
                    f"not applicable: missing {', '.join(missing)}",
                )
            )
            continue
        # Facts not on bus events. in_night_window defaults to True: decision_bench scenarios
        # and the nightsim stack are always night; session_replay sets it from its capture.
        # things_to_avoid comes from explicit meta or the profile. restroom_need_resolved is
        # session state, derived from GoalChanged below.
        profile_meta = trace.meta.get("profile") or {}
        things_to_avoid = trace.meta.get("things_to_avoid")
        if things_to_avoid is None and isinstance(profile_meta, dict):
            things_to_avoid = profile_meta.get("things_to_avoid") or ()
        in_night_window = bool(trace.meta.get("in_night_window", True))
        recent_texts: list[str] = []
        for event in trace.events:
            if event.t > action.t:
                break
            if event.type == "SessionState" and event.data.get("phase") == "IDLE":
                recent_texts.clear()
            elif event.type == "Utterance" and _state(trace, event.t)["phase"] in {
                "OBSERVING",
                "ENGAGED",
                "ESCALATED",
            }:
                recent_texts.append(event.data.get("text", ""))
        recent = tuple(recent_texts[-3:])
        readings = [p for p in trace.of_type("PersonState") if p.t <= action.t]
        last_lie = None
        for reading in readings:
            if reading.data.get("state") == "in_bed" and last_lie is None:
                last_lie = reading.t
            elif reading.data.get("state") != "in_bed":
                last_lie = None
        settled = (
            person.data.get("state") == "in_bed"
            and last_lie is not None
            and not any(last_lie <= u <= action.t for u in _utterance_times(trace))
        )
        ctx = VetoContext(
            phase=state["phase"],
            goal=state["goal"],
            person_state=person.data.get("state"),
            recent_utterances=recent,
            settled=settled,
            in_night_window=in_night_window,
            things_to_avoid=tuple(things_to_avoid or ()),
            restroom_need_resolved=_restroom_need_resolved(trace, action.t),
        )
        proposals = []
        if action.type == "Say":
            proposals = [
                Proposal("strategy", action.data.get("strategy", "")),
                Proposal(
                    "say",
                    action.data.get("strategy", ""),
                    text=action.data.get("text", ""),
                    terminal=str(action.data.get("strategy", "")).startswith("escalate"),
                ),
            ]
        elif action.type == "Show":
            if action.data.get("strategy"):
                proposals = [Proposal("strategy", action.data["strategy"])]
            else:
                out.append(
                    _result(
                        trace,
                        "SM-5",
                        "info",
                        True,
                        action.t,
                        action.t,
                        "not applicable: missing Show strategy",
                    )
                )
                continue
        elif action.type == "Notify":
            proposals = [Proposal("notify", action.data.get("level", ""))]
        checked += 1
        # The person's latest reading may not have been read by the agent yet: the loop reads
        # inputs only between LLM calls, and some calls (plan) publish no Activity, so a
        # reading this recent may still be unseen when the action was decided.
        latest_reading = readings[-1] if readings else None
        recent_reading = (
            latest_reading is not None and action.t - latest_reading.t < thresholds.loop_lag_s
        )
        for proposal in proposals:
            verdict = check(proposal, ctx)
            if not verdict.allowed and recent_reading and len(readings) > 1:
                before = readings[-2]
                unseen = dataclasses.replace(
                    ctx,
                    person_state=before.data.get("state"),
                    settled=ctx.settled and before.data.get("state") == "in_bed",
                )
                if check(proposal, unseen).allowed:
                    out.append(
                        _result(
                            trace,
                            "SM-5",
                            "minor",
                            False,
                            action.t,
                            action.t,
                            f"possibly unseen state: {verdict.rule} depends on the "
                            f"{latest_reading.data.get('state')} reading "
                            f"{action.t - latest_reading.t:.1f}s before the action",
                            [
                                f"{action.type}@{action.t}: {action.data.get('text') or ''}",
                                f"PersonState@{latest_reading.t}: "
                                f"{latest_reading.data.get('state')}",
                            ],
                        )
                    )
                    continue
            if not verdict.allowed and not ctx.restroom_need_resolved:
                # restroom_need_resolved is derived; a wants_bed intent can also set it without
                # a GoalChanged. A denial that disappears when it is True is only a suspicion.
                resolved = check(proposal, dataclasses.replace(ctx, restroom_need_resolved=True))
                interpretations_recorded = any(
                    e.kind == "decision" and e.data.get("decision") == "interpreted"
                    for e in trace.events
                )
                if (
                    resolved.allowed
                    and not interpretations_recorded
                    and _spoke_after_toilet_need(trace, action.t)
                ):
                    out.append(
                        _result(
                            trace,
                            "SM-5",
                            "minor",
                            False,
                            action.t,
                            action.t,
                            f"possible veto bypass: {verdict.rule} "
                            "(depends on restroom_need_resolved, not observable in the trace)",
                            [f"{action.type}@{action.t}: {action.data.get('text') or ''}"],
                        )
                    )
                    continue
            if not verdict.allowed:
                out.append(
                    _result(
                        trace,
                        "SM-5",
                        "critical",
                        False,
                        action.t,
                        action.t,
                        f"published action vetoed: {verdict.rule}",
                        [action.type, verdict.reason or ""],
                    )
                )
    _summary(trace, "SM-5", checked, out, "no faithfully reconstructable actions")


# Checks that are violations on any Say. addresses_by_name is a positive requirement (it fails
# whenever the name is absent), so it is scenario-specific and not run here.
WORDING_CHECKS = frozenset(
    {"conjunction_but", "avoid_terms", "states_clock_time", "invents_proper_noun"}
)


def wording_checks(trace: Trace) -> list[InvariantResult]:
    try:
        from dialogue_bench.checks import CHECK_NAMES, CheckContext, CheckStatus, run_checks
    except ImportError:
        return []
    out = []
    for say in trace.of_type("Say"):
        utt = _latest(trace, "Utterance", say.t)
        ctx = CheckContext(
            profile=trace.meta.get("profile", {}),
            utterance=utt.data.get("text") if utt else None,
            time_words="",
            scene_note=None,
            caregiver_phrase_template="",
            avoid_terms=tuple(trace.meta.get("things_to_avoid") or ()),
        )
        for hit in run_checks(say.data.get("text", ""), sorted(WORDING_CHECKS & CHECK_NAMES), ctx):
            if hit.status == CheckStatus.FAIL:
                out.append(
                    _result(
                        trace,
                        f"WORD-{hit.name}",
                        "minor",
                        False,
                        say.t,
                        say.t,
                        hit.detail or hit.name,
                        [say.data.get("text", "")],
                    )
                )
    return out
