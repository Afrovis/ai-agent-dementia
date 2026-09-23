"""Ordered output comparison for live and in-process traces.

Outputs are matched by content (Say strategy, goal change, light state, notify level,
phase change, Show face/headline) in order, so the same decision made at a different time
lines up and its time shift becomes the difference to explain. SessionState heartbeats
that repeat the previous phase, goal and strategy index are not decisions and are skipped.
"""

from __future__ import annotations

from .trace import Trace, TraceEvent

OUTPUTS = {"SessionState", "GoalChanged", "Say", "Show", "Notify", "LightCommand"}


def _key(event: TraceEvent) -> tuple:
    d = event.data
    if event.type == "Say":
        return ("Say", d.get("strategy"))
    if event.type == "GoalChanged":
        return ("GoalChanged", d.get("from_goal"), d.get("to_goal"))
    if event.type == "LightCommand":
        return ("LightCommand", d.get("state"))
    if event.type == "Notify":
        return ("Notify", d.get("level"))
    if event.type == "SessionState":
        return ("SessionState", d.get("phase"), d.get("goal"), d.get("strategy_index"))
    if event.type == "Show":
        return ("Show", d.get("face"), d.get("headline"))
    return (event.type,)


def _decisions(trace: Trace) -> list[TraceEvent]:
    out: list[TraceEvent] = []
    last_session = None
    for event in trace.events:
        if event.type not in OUTPUTS:
            continue
        if event.type == "SessionState":
            key = _key(event)
            if key == last_session:
                continue
            last_session = key
        out.append(event)
    return out


def _label(event: TraceEvent) -> str:
    return " ".join(str(part) for part in _key(event)[1:] if part is not None)


def compare(live: Trace, inprocess: Trace, tolerance_s: float = 3) -> list[dict]:
    left = _decisions(live)
    right = _decisions(inprocess)
    rows = []
    used: set[int] = set()
    for event in left:
        key = _key(event)
        # First unused decision with the same content; events published in one loop tick
        # can come out in a different order, so no strict ordering cursor.
        match = next(
            (i for i in range(len(right)) if i not in used and _key(right[i]) == key), None
        )
        if match is None:
            rows.append(
                {
                    "status": "only-live",
                    "type": event.type,
                    "what": _label(event),
                    "live_t": round(event.t, 2),
                    "inprocess_t": None,
                    "delta_s": None,
                }
            )
            continue
        used.add(match)
        delta = round(event.t - right[match].t, 2)
        rows.append(
            {
                "status": "matched" if abs(delta) <= tolerance_s else "shifted",
                "type": event.type,
                "what": _label(event),
                "live_t": round(event.t, 2),
                "inprocess_t": round(right[match].t, 2),
                "delta_s": delta,
            }
        )
    rows.extend(
        {
            "status": "only-in-process",
            "type": event.type,
            "what": _label(event),
            "live_t": None,
            "inprocess_t": round(event.t, 2),
            "delta_s": None,
        }
        for i, event in enumerate(right)
        if i not in used
    )
    return rows


def render(rows: list[dict]) -> str:
    keys = ("status", "type", "what", "live_t", "inprocess_t", "delta_s")
    lines = [
        "| Status | Type | What | Live s | In-process s | Delta s |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[key]) if row[key] is not None else "" for key in keys) + " |"
        )
    lines.append(
        "\n"
        + ", ".join(
            f"{status}: {sum(row['status'] == status for row in rows)}"
            for status in ("matched", "shifted", "only-live", "only-in-process")
        )
    )
    return "\n".join(lines)
