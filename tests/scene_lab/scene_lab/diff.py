"""Ordered output comparison for live and in-process traces."""

from __future__ import annotations

from .trace import Trace

OUTPUTS = {"SessionState", "GoalChanged", "Say", "Show", "Notify", "LightCommand"}


def compare(live: Trace, inprocess: Trace, tolerance_s: float = 3) -> list[dict]:
    left = [e for e in live.events if e.type in OUTPUTS]
    right = [e for e in inprocess.events if e.type in OUTPUTS]
    rows = []
    used = set()
    for event in left:
        match = next(
            (
                i
                for i, other in enumerate(right)
                if i not in used
                and other.type == event.type
                and abs(other.t - event.t) <= tolerance_s
            ),
            None,
        )
        if match is None:
            rows.append(
                {
                    "status": "only-live",
                    "type": event.type,
                    "live_t": event.t,
                    "inprocess_t": None,
                    "delta_s": None,
                }
            )
        else:
            used.add(match)
            rows.append(
                {
                    "status": "matched",
                    "type": event.type,
                    "live_t": event.t,
                    "inprocess_t": right[match].t,
                    "delta_s": round(event.t - right[match].t, 3),
                }
            )
    rows.extend(
        {
            "status": "only-in-process",
            "type": event.type,
            "live_t": None,
            "inprocess_t": event.t,
            "delta_s": None,
        }
        for i, event in enumerate(right)
        if i not in used
    )
    return rows


def render(rows: list[dict]) -> str:
    lines = [
        "| Status | Type | Live s | In-process s | Delta s |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]) if row[key] is not None else ""
                for key in ("status", "type", "live_t", "inprocess_t", "delta_s")
            )
            + " |"
        )
    lines.append(
        "\n"
        + ", ".join(
            f"{status}: {sum(row['status'] == status for row in rows)}"
            for status in ("matched", "only-live", "only-in-process")
        )
    )
    return "\n".join(lines)
