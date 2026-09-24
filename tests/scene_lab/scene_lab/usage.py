"""Claude token use per call, appended to the run's `usage.jsonl`.

Every `claude -p` call in a run (director, mind, triage stages) writes one row,
including failed calls when the CLI's error still carries its usage (a usage
limit does). `summarize` totals them per role and model for `scene_lab usage`.
Costs are Claude Code's list-price estimate; the calls run on the subscription.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "thinking")


def payload_from_error(message: str | None) -> dict | None:
    """The CLI's JSON result embedded in an error message, if there is one."""
    if not message or "{" not in message:
        return None
    try:
        payload = json.loads(message[message.index("{") :])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def from_payload(payload: dict | None) -> dict:
    """Token counts from a raw CLI result or from `run_claude`'s return value."""
    payload = payload or {}
    usage = payload.get("usage") or {}
    model_usage = payload.get("modelUsage") or payload.get("model_usage") or {}
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0
    duration = payload.get("duration_ms")
    return {
        "model": next(iter(model_usage), None) or payload.get("model"),
        "input": usage.get("input_tokens") or 0,
        "output": usage.get("output_tokens") or 0,
        "cache_read": usage.get("cache_read_input_tokens") or 0,
        "cache_write": usage.get("cache_creation_input_tokens") or 0,
        "thinking": thinking,
        "cost_usd": round(payload.get("total_cost_usd") or 0, 4),
        "turns": payload.get("num_turns"),
        "duration_s": round(duration / 1000, 1) if duration else None,
    }


def record(
    path: Path | None,
    role: str,
    *,
    payload: dict | None = None,
    error: str | None = None,
    requested_model: str | None = None,
    **extra,
) -> None:
    """Append one row; `path` None disables logging (unit tests, ad-hoc use)."""
    if path is None:
        return
    if payload is None and error:
        payload = payload_from_error(error)
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "role": role,
        "requested_model": requested_model,
        "ok": error is None,
        **from_payload(payload),
        **extra,
    }
    if error is not None:
        row["error"] = error[:300]
    with Path(path).open("a") as out:
        out.write(json.dumps(row) + "\n")


def summarize(path: Path) -> str:
    """A markdown table of calls and tokens per role and model."""
    if not path.exists():
        return f"No usage recorded: {path} does not exist.\n"
    totals: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"calls": 0, "failed": 0, "cost_usd": 0.0, **dict.fromkeys(TOKEN_FIELDS, 0)}
    )
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total = totals[(row["role"], row.get("model") or row.get("requested_model") or "-")]
        total["calls"] += 1
        total["failed"] += not row.get("ok", True)
        total["cost_usd"] += row.get("cost_usd") or 0
        for name in TOKEN_FIELDS:
            total[name] += row.get(name) or 0
    lines = [
        "| role | model | calls | failed | input | cache read | cache write | output "
        "| thinking | list cost (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    grand = 0.0
    for (role, model), t in sorted(totals.items()):
        grand += t["cost_usd"]
        lines.append(
            f"| {role} | {model} | {t['calls']} | {t['failed']} | {t['input']:,} "
            f"| {t['cache_read']:,} | {t['cache_write']:,} | {t['output']:,} "
            f"| {t['thinking']:,} | {t['cost_usd']:.2f} |"
        )
    lines.append(f"\nTotal list-price estimate: {grand:.2f} USD (subscription, not billed).")
    return "\n".join(lines) + "\n"
