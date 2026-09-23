"""Turn a flagged live scene moment into reviewable offline regressions."""

from __future__ import annotations

import json
import re
import tempfile
from datetime import timedelta
from pathlib import Path

import yaml
from decision_bench.schema import load_scenario
from session_replay.core import extract, load_expect, parse_ts, read_jsonl

MEDIA_KEYS = {"frames", "audio", "pcm16", "jpeg"}
ROOT = Path(__file__).resolve().parents[3]
REPLAY_DIR = ROOT / "tests/session_replay/scenarios"
BENCH_DIR = ROOT / "tests/decision_bench/fixtures/scenarios"


def register(subparsers):
    """Add the promote command to scene_lab's argparse subparsers."""
    parser = subparsers.add_parser("promote", help="draft offline tests from a live scene")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--at", type=float, required=True)
    parser.add_argument(
        "--to", choices=("session_replay", "decision_bench", "both"), default="session_replay"
    )
    parser.add_argument("--before", type=float, default=120)
    parser.add_argument("--after", type=float, default=0)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--name")
    claude = parser.add_mutually_exclusive_group()
    claude.add_argument("--claude", dest="claude", action="store_true")
    claude.add_argument("--no-claude", dest="claude", action="store_false")
    parser.set_defaults(claude=False)
    return parser


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _safe(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if any(part in str(key).lower() for part in MEDIA_KEYS):
                raise ValueError(f"media key in promoted text: {key}")
            _safe(child)
    elif isinstance(value, list):
        for child in value:
            _safe(child)


def _input_at(rows, event_type, at, *, predicate=lambda row: True):
    return next(
        (
            row
            for row in reversed(rows)
            if row["event_type"] == event_type and row["t"] <= at and predicate(row)
        ),
        None,
    )


def _anchor(row):
    payload = row["payload"]
    if row["event_type"] == "Utterance":
        return {"heard": payload["text"]}
    return {"person": {"state": payload["state"], "zone": payload["zone"]}}


def template(check: str, rows: list[dict], at: float, bug: dict | None = None) -> dict:
    """Draft an expectation using only evidence at or before the flagged moment."""
    bug = bug or {}
    utterance = _input_at(rows, "Utterance", at)
    person = _input_at(rows, "PersonState", at)
    absent = _input_at(
        rows, "PersonState", at, predicate=lambda r: r["payload"]["state"] == "absent"
    )
    bed = _input_at(rows, "PersonState", at, predicate=lambda r: r["payload"]["state"] == "in_bed")
    up = _input_at(
        rows,
        "PersonState",
        at,
        predicate=lambda r: r["payload"]["state"] not in {"in_bed", "absent"},
    )
    if check == "TT-1" and utterance:
        return {"after": _anchor(utterance), "within_s": 5, "events": [{"type": "Say"}]}
    if check == "TT-5" and person:
        person = next(row for row in rows if row["event_type"] == "PersonState")
        return {"after": _anchor(person), "within_s": max(0, at - person["t"]), "min_spacing_s": 8}
    if check == "TT-6" and absent:
        return {
            "after": _anchor(absent),
            "within_s": max(0, at - absent["t"]),
            "absent": [{"type": "Say"}],
        }
    if check == "SM-1" and bed:
        return {
            "after": _anchor(bed),
            "delay_s": 30,
            "within_s": max(30, at - bed["t"]),
            "absent": [{"type": "Say"}, {"type": "Notify"}, {"type": "GoalChanged"}],
        }
    if check == "SM-3" and up:
        return {"after": _anchor(up), "within_s": 60, "events": [{"type": "Say"}]}
    if check == "SM-5" and (utterance or person):
        event_type = next(
            (x for x in ("Say", "Notify", "Show") if x in str(bug.get("evidence", []))), "Say"
        )
        return {
            "after": _anchor(utterance or person),
            "within_s": max(0, at - (utterance or person)["t"]),
            "absent": [{"type": event_type}],
        }
    raise ValueError(f"cannot draft {check}: required input is absent from the window")


def _interpretations(raw_rows, selected):
    """Read explicit interpretation records where present; unknown speech stays unclear."""
    result = {}
    for row in raw_rows:
        payload = row.get("payload") or {}
        if row.get("event_type") == "Activity" and payload.get("kind") == "decision":
            try:
                detail = json.loads(payload.get("detail") or "{}")
            except (TypeError, ValueError):
                continue
            if (
                detail.get("decision") in {"interpreted", "interpretation"}
                and detail.get("text")
                and detail.get("intent")
                and "distress" in detail
            ):
                result[detail["text"]] = {
                    "intent": detail["intent"],
                    "distress": detail["distress"],
                }
    for row in selected:
        if row["event_type"] == "Utterance":
            text = row["payload"]["text"]
            result.setdefault(text, {"intent": "unclear", "distress": 0})
    return result


def _refine(spec, bugs, rows):
    from decision_bench.annotate import run_claude

    schema = {
        "type": "object",
        "properties": {"expect": {"type": "array", "items": {"type": "object"}}},
        "required": ["expect"],
        "additionalProperties": False,
    }
    with tempfile.TemporaryDirectory() as directory:
        system = Path(directory) / "system.txt"
        system.write_text(
            "Refine only the provided expect list using the session_replay vocabulary. "
            "Keep each window at or before the flagged moment. Return JSON only."
        )
        answer = run_claude(
            system,
            json.dumps({"draft": spec["expect"], "bugs": bugs, "inputs": rows}),
            schema,
            "sonnet",
            "low",
        )
    candidate = answer.get("structured_output", answer)
    refined = {**spec, "expect": candidate["expect"]}
    _validate_expect(refined, rows)
    return refined


def _validate_expect(spec, rows):
    if not isinstance(spec.get("expect"), list) or not spec["expect"]:
        raise ValueError("draft needs expectations")
    for item in spec["expect"]:
        if set(item) - {"after", "within_s", "delay_s", "min_spacing_s", "events", "absent"}:
            raise ValueError("unsupported expectation key")
        if not isinstance(item.get("after"), dict) or not 0 <= item.get("within_s", -1):
            raise ValueError("invalid expectation anchor or window")
        if not 0 <= item.get("delay_s", 0) <= item["within_s"]:
            raise ValueError("invalid expectation delay")
        if "min_spacing_s" in item and not 0 < item["min_spacing_s"]:
            raise ValueError("invalid minimum spacing")
        if not any(_anchor(row) == item["after"] for row in rows):
            raise ValueError("expectation anchor is not in promoted inputs")
        for key in ("events", "absent"):
            if not isinstance(item.get(key, []), list) or any(
                not isinstance(x, dict) or not x.get("type") for x in item.get(key, [])
            ):
                raise ValueError("invalid event pattern")
    _safe(spec)


def _write_expect(path, spec, source):
    header = f"# Source: {source}\nstatus: draft  # human approval required\n"
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
    load_expect(path)


def promote(
    run_dir: str | Path,
    *,
    scene: str,
    at: float,
    to: str = "session_replay",
    before: float = 120,
    after: float = 0,
    out_dir: str | Path | None = None,
    name: str | None = None,
    claude: bool = False,
) -> dict[str, Path]:
    """Promote one scene window, returning paths to the generated artifacts."""
    if before < 0 or after < 0 or at < 0 or to not in {"session_replay", "decision_bench", "both"}:
        raise ValueError("invalid promotion window or target")
    run_dir = Path(run_dir)
    folder = run_dir / scene
    card_path = folder / "scene.yaml"
    card = yaml.safe_load(card_path.read_text())
    _safe(card)
    if card.get("id") != scene:
        raise ValueError("scene id does not match scene card")
    export = read_jsonl(folder / "export.jsonl")
    if not export:
        raise ValueError("scene export is empty")
    origin = parse_ts(export[0]["ts"])
    if (folder / "report.json").exists():
        report = json.loads((folder / "report.json").read_text())
        if report.get("scene_start"):
            origin = parse_ts(report["scene_start"])
    last = max((parse_ts(r["ts"]) - origin).total_seconds() for r in export)
    start, end = max(0, at - before), min(last, at + after)
    if start > end or at > last:
        raise ValueError("flagged moment lies outside scene export")
    selected = []
    for row in export:
        t = (parse_ts(row["ts"]) - origin).total_seconds()
        if start <= t <= end:
            selected.append({**row, "t": t})
    inputs = [
        r
        for r in selected
        if r.get("event_type") in {"PersonState", "Utterance"}
        and r.get("stream") in {"person", "speech_in"}
    ]
    if not inputs:
        raise ValueError("window contains no replay inputs; increase --before")
    bugs = [
        r
        for r in read_jsonl(run_dir / "bugs.jsonl")
        if r.get("scene") == scene and abs(float(r.get("t", -9999)) - at) <= 2
    ]
    if not bugs:
        print(
            f"No bug entry for {scene} within 2 s of {at:g}; promoting without an invariant draft."
        )
    slug = _slug(name or f"promoted-{scene}-{at:g}")
    if not slug:
        raise ValueError("name has no slug characters")
    outputs = {}
    targets = ["session_replay", "decision_bench"] if to == "both" else [to]
    for target in targets:
        destination = (
            Path(out_dir) / target
            if out_dir and to == "both"
            else Path(out_dir)
            if out_dir
            else REPLAY_DIR
            if target == "session_replay"
            else BENCH_DIR
        )
        destination.mkdir(parents=True, exist_ok=True)
        if target == "session_replay":
            scenario = destination / f"{slug}.jsonl"
            with tempfile.TemporaryDirectory() as directory:
                staged = Path(directory) / "scenario.jsonl"
                extract(
                    folder / "export.jsonl",
                    staged,
                    since=origin + timedelta(seconds=start),
                    until=origin + timedelta(seconds=end),
                )
                kept = read_jsonl(staged)
                _safe(kept)
                if not any(r.get("event_type") in {"PersonState", "Utterance"} for r in kept):
                    raise ValueError("extract produced no replay inputs")
                scenario.write_bytes(staged.read_bytes())
            spec = {
                "llm": "recorded",
                "interpretations": _interpretations(export, kept),
                "expect": [
                    template(b["check"], inputs, at, b)
                    for b in bugs
                    if b.get("check") in {"TT-1", "TT-5", "TT-6", "SM-1", "SM-3", "SM-5"}
                ],
            }
            interpretation_row = {
                "event_type": "InterpretationMap",
                "observed": True,
                "payload": {"interpretations": spec["interpretations"]},
            }
            _safe(interpretation_row)
            with scenario.open("a") as handle:
                handle.write(json.dumps(interpretation_row, ensure_ascii=False) + "\n")
            if spec["expect"]:
                _validate_expect(spec, inputs)
                if claude:
                    try:
                        spec = _refine(spec, bugs, inputs)
                    except Exception as exc:
                        print(f"Claude refinement failed; keeping template: {exc}")
                expect_path = destination / f"{slug}.expect.yaml"
                checks = ", ".join(b["check"] for b in bugs)
                _write_expect(
                    expect_path, spec, f"{run_dir.name}/{scene} t={at:g}; checks: {checks}"
                )
                outputs["expect"] = expect_path
                print(expect_path.read_text())
                print(
                    f"python -m session_replay run {scenario} --expect {expect_path} "
                    "--llm recorded --llm-latency recorded"
                )
            (destination / f"{slug}.scene.yaml").write_bytes(card_path.read_bytes())
            outputs["session_replay"] = scenario
        else:
            scenario = destination / f"{slug}.yaml"
            timeline = []
            for row in inputs:
                payload = row["payload"]
                entry = {"t": round(row["t"] - start, 3)}
                if row["event_type"] == "PersonState":
                    entry["person"] = {
                        k: payload[k] for k in ("state", "zone", "confidence") if k in payload
                    }
                else:
                    entry["utterance"] = {
                        k: payload[k] for k in ("text", "confidence", "duration_s") if k in payload
                    }
                timeline.append(entry)
            question = (
                "; ".join(b.get("summary", b.get("check", "")) for b in bugs)
                or "What should the agent do at this moment?"
            )
            bench = {
                "id": slug,
                "category": card["category"],
                "summary": f"Promoted from {scene} at {at:g} s",
                "start": str(card.get("start", "02:00")),
                "timeline": timeline,
                "checkpoints": [
                    {
                        "id": "flagged-moment",
                        "window": [max(0, at - start), max(0, at - start)],
                        "question": question,
                    }
                ],
            }
            _safe(bench)
            scenario.write_text(yaml.safe_dump(bench, sort_keys=False, allow_unicode=True))
            load_scenario(scenario)
            outputs["decision_bench"] = scenario
    return outputs
