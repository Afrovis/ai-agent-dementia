"""Append-only issue records and readable run reports."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .invariants import InvariantResult
from .thresholds import Thresholds, load
from .trace import Trace, playbacks

RANK = {"critical": 0, "major": 1, "minor": 2, "review": 3, "info": 4}
DESCRIPTIONS = {
    "TT-1": "Reply to each final utterance within five seconds.",
    "TT-2": "Review whether a question or request was answered or redirected.",
    "TT-3": "Do not start playback over the person's speech.",
    "TT-4": "Stop interruptible playback promptly on barge-in.",
    "TT-5": "Keep eight seconds between unprompted playbacks.",
    "TT-6": "Do not speak into an empty room.",
    "TT-7": "Do not publish a stale reply.",
    "SM-1": "Stay silent once the person is settled in bed.",
    "SM-2": "Escalate only with a justified cause.",
    "SM-3": "Keep an engaged session responsive and end it cleanly.",
    "SM-4": "Turn the hallway light off after the person returns to bed.",
    "SM-5": "Do not publish an action forbidden by a veto rule.",
    "TM-1": "Keep input-to-effect loop lag within three seconds.",
    "TM-2": "Measure reply latency from utterance end to playback start.",
    "TM-3": "Measure bathroom-path reading to goal change.",
    "WORD-conjunction_but": "Avoid wording that negates reassurance with 'but'.",
    "WORD-avoid_terms": "Avoid terms listed in the person's profile.",
    "WORD-states_clock_time": "Do not state an unsupported clock time.",
    "WORD-invents_proper_noun": "Do not invent a person or place name.",
    "HARNESS": "Scene lab infrastructure or director failed.",
}


def decode(key: str) -> dict[str, str]:
    parts = (key.split("|") + ["-"] * 5)[:5]
    return dict(zip(("check", "phase", "goal", "strategy", "drop reason"), parts))


def _context_lines(key: str, items: list[dict]) -> list[str]:
    context = decode(key)
    first = items[0]
    severity = min((i["severity"] for i in items), key=lambda s: RANK.get(s, 99))
    return [
        f"Severity: {severity}; count: {len(items)}",
        f"Check {context['check']}: {DESCRIPTIONS.get(context['check'], 'See invariant report.')}",
        "Context: "
        + "; ".join(
            f"{name}: {context[name]}" for name in ("phase", "goal", "strategy", "drop reason")
        ),
        f"First: {first['summary']}",
        *[f"Evidence: {line}" for line in first.get("evidence", [])],
        "Occurrences: "
        + ", ".join(f"[{i['run']}/{i['scene']} at {i['t']:g}s]({i['report']})" for i in items),
    ]


class BugEntry(BaseModel):
    run: str
    scene: str
    t: float
    check: str
    severity: Literal["critical", "major", "minor", "review", "info"]
    origin: Literal["agent", "harness"]
    fingerprint: str
    summary: str
    evidence: list[str]
    report: str
    commit: str
    model: str


def fingerprint(result: InvariantResult) -> str:
    fields = ("phase", "goal", "strategy", "drop_reason")
    return "|".join(
        [result.id]
        + [
            str(result.context[key]) if result.context.get(key) is not None else "-"
            for key in fields
        ]
    )


def results_to_entries(
    results: list[InvariantResult], run: str, scene: str, commit: str, model: str
) -> list[BugEntry]:
    return [
        BugEntry(
            run=run,
            scene=scene,
            t=r.t,
            check=r.id,
            severity=r.severity,
            origin="agent",
            fingerprint=fingerprint(r),
            summary=r.reason,
            evidence=r.evidence,
            report=f"{scene}/report.md#t={r.t:g}",
            commit=commit,
            model=model,
        )
        for r in results
        if not r.passed or r.severity == "review"
    ]


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def harness_error(run: str, scene: str, message: str) -> BugEntry:
    # CLI errors (claude, docker) arrive with terminal colour codes.
    message = _ANSI.sub("", message)
    return BugEntry(
        run=run,
        scene=scene,
        t=0,
        check="HARNESS",
        severity="major",
        origin="harness",
        fingerprint="HARNESS|-|-|-|-",
        summary=message,
        evidence=[message],
        report=f"{scene}/report.md#t=0",
        commit="-",
        model="-",
    )


def _default_root() -> Path:
    if os.environ.get("SCENE_LAB_RUNS"):
        return Path(os.environ["SCENE_LAB_RUNS"])
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        repo = Path(common).resolve().parent
    except (OSError, subprocess.CalledProcessError):
        repo = Path.cwd()
    return repo.parent / "data-ai-agent-dementia/analysis/scene-lab/runs"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class RunDir:
    def __init__(
        self,
        kind: str,
        root: str | Path | None = None,
        *,
        path: str | Path | None = None,
        thresholds: Thresholds | None = None,
    ):
        self.root = Path(root) if root else _default_root()
        self.thresholds = thresholds or load()
        if path:
            self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            base = datetime.now().strftime("%Y-%m-%dT%H%M") + f"-{kind}"
            self.path = self.root / base
            suffix = 2
            while True:
                try:
                    self.path.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    self.path = self.root / f"{base}-{suffix}"
                    suffix += 1
        self.id = self.path.name

    def append(self, entries: list[BugEntry]) -> None:
        with (self.path / "bugs.jsonl").open("a", encoding="utf-8") as out:
            for entry in entries:
                out.write(entry.model_dump_json() + "\n")
            out.flush()
            os.fsync(out.fileno())
        self.render_bugs_md()

    def write_scene(
        self, scene_id: str, trace: Trace, results: list[InvariantResult], extra: dict | None = None
    ) -> Path:
        scene = self.path / scene_id
        scene.mkdir(parents=True, exist_ok=True)
        trace.write_jsonl(scene / "trace.jsonl")
        extra = extra or {}
        report = {
            "results": [r.model_dump(mode="json") for r in results],
            "thresholds": self.thresholds.model_dump(),
            "commit": extra.get("commit", "-"),
            "model": extra.get("model", "-"),
            "playback_estimated": any(p.estimated for p in playbacks(trace, self.thresholds)),
            **extra,
        }
        (scene / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            f"# {scene_id}",
            "",
            f"Thresholds: `{json.dumps(report['thresholds'], sort_keys=True)}`",
            f"Playback estimated: {report['playback_estimated']}",
            "",
            "## Timeline",
            "",
        ]
        for event in trace.events:
            key = {
                key: event.data[key]
                for key in (
                    "state",
                    "zone",
                    "phase",
                    "goal",
                    "strategy",
                    "text",
                    "level",
                    "kind",
                    "detail",
                )
                if key in event.data
            }
            lines.append(
                f'<a id="t={event.t:g}"></a> {event.t:g}s {event.type} '
                + json.dumps(key, ensure_ascii=False)
            )
        lines.extend(["", "## Results", ""])
        for result in results:
            lines.append(
                f'<a id="t={result.t:g}"></a> {result.id} {result.severity}: '
                + result.reason
                + (" (passed)" if result.passed else " (failed)")
            )
            lines.extend(f"  - {item}" for item in result.evidence)
        (scene / "report.md").write_text("\n".join(lines) + "\n")
        return scene

    def render_bugs_md(self) -> Path:
        grouped = defaultdict(list)
        for item in _read(self.path / "bugs.jsonl"):
            grouped[item["fingerprint"]].append(item)
        meta = getattr(self, "metadata", {})
        counts = Counter(item["severity"] for items in grouped.values() for item in items)
        lines = [
            f"# Bugs: {self.id}",
            "",
            f"Run: {self.id}; kind: {meta.get('kind', '-')}; "
            f"commit: {meta.get('commit', '-')}; model: {meta.get('model', '-')}; "
            f"contention: {meta.get('contention', '-')}; "
            f"scenes run: {meta.get('scene_count', '-')}",
            "Counts: " + ", ".join(f"{severity}: {counts[severity]}" for severity in RANK),
            "",
        ]
        for key, items in sorted(
            grouped.items(),
            key=lambda pair: (
                min(RANK.get(i["severity"], 99) for i in pair[1]),
                -len(pair[1]),
                pair[0],
            ),
        ):
            lines.extend([f"## {key} ({len(items)})", ""])
            lines.extend(f"- {line}" for line in _context_lines(key, items))
            lines.append("")
        path = self.path / "bugs.md"
        path.write_text("\n".join(lines))
        return path

    def finish(self, kind: str, commit: str, model: str, hours: float, scene_count: int) -> None:
        self.metadata = {
            **getattr(self, "metadata", {}),
            "kind": kind,
            "commit": commit,
            "model": model,
            "hours": hours,
            "scene_count": scene_count,
        }
        self.render_bugs_md()
        counts = Counter(item["severity"] for item in _read(self.path / "bugs.jsonl"))
        record = {
            "id": self.id,
            "kind": kind,
            "commit": commit,
            "model": model,
            "hours": hours,
            "scene_count": scene_count,
            "bug_counts": {severity: counts[severity] for severity in RANK},
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "index.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
            out.flush()
            os.fsync(out.fileno())


def merge(run_dirs: list[str | Path]) -> list[dict]:
    paths = sorted([Path(path) for path in run_dirs], key=lambda path: path.name)
    if not paths:
        return []
    entries = {path.name: _read(path / "bugs.jsonl") for path in paths}
    counts = {name: Counter(item["fingerprint"] for item in rows) for name, rows in entries.items()}
    first, last = counts[paths[0].name], counts[paths[-1].name]
    keys = set().union(*(set(count) for count in counts.values()))
    return [
        {
            "fingerprint": key,
            "counts": {name: count[key] for name, count in counts.items()},
            "occurrences": [
                {**item, "report": str(path / item["report"])}
                for path in paths
                for item in entries[path.name]
                if item["fingerprint"] == key
            ],
            "status": "persisting"
            if key in first and key in last
            else "gone"
            if key not in last
            else "new",
        }
        for key in sorted(keys)
    ]


def render_merge_md(rows: list[dict]) -> str:
    lines = ["# Scene lab bug comparison", ""]
    for row in rows:
        counts = ", ".join(f"{run}: {count}" for run, count in row["counts"].items())
        lines.extend([f"## {row['fingerprint']} — {row['status']}", "", f"Counts: {counts}"])
        lines.extend(f"- {line}" for line in _context_lines(row["fingerprint"], row["occurrences"]))
        lines.append("")
    return "\n".join(lines) + "\n"
