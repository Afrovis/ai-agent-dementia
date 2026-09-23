"""Append-only issue records and readable run reports."""

from __future__ import annotations

import json
import os
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


def harness_error(run: str, scene: str, message: str) -> BugEntry:
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
        lines = [f"# Bugs: {self.id}", ""]
        for key, items in sorted(
            grouped.items(),
            key=lambda pair: (
                min(RANK.get(i["severity"], 99) for i in pair[1]),
                -len(pair[1]),
                pair[0],
            ),
        ):
            lines.extend([f"## {key} ({len(items)})", ""])
            for item in items:
                lines.append(
                    f"- {item['severity']}: {item['summary']} — "
                    + f"[{item['scene']} at {item['t']:g}s]({item['report']})"
                )
            lines.append("")
        path = self.path / "bugs.md"
        path.write_text("\n".join(lines))
        return path

    def finish(self, kind: str, commit: str, model: str, hours: float, scene_count: int) -> None:
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
    counts = {
        path.name: Counter(item["fingerprint"] for item in _read(path / "bugs.jsonl"))
        for path in paths
    }
    first, last = counts[paths[0].name], counts[paths[-1].name]
    keys = set().union(*(set(count) for count in counts.values()))
    return [
        {
            "fingerprint": key,
            "counts": {name: count[key] for name, count in counts.items()},
            "status": "persisting"
            if key in first and key in last
            else "gone"
            if key in first
            else "new",
        }
        for key in sorted(keys)
    ]


def render_merge_md(rows: list[dict]) -> str:
    lines = [
        "# Scene lab bug comparison",
        "",
        "| Fingerprint | Status | Counts by run |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        counts = ", ".join(f"{run}: {count}" for run, count in row["counts"].items())
        lines.append(f"| {row['fingerprint']} | {row['status']} | {counts} |")
    return "\n".join(lines) + "\n"
