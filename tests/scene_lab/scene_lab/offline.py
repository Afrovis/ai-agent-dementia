"""Shared invariant reporting for the two in-process offline benches."""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .bugs import RunDir, harness_error, results_to_entries
from .invariants import check_trace
from .thresholds import load
from .trace import playbacks


class InvariantRun:
    """Collect several offline scenes in one durable scene_lab run."""

    def __init__(
        self,
        kind: str,
        model: str,
        thresholds_path: str | Path | None = None,
        root: str | Path | None = None,
        tt2_judge=None,
    ) -> None:
        if kind not in {"decision_bench", "session_replay"}:
            raise ValueError(f"invalid offline run kind: {kind}")
        self.kind = kind
        self.model = model
        self.thresholds_path = str(thresholds_path) if thresholds_path else "default"
        self.thresholds = load(thresholds_path)
        self.run = RunDir(kind, root=root, thresholds=self.thresholds)
        self.judge = tt2_judge
        self.commit = self._commit()
        self.counts: dict[str, Counter] = defaultdict(Counter)
        self.scenes = 0
        self.estimated = False
        self.llm_latency = "none"

    @staticmethod
    def _commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    def add(self, scene_id: str, trace, extra: dict | None = None) -> None:
        self.llm_latency = trace.meta.get("llm_latency", "none")
        results = check_trace(trace, self.thresholds, judge=self.judge)
        self.run.write_scene(
            scene_id,
            trace,
            results,
            {
                "commit": self.commit,
                "model": self.model,
                "llm_latency": self.llm_latency,
                **(extra or {}),
            },
        )
        self.run.append(results_to_entries(results, self.run.id, scene_id, self.commit, self.model))
        self.scenes += 1
        self.estimated |= any(item.estimated for item in playbacks(trace, self.thresholds))
        for result in results:
            self.counts[result.id]["evaluated"] += 1
            if not result.passed or result.severity == "review":
                self.counts[result.id][result.severity] += 1

    def add_error(self, scene_id: str, message: str) -> None:
        self.run.append([harness_error(self.run.id, scene_id, message)])
        self.counts["HARNESS"]["evaluated"] += 1
        self.counts["HARNESS"]["major"] += 1

    def close(self, hours: float | None = None) -> tuple[dict, str]:
        self.run.finish(self.kind, self.commit, self.model, hours or 0.0, self.scenes)
        counts = {check: dict(value) for check, value in sorted(self.counts.items())}
        summary = {
            "run_path": str(self.run.path),
            "checks": counts,
            "thresholds": self.thresholds.model_dump(),
            "thresholds_path": self.thresholds_path,
            "playback_estimated": self.estimated,
            "scene_count": self.scenes,
            "llm_latency": self.llm_latency,
        }
        lines = [
            f"Invariants run: {self.run.path}",
            f"Thresholds: {self.thresholds_path}; playback estimated: {self.estimated}",
            f"LLM latency: {self.llm_latency}",
        ]
        for check, count in counts.items():
            failures = (
                ", ".join(
                    f"{severity}={count[severity]}"
                    for severity in ("critical", "major", "minor", "review", "info")
                    if count.get(severity)
                )
                or "none"
            )
            lines.append(f"{check}: evaluated={count['evaluated']}, failed by severity: {failures}")
        return summary, "\n".join(lines)
