"""Private per-frame labelling through the host's local Ollama VLM."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from capture.motion import frame_signature, motion_score

from video_eval.common import matching_meta, read_jsonl, update_index, write_jsonl, write_meta
from video_eval.labels import LABEL_PROMPT, failed_label_record, label_record, validate_label
from video_eval.paths import EvalPaths

DEFAULT_MODEL = "qwen3-vl:8b"
FAST_MODEL = "gemma4:e4b-mlx"


class LocalLabeller(Protocol):
    def label(self, image: bytes) -> Any: ...


class OllamaLabeller:
    def __init__(self, *, base_url: str, model: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    def label(self, image: bytes) -> Any:
        response = self._post(
            "/api/chat",
            {
                "format": "json",
                "messages": [
                    {
                        "content": LABEL_PROMPT,
                        "images": [base64.b64encode(image).decode("ascii")],
                        "role": "user",
                    }
                ],
                "model": self.model,
                "options": {"temperature": 0},
                "stream": False,
            },
        )
        try:
            return json.loads(response["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Ollama returned invalid JSON content") from exc

    def close(self) -> None:
        # Explicitly return scarce unified memory to the Docker stack.
        self._post("/api/generate", {"keep_alive": 0, "model": self.model})


def adaptive_indices(frames: list[dict[str, Any]], root: Path, threshold: float) -> set[int]:
    """Select all moving frames and one in ten consecutive still frames."""
    selected: set[int] = set()
    previous: tuple[int, ...] | None = None
    still_count = 0
    for position, frame in enumerate(frames):
        signature = frame_signature((root / frame["review_path"]).read_bytes())
        moving = previous is None or motion_score(signature, previous) >= threshold
        if moving:
            selected.add(position)
            still_count = 0
        else:
            still_count += 1
            if still_count % 10 == 0:
                selected.add(position)
        previous = signature
    return selected


def _call_with_retry(labeller: LocalLabeller, jpeg: bytes):
    last_error = "invalid model response"
    for attempt in range(1, 3):
        try:
            return validate_label(labeller.label(jpeg)), None, attempt
        except (ValueError, RuntimeError) as exc:
            last_error = str(exc)
    return None, last_error, 2


def label_local(
    clip_id: str,
    *,
    root: Path | None = None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = "http://localhost:11434",
    adaptive: bool = True,
    motion_threshold: float = 0.02,
    force: bool = False,
    labeller_factory: Callable[[], LocalLabeller] | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    paths = EvalPaths.for_clip(clip_id, root)
    output = paths.labels / "local.jsonl"
    meta_path = paths.labels / "local.meta.json"
    parameters = {
        "adaptive": adaptive,
        "clip_id": clip_id,
        "model": model,
        "motion_threshold": motion_threshold,
        "ollama_url": ollama_url,
    }
    if not force and output.exists() and matching_meta(meta_path, parameters):
        return {"status": "skipped", "frames": len(read_jsonl(output)), "model": model}
    frames = read_jsonl(paths.frames)
    selected = (
        adaptive_indices(frames, paths.root, motion_threshold)
        if adaptive
        else set(range(len(frames)))
    )
    labeller = (
        labeller_factory() if labeller_factory else OllamaLabeller(base_url=ollama_url, model=model)
    )
    records: list[dict[str, Any]] = []
    last_record: dict[str, Any] | None = None
    calls = 0
    try:
        for position, frame in enumerate(frames):
            if position not in selected and last_record is not None:
                record = dict(last_record)
                record.update(
                    {
                        "frame_index": int(frame["frame_index"]),
                        "t_s": float(frame["t_s"]),
                        "propagated_from": last_record["frame_index"],
                    }
                )
            else:
                label, error, attempts = _call_with_retry(
                    labeller, (paths.root / frame["review_path"]).read_bytes()
                )
                calls += attempts
                if label is None:
                    record = failed_label_record(
                        frame_index=int(frame["frame_index"]),
                        t_s=float(frame["t_s"]),
                        labeller=f"ollama:{model}",
                        reason=error or "invalid",
                    )
                else:
                    record = label_record(
                        label,
                        frame_index=int(frame["frame_index"]),
                        t_s=float(frame["t_s"]),
                        labeller=f"ollama:{model}",
                    )
                last_record = record
            records.append(record)
            # A long clip takes hours; keep valid progress recoverable after every call.
            if position in selected:
                write_jsonl(output, records)
    finally:
        close = getattr(labeller, "close", None)
        if close is not None:
            close()
    write_jsonl(output, records)
    write_meta(
        meta_path,
        command="label-local",
        parameters=parameters,
        started_at=started,
        versions=("pillow", "ollama"),
    )
    update_index(paths.root, clip_id, "label_local", "complete")
    return {"status": "complete", "frames": len(records), "model": model, "model_calls": calls}
