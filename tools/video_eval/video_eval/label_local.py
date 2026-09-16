"""Private per-frame labelling through a local VLM, either Ollama's HTTP API
or mlx-vlm running natively on Apple Silicon."""

from __future__ import annotations

import base64
import json
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from capture.motion import frame_signature, motion_score

from video_eval.common import matching_meta, read_jsonl, update_index, write_jsonl, write_meta
from video_eval.labels import (
    LABEL_PROMPT,
    LABEL_SCHEMA,
    failed_label_record,
    label_record,
    validate_label,
)
from video_eval.paths import EvalPaths

DEFAULT_MODEL = "qwen3-vl:8b"
FAST_MODEL = "gemma4:e4b-mlx"
# 8-bit rather than 4-bit: measured more accurate at native-MLX speed, and
# labelling is async so the extra latency over 4-bit doesn't matter.
DEFAULT_MLX_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-8bit"


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
                "format": LABEL_SCHEMA,
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


def _mlx_load(model_id: str):
    """Load an mlx-vlm model plus everything needed to run one JSON-schema
    constrained generation per frame. Imported lazily so mlx-vlm stays an
    optional dependency for hosts that only ever use the Ollama backend."""
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.structured import build_json_schema_logits_processor
    from mlx_vlm.utils import load_config

    model, processor = load(model_id)
    config = load_config(model_id)
    prompt = apply_chat_template(processor, config, LABEL_PROMPT, num_images=1)
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]
    schema_processor = build_json_schema_logits_processor(tokenizer, LABEL_SCHEMA)
    return model, processor, prompt, eos_token_id, schema_processor


def _mlx_generate(model, processor, prompt: str, image_path: str, logits_processor) -> str:
    from mlx_vlm import generate

    result = generate(
        model,
        processor,
        prompt,
        [image_path],
        max_tokens=200,
        temperature=0.0,
        verbose=False,
        logits_processors=[logits_processor],
    )
    return result.text if hasattr(result, "text") else str(result)


class _StoppingJsonProcessor:
    """Wraps mlx-vlm's llguidance JSON-schema logits processor. Once the
    grammar is satisfied, llguidance raises if generation is pushed past the
    closing brace instead of ending cleanly; catch that and force EOS so a
    valid response never turns into a crashed call."""

    def __init__(self, inner, eos_token_id: int) -> None:
        self._inner = inner
        self._eos_token_id = eos_token_id
        self._done = False

    def __call__(self, input_ids, logits):
        if self._done:
            return self._forced_eos(logits)
        try:
            return self._inner(input_ids, logits)
        except ValueError:
            self._done = True
            return self._forced_eos(logits)

    def _forced_eos(self, logits):
        import mlx.core as mx
        import numpy as np

        arr = np.full(tuple(logits.shape), -1e9, dtype=np.float32)
        arr[..., self._eos_token_id] = 0.0
        return mx.array(arr, dtype=logits.dtype)


class MlxLabeller:
    """Labelling through mlx-vlm running natively on Apple Silicon, instead
    of through Ollama's HTTP API. Measured roughly 6x faster per frame than
    the same model size through Ollama, with no request timeouts (see
    docs/VIDEO_EVAL.md)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MLX_MODEL,
        loader: Callable[[str], tuple] = _mlx_load,
        generator: Callable[..., str] = _mlx_generate,
    ) -> None:
        self.model = model
        self._generate = generator
        (
            self._model,
            self._processor,
            self._prompt,
            self._eos_token_id,
            self._schema_processor,
        ) = loader(model)

    def label(self, image: bytes) -> Any:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
            handle.write(image)
            handle.flush()
            logits_processor = _StoppingJsonProcessor(
                self._schema_processor.clone(), self._eos_token_id
            )
            text = self._generate(
                self._model, self._processor, self._prompt, handle.name, logits_processor
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("mlx-vlm returned invalid JSON content") from exc

    def close(self) -> None:
        pass


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
    backend: str = "ollama",
    model: str | None = None,
    ollama_url: str = "http://localhost:11434",
    adaptive: bool = True,
    motion_threshold: float = 0.02,
    force: bool = False,
    labeller_factory: Callable[[], LocalLabeller] | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    if model is None:
        model = DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_MODEL
    paths = EvalPaths.for_clip(clip_id, root)
    output = paths.labels / "local.jsonl"
    meta_path = paths.labels / "local.meta.json"
    parameters = {
        "adaptive": adaptive,
        "backend": backend,
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
    if labeller_factory:
        labeller = labeller_factory()
    elif backend == "mlx":
        labeller = MlxLabeller(model=model)
    elif backend == "ollama":
        labeller = OllamaLabeller(base_url=ollama_url, model=model)
    else:
        raise ValueError(f"unknown backend: {backend!r}")
    labeller_tag = f"{backend}:{model}"
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
                        labeller=labeller_tag,
                        reason=error or "invalid",
                    )
                else:
                    record = label_record(
                        label,
                        frame_index=int(frame["frame_index"]),
                        t_s=float(frame["t_s"]),
                        labeller=labeller_tag,
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
        versions=("pillow", backend),
    )
    update_index(paths.root, clip_id, "label_local", "complete")
    return {"status": "complete", "frames": len(records), "model": model, "model_calls": calls}
