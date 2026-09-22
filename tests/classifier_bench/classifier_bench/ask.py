"""Ask a candidate classifier one action question at a time."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

from decision_bench.annotate import (
    AnnotatorError,
    normalize_ollama_output,
    ollama_prompt_with_schema,
)

PROBE_PROMPT_PATH = Path(__file__).with_name("probe_prompt.md")
LABELS = ("acceptable", "forbidden", "irrelevant")
OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "label": {"enum": list(LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["label", "confidence"],
    "additionalProperties": False,
}

# A runner performs one attempt. ask_questions owns retry policy so tests and future
# backends can inject the same small interface.
Runner = Callable[[Path, str, dict[str, object], str, float], dict[str, object]]


def _parse_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        _, _, stripped = stripped.partition("\n")
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def _validate_answer(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("answer is not an object")
    if set(value) != {"label", "confidence"}:
        raise ValueError("answer must contain only label and confidence")
    label = value["label"]
    confidence = value["confidence"]
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise ValueError("confidence must be a number")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return {"label": label, "confidence": float(confidence)}


def run_ollama(
    system_prompt_path: Path,
    user_prompt: str,
    schema: dict[str, object],
    model: str,
    temperature: float,
    *,
    base_url: str = "http://127.0.0.1:11434",
) -> dict[str, object]:
    """Perform one Ollama JSON-mode call for a classification attempt."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": ollama_prompt_with_schema(user_prompt, schema)},
        ],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 32768,
            "num_predict": 256,
        },
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw_response = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AnnotatorError(f"ollama request failed: {exc}") from exc
    try:
        response_payload = json.loads(raw_response)
        message = response_payload["message"]
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("response message content is not a string")
        parsed = _parse_object(content)
        # Shared normalization is intentionally applied before validation. For this small
        # schema it is normally a no-op, while keeping Ollama handling aligned with annotate.
        structured = normalize_ollama_output(parsed)
        return _validate_answer(structured)
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise AnnotatorError(f"ollama returned invalid structured output: {exc}") from exc


def run_stub(
    system_prompt_path: Path,
    user_prompt: str,
    schema: dict[str, object],
    model: str,
    temperature: float,
) -> dict[str, object]:
    """Return a stable local answer for smoke tests."""
    del system_prompt_path, schema, model, temperature
    digest = hashlib.sha256(user_prompt.encode()).digest()
    return {"label": LABELS[digest[0] % len(LABELS)], "confidence": 0.5}


def user_prompt(question: Mapping[str, object], guidelines_text: str) -> str:
    """Build the prompt containing exactly one candidate action."""
    return (
        f"## Guidelines\n\n{guidelines_text.rstrip()}\n\n"
        f"## Serialized state\n\n{question['state']}\n\n"
        f"## Checkpoint question\n\n{question['question']}\n\n"
        "## One candidate action\n\n"
        f"{json.dumps(question['action'], ensure_ascii=False, sort_keys=True)}"
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def ask_questions(
    questions: list[dict[str, object]],
    output_path: Path,
    *,
    guidelines_text: str,
    runner: Runner,
    model: str = "gemma4:e4b-mlx",
    selected_set: str = "all",
    limit: int | None = None,
    force: bool = False,
    system_prompt_path: Path = PROBE_PROMPT_PATH,
) -> list[dict[str, object]]:
    """Ask selected questions, recording failures and continuing the run."""
    candidates = [
        item for item in questions if selected_set == "all" or item.get("set") == selected_set
    ]
    if limit is not None:
        candidates = candidates[:limit]
    existing = read_jsonl(output_path)
    candidate_qids = {str(item["qid"]) for item in candidates}
    if force and output_path.exists():
        retained = [item for item in existing if str(item.get("qid")) not in candidate_qids]
        with output_path.open("w", encoding="utf-8") as handle:
            for item in retained:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        existing = retained
    completed = {str(item.get("qid")) for item in existing}
    pending = [item for item in candidates if str(item["qid"]) not in completed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, object]] = []
    with output_path.open("a", encoding="utf-8") as handle:
        for question in pending:
            prompt = user_prompt(question, guidelines_text)
            started = perf_counter()
            answer: dict[str, object] | None = None
            error = "unknown failure"
            attempts = 0
            for attempts, temperature in enumerate((0.0, 0.3), 1):
                try:
                    answer = _validate_answer(
                        runner(system_prompt_path, prompt, OUTPUT_SCHEMA, model, temperature)
                    )
                    break
                except Exception as exc:  # one bad question must never abort a run
                    error = f"{type(exc).__name__}: {exc}"
            record = dict(question)
            if answer is not None:
                record.update(answer)
            else:
                record["error"] = error
            record["latency_seconds"] = perf_counter() - started
            record["attempts"] = attempts
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            written.append(record)
    return written
