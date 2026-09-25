"""Generate model annotations for decision scenarios."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter

import yaml
from pydantic import ValidationError

from decision_bench.schema import (
    ACTION_VALUES,
    DEFAULT_PROFILE_PATH,
    GUIDELINES_PATH,
    PACKAGE_ROOT,
    SCENARIOS_DIR,
    Checkpoint,
    Scenario,
    TimelineEvent,
    guideline_clauses,
    load_default_profile,
    load_scenarios,
)
from decision_bench.triage import scenario_blocked, scenario_flags
from decision_bench.vote import Vote, vote_scenario

ANNOTATOR_PROMPT_PATH = Path(__file__).with_name("annotator_prompt.md")
ANNOTATIONS_DIR = PACKAGE_ROOT / "annotations"


class AnnotatorError(RuntimeError):
    """The isolated annotator failed or returned invalid labels."""


def citable_clauses() -> list[str]:
    """Return the human-checked guideline clause ids."""
    return sorted(clause_id for clause_id, checked in guideline_clauses().items() if checked)


def _scenario_data(scenario: Scenario) -> dict[str, object]:
    timeline = []
    for event in scenario.timeline:
        item: dict[str, object] = {"t": event.t}
        if event.person is not None:
            item["person"] = event.person.model_dump(mode="json")
        else:
            assert event.utterance is not None
            item["utterance"] = event.utterance.model_dump(mode="json")
        timeline.append(item)
    return {
        "id": scenario.id,
        "category": scenario.category,
        "summary": scenario.summary,
        "start": scenario.start.strftime("%H:%M"),
        "voice_clip": scenario.voice_clip,
        "timeline": timeline,
        "checkpoints": [
            {"id": item.id, "window": item.window, "question": item.question}
            for item in scenario.checkpoints
        ],
    }


def _yaml(value: object) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=10_000).rstrip()


def scenario_section(scenario: Scenario) -> str:
    """Return exactly the Scenario section text sent to the annotator."""
    return _yaml(_scenario_data(scenario))


def annotator_input(
    scenario: Scenario,
    profile: dict[str, object],
    guidelines_text: str,
    citable: list[str],
) -> str:
    """Build the annotator's repository-independent user prompt."""
    ids = ", ".join(checkpoint.id for checkpoint in scenario.checkpoints)
    return (
        f"## Citable clauses\n\n{', '.join(citable)}\n\n"
        f"## Guideline pack\n\n{guidelines_text.rstrip()}\n\n"
        f"## Profile\n\n{_yaml(profile)}\n\n"
        f"## Scenario\n\n{scenario_section(scenario)}\n\n"
        f"Provide the structured output for every checkpoint id listed: {ids}."
    )


def _action_schema() -> dict[str, object]:
    choices = []
    for kind, values in ACTION_VALUES.items():
        choices.append(
            {
                "type": "object",
                "properties": {kind: {"enum": sorted(values)}},
                "required": [kind],
                "additionalProperties": False,
            }
        )
    return {"oneOf": choices}


def output_schema(scenario: Scenario, citable: list[str]) -> dict[str, object]:
    """Build the strict JSON Schema requested from Claude."""
    checkpoint = {
        "type": "object",
        "properties": {
            "id": {"enum": [item.id for item in scenario.checkpoints]},
            "acceptable": {"type": "array", "items": _action_schema()},
            "must_not": {"type": "array", "items": _action_schema()},
            "escalate_by": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
            "trigger": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "null"}]},
            "rationale": {"type": "string"},
            "cites": {"type": "array", "items": {"enum": citable}},
            "uncertain": {"type": "string"},
            "needs_review": {"type": "boolean"},
        },
        "required": [
            "id",
            "acceptable",
            "must_not",
            "escalate_by",
            "trigger",
            "rationale",
            "cites",
            "uncertain",
            "needs_review",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "checkpoints": {"type": "array", "items": checkpoint},
            "scenario_notes": {"type": "string"},
        },
        "required": ["checkpoints", "scenario_notes"],
        "additionalProperties": False,
    }


def validate_annotation(
    scenario: Scenario, data: dict[str, object], citable: list[str]
) -> tuple[list[Checkpoint], list[str]]:
    """Validate structured labels and return checkpoints plus readable problems."""
    problems: list[str] = []
    raw_checkpoints = data.get("checkpoints")
    if not isinstance(raw_checkpoints, list):
        return [], ["checkpoints must be a list"]
    expected = [item.id for item in scenario.checkpoints]
    raw_ids = [item.get("id") for item in raw_checkpoints if isinstance(item, dict)]
    counts = Counter(raw_ids)
    missing = sorted(set(expected) - set(raw_ids))
    extra = sorted(set(raw_ids) - set(expected), key=str)
    duplicates = sorted((str(key) for key, count in counts.items() if count > 1))
    if missing:
        problems.append(f"missing checkpoint ids: {', '.join(missing)}")
    if extra:
        problems.append(f"unknown checkpoint ids: {', '.join(map(str, extra))}")
    if duplicates:
        problems.append(f"duplicate checkpoint ids: {', '.join(duplicates)}")
    if len(raw_ids) != len(raw_checkpoints):
        problems.append("every checkpoint must be an object with an id")

    source = {item.id: item for item in scenario.checkpoints}
    labelled: list[Checkpoint] = []
    allowed = set(citable)
    for raw in raw_checkpoints:
        if not isinstance(raw, dict) or raw.get("id") not in source:
            continue
        checkpoint_id = str(raw["id"])
        if (
            not raw.get("acceptable")
            and not raw.get("must_not")
            and not raw.get("doubtful_acceptable")
            and not raw.get("doubtful_must_not")
            and raw.get("escalate_by") is None
        ):
            problems.append(f"{checkpoint_id}: needs at least one label")
        if raw.get("trigger") is not None and raw.get("escalate_by") is None:
            problems.append(f"{checkpoint_id}: trigger requires escalate_by")
        cites = raw.get("cites", [])
        if isinstance(cites, list):
            invalid = sorted(str(cite) for cite in cites if cite not in allowed)
            if invalid:
                problems.append(
                    f"{checkpoint_id}: cites clauses that are not citable: {', '.join(invalid)}"
                )
        try:
            built = Checkpoint.model_validate(
                {
                    "id": checkpoint_id,
                    "window": source[checkpoint_id].window,
                    "acceptable": raw.get("acceptable", []),
                    "must_not": raw.get("must_not", []),
                    "doubtful_acceptable": raw.get("doubtful_acceptable", []),
                    "doubtful_must_not": raw.get("doubtful_must_not", []),
                    "escalate_by": raw.get("escalate_by"),
                    "trigger": raw.get("trigger"),
                    "threshold_source": "caregiver" if raw.get("escalate_by") is not None else None,
                    "rationale": raw.get("rationale"),
                    "cites": cites,
                }
            )
            labelled.append(built)
        except ValidationError as exc:
            problems.extend(
                f"{checkpoint_id}: {error['msg']}" for error in exc.errors(include_url=False)
            )

    if len(labelled) == len(expected) and not problems:
        ordered = {item.id: item for item in labelled}
        labelled = [ordered[item] for item in expected]
        try:
            scenario.model_copy(update={"checkpoints": tuple(labelled)}).__class__.model_validate(
                scenario.model_dump(mode="json")
                | {
                    "start": scenario.start.strftime("%H:%M"),
                    "checkpoints": [item.model_dump(mode="json") for item in labelled],
                }
            )
        except ValidationError as exc:
            problems.extend(error["msg"] for error in exc.errors(include_url=False))
    return labelled, problems


def run_claude(
    system_prompt_path: Path,
    user_prompt: str,
    schema: dict[str, object],
    model: str,
    effort: str,
) -> dict[str, object]:
    """Run Claude in an isolated empty directory and return its structured result.

    This is the Claude Code CLI on the logged-in claude.ai subscription. API-key
    variables are removed from its environment so a run can never fall back to
    paid API billing; `total_cost_usd` is Claude Code's list-price estimate,
    not a charge.
    """
    executable = shutil.which("claude")
    if executable is None:
        raise AnnotatorError("claude executable was not found on PATH")
    command = [
        executable,
        "-p",
        "--model",
        model,
        "--effort",
        effort,
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt-file",
        str(system_prompt_path.resolve()),
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema),
    ]
    try:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                command,
                input=user_prompt,
                text=True,
                capture_output=True,
                timeout=900,
                cwd=directory,
                check=False,
                env=_subscription_env(),
            )
    except subprocess.TimeoutExpired as exc:
        raise AnnotatorError("claude annotation timed out after 900 seconds") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AnnotatorError(f"claude exited with status {completed.returncode}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AnnotatorError(f"claude returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnnotatorError("claude returned JSON that is not an object")
    if payload.get("is_error"):
        raise AnnotatorError(f"claude reported an error: {payload.get('result', 'unknown error')}")
    if not isinstance(payload.get("structured_output"), dict):
        raise AnnotatorError("claude response is missing structured_output")
    usage = payload.get("modelUsage") or {}
    used_model = next(iter(usage), model) if isinstance(usage, dict) else model
    return {
        "structured_output": payload["structured_output"],
        "model": used_model,
        "session_id": payload.get("session_id"),
        "total_cost_usd": payload.get("total_cost_usd", 0),
        # Token counts for callers that log usage (scene_lab.usage).
        "usage": payload.get("usage") or {},
        "model_usage": usage if isinstance(usage, dict) else {},
        "num_turns": payload.get("num_turns"),
        "duration_ms": payload.get("duration_ms"),
    }


def _json_schema_problems(value: object, schema: dict[str, object], path: str = "$") -> list[str]:
    """Validate the small JSON Schema subset used by :func:`output_schema`."""
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path}: value is not in enum"]
    if "anyOf" in schema:
        choices = schema["anyOf"]
        if not any(not _json_schema_problems(value, choice, path) for choice in choices):
            return [f"{path}: value does not match any allowed schema"]
        return []
    if "oneOf" in schema:
        matches = sum(not _json_schema_problems(value, choice, path) for choice in schema["oneOf"])
        return [] if matches == 1 else [f"{path}: value does not match exactly one schema"]

    expected = schema.get("type")
    matches_type = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: isinstance(item, int | float) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if isinstance(expected, str) and not matches_type[expected](value):
        return [f"{path}: expected {expected}"]

    problems: list[str] = []
    if expected == "object":
        assert isinstance(value, dict)
        properties = schema.get("properties", {})
        assert isinstance(properties, dict)
        for name in schema.get("required", []):
            if name not in value:
                problems.append(f"{path}: missing required property {name}")
        if schema.get("additionalProperties") is False:
            for name in value.keys() - properties.keys():
                problems.append(f"{path}: unexpected property {name}")
        for name, item in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, dict):
                problems.extend(_json_schema_problems(item, child_schema, f"{path}.{name}"))
    elif expected == "array":
        assert isinstance(value, list)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                problems.extend(_json_schema_problems(item, item_schema, f"{path}[{index}]"))
    elif expected == "number":
        assert isinstance(value, int | float) and not isinstance(value, bool)
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: value is below minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: value is not above exclusive minimum")
    return problems


def ollama_prompt_with_schema(user_prompt: str, schema: dict[str, object]) -> str:
    """Append the JSON Schema as instructions instead of an Ollama grammar."""
    return (
        user_prompt.rstrip() + "\n\n## Required output format\n\n"
        "Return a single JSON object only, with no Markdown fence and no commentary. "
        "It must conform exactly to this JSON Schema:\n\n"
        "```json\n" + json.dumps(schema, indent=2, sort_keys=True) + "\n```"
    )


def _normalize_action(action: object) -> object:
    if isinstance(action, dict) and len(action) == 1:
        key, value = next(iter(action.items()))
        if isinstance(value, dict) and len(value) == 1:
            nested_key, nested_value = next(iter(value.items()))
            if key in ACTION_VALUES:
                if nested_key in ACTION_VALUES[key]:
                    return {key: nested_key}
                if isinstance(nested_value, str) and nested_value in ACTION_VALUES[key]:
                    return {key: nested_value}
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            return {key: value[0]}
        return dict(action)

    if isinstance(action, str):
        for key in ACTION_VALUES:
            for separator in (" ", ":"):
                prefix = key + separator
                if action.startswith(prefix) and action[len(prefix) :].strip():
                    return {key: action[len(prefix) :].strip()}
        matching_keys = [key for key, values in ACTION_VALUES.items() if action in values]
        if len(matching_keys) == 1:
            return {matching_keys[0]: action}
    return action


def normalize_ollama_output(
    value: dict[str, object],
    *,
    notes: set[str] | None = None,
    citable: list[str] | None = None,
) -> dict[str, object]:
    """Return a copy with known Ollama output-shape mistakes repaired."""
    normalized = dict(value)
    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, list):
        return normalized
    normalized_checkpoints = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            normalized_checkpoints.append(checkpoint)
            continue
        normalized_checkpoint = dict(checkpoint)
        if not {"id", "acceptable", "must_not"} <= checkpoint.keys():
            normalized_checkpoints.append(normalized_checkpoint)
            continue
        checkpoint_id = str(checkpoint["id"])
        for field in ("acceptable", "must_not"):
            actions = checkpoint.get(field)
            if isinstance(actions, list):
                normalized_actions = []
                for action in actions:
                    normalized_action = _normalize_action(action)
                    normalized_actions.append(normalized_action)
                    if notes is not None and normalized_action != action:
                        kind = next(iter(normalized_action))
                        if (
                            isinstance(action, dict)
                            and len(action) == 1
                            and isinstance(next(iter(action.values())), dict)
                        ):
                            repair = f"collapsed nested {kind} action"
                        elif isinstance(action, dict):
                            repair = f"collapsed list {kind} action"
                        else:
                            repair = f"normalized string {kind} action"
                        notes.add(f"{checkpoint_id}: {repair}")
                normalized_checkpoint[field] = normalized_actions
        defaults: dict[str, object] = {
            "escalate_by": None,
            "trigger": None,
            "uncertain": "",
            "needs_review": False,
            "cites": [],
        }
        for field, default in defaults.items():
            if field not in checkpoint:
                normalized_checkpoint[field] = default
                if notes is not None:
                    notes.add(f"{checkpoint_id}: filled missing {field}")
        cites = normalized_checkpoint.get("cites")
        if citable is not None and isinstance(cites, list):
            allowed_cites = set(citable)
            valid_cites = [cite for cite in cites if cite in allowed_cites]
            dropped_cites = [str(cite) for cite in cites if cite not in allowed_cites]
            normalized_checkpoint["cites"] = valid_cites
            if notes is not None and dropped_cites:
                noun = "citation" if len(dropped_cites) == 1 else "citations"
                notes.add(
                    f"{checkpoint_id}: dropped {len(dropped_cites)} invalid {noun} "
                    f"({', '.join(dropped_cites)})"
                )
        if normalized_checkpoint.get("escalate_by") is None:
            for field in ("trigger", "threshold_source"):
                if normalized_checkpoint.get(field) is not None:
                    normalized_checkpoint[field] = None
                    if notes is not None:
                        notes.add(f"{checkpoint_id}: dropped {field} without escalate_by")
        normalized_checkpoints.append(normalized_checkpoint)
    normalized["checkpoints"] = normalized_checkpoints
    return normalized


def _parse_ollama_content(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        first_line, separator, remainder = text.partition("\n")
        if separator and first_line.removeprefix("```").strip().lower() in ("", "json"):
            text = remainder.rstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
    start = text.find("{")
    if start < 0:
        raise ValueError("structured output does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index, character in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end < 0:
        raise ValueError("structured output contains an incomplete JSON object")
    structured = json.loads(text[start : end + 1])
    if not isinstance(structured, dict):
        raise ValueError("structured output is not an object")
    return structured


def run_ollama(
    system_prompt_path: Path,
    user_prompt: str,
    schema: dict[str, object],
    model: str,
    effort: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    think: bool = True,
) -> dict[str, object]:
    """Run an annotator through Ollama's JSON-mode chat endpoint."""
    del effort
    prompt = ollama_prompt_with_schema(user_prompt, schema)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "stream": False,
        "think": think,
        "options": {"temperature": 0, "num_ctx": 32768, "num_predict": 4096},
    }
    failures: list[str] = []
    attempt_temperatures: list[float] = []
    normalization_changed = False
    normalization_notes: set[str] = set()
    checkpoint_schema = schema.get("properties", {}).get("checkpoints", {})
    cites_schema = checkpoint_schema.get("items", {}).get("properties", {}).get("cites", {})
    citable = cites_schema.get("items", {}).get("enum")
    if not isinstance(citable, list) or not all(isinstance(item, str) for item in citable):
        citable = None
    started = perf_counter()
    for attempt in (1, 2):
        temperature = 0 if attempt == 1 else 0.3
        payload["options"]["temperature"] = temperature
        attempt_temperatures.append(temperature)
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw_response = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise AnnotatorError(f"ollama request failed: {exc}") from exc
        previous_response = raw_response.decode(errors="replace")
        try:
            response_payload = json.loads(raw_response)
            if not isinstance(response_payload, dict):
                raise ValueError("response is not an object")
            message = response_payload.get("message")
            if not isinstance(message, dict):
                raise ValueError("response is missing message")
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("response message content is not a string")
            previous_response = content
            parsed = _parse_ollama_content(content)
            structured = normalize_ollama_output(parsed, notes=normalization_notes, citable=citable)
            normalization_changed = normalization_changed or structured != parsed
            schema_problems = _json_schema_problems(structured, schema)
            if schema_problems:
                raise ValueError("; ".join(schema_problems))
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(str(exc))
            if attempt == 2:
                raise AnnotatorError(
                    "ollama returned invalid structured output after retry: " + str(exc)
                ) from exc
            payload["messages"][1]["content"] = (
                prompt
                + "\n\n## Your previous answer was rejected\n\n"
                + f"- {exc}\n\n"
                + "Previous response:\n\n"
                + previous_response
            )
            continue
        return {
            "structured_output": structured,
            "model": model,
            "backend": "ollama",
            "session_id": None,
            "total_cost_usd": 0.0,
            "runner_attempts": attempt,
            "attempt_temperatures": attempt_temperatures,
            "runtime_seconds": perf_counter() - started,
            "parse_validation_failures": failures,
            "normalization_changed": normalization_changed,
            "normalization_notes": sorted(normalization_notes),
        }
    raise AssertionError("unreachable")


_API_KEY_VARIABLES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)


def _subscription_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _API_KEY_VARIABLES}


Runner = Callable[[Path, str, dict[str, object], str, str], dict[str, object]]


def annotate_scenario(
    scenario: Scenario,
    *,
    profile: dict[str, object] | None = None,
    guidelines_text: str | None = None,
    citable: list[str] | None = None,
    model: str = "opus",
    effort: str = "high",
    runner: Runner = run_claude,
    system_prompt_path: Path = ANNOTATOR_PROMPT_PATH,
) -> dict[str, object]:
    """Annotate one scenario, retrying invalid structured output once."""
    if profile is None:
        profile = load_default_profile() | scenario.profile
    if guidelines_text is None:
        guidelines_text = GUIDELINES_PATH.read_text(encoding="utf-8")
    if citable is None:
        citable = citable_clauses()
    prompt = annotator_input(scenario, profile, guidelines_text, citable)
    schema = output_schema(scenario, citable)
    previous: dict[str, object] | None = None
    total_cost = 0.0
    total_runtime = 0.0
    runner_attempts = 0
    attempt_temperatures: list[float] = []
    parse_validation_failures: list[str] = []
    normalization_changed = False
    has_normalization_metadata = False
    normalization_notes: set[str] = set()
    for attempts in (1, 2):
        current = prompt
        if previous is not None:
            current += (
                "\n\n## Your previous answer was rejected\n\n"
                + "\n".join(f"- {problem}" for problem in previous["problems"])
                + "\n\nPrevious JSON:\n\n"
                + json.dumps(previous["output"], indent=2)
            )
        result = runner(system_prompt_path, current, schema, model, effort)
        total_cost += float(result.get("total_cost_usd", 0))
        total_runtime += float(result.get("runtime_seconds", 0))
        runner_attempts += int(result.get("runner_attempts", 1))
        attempt_temperatures.extend(result.get("attempt_temperatures", []))
        parse_validation_failures.extend(result.get("parse_validation_failures", []))
        if "normalization_changed" in result:
            has_normalization_metadata = True
            normalization_changed = normalization_changed or bool(result["normalization_changed"])
        if "normalization_notes" in result:
            normalization_notes.update(map(str, result["normalization_notes"]))
        output = result.get("structured_output")
        if not isinstance(output, dict):
            raise AnnotatorError("annotator runner returned no structured_output mapping")
        checkpoints, problems = validate_annotation(scenario, output, citable)
        if not problems:
            completed = result | {
                "attempts": attempts,
                "checkpoints": checkpoints,
                "total_cost_usd": total_cost,
                "runner_attempts": runner_attempts,
                "attempt_temperatures": attempt_temperatures,
                "runtime_seconds": total_runtime,
                "parse_validation_failures": parse_validation_failures,
            }
            if has_normalization_metadata:
                completed["normalization_changed"] = normalization_changed
                completed["normalization_notes"] = sorted(normalization_notes)
            return completed
        parse_validation_failures.extend(problems)
        previous = {"problems": problems, "output": output}
    raise AnnotatorError("annotation remained invalid after retry: " + "; ".join(problems))


class _FlowMap(dict):
    pass


class _FlowList(list):
    pass


class _Folded(str):
    pass


class _Dumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


_Dumper.add_representer(
    _FlowMap,
    lambda dumper, value: dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=True),
)
_Dumper.add_representer(
    _FlowList,
    lambda dumper, value: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", value, flow_style=True
    ),
)
_Dumper.add_representer(
    _Folded,
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:str", value, style=">"),
)


def _action_list(actions: object) -> list[_FlowMap]:
    values = actions if isinstance(actions, list | tuple) else []
    return [
        _FlowMap(
            item.model_dump(exclude_none=True)
            if isinstance(item, object) and hasattr(item, "model_dump")
            else item
        )
        for item in values
    ]


def _checkpoint_output(raw: dict[str, object], checkpoint: Checkpoint) -> dict[str, object]:
    return {
        "id": checkpoint.id,
        "acceptable": _action_list(checkpoint.acceptable),
        "must_not": _action_list(checkpoint.must_not),
        "escalate_by": checkpoint.escalate_by,
        "trigger": checkpoint.trigger,
        "rationale": _Folded(checkpoint.rationale or ""),
        "cites": _FlowList(checkpoint.cites),
        "uncertain": str(raw.get("uncertain", "")),
        "needs_review": bool(raw.get("needs_review", False)),
    }


def dump_yaml(value: object) -> str:
    return yaml.dump(value, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def _checkpoints_list(run: dict[str, object]) -> list[dict[str, object]]:
    output = run["structured_output"]
    assert isinstance(output, dict)
    raw = {item["id"]: item for item in output["checkpoints"]}
    return [_checkpoint_output(raw[item.id], item) for item in run["checkpoints"]]


def opinion_doc(run: dict[str, object]) -> dict[str, object]:
    """The `second_opinion`/`third_opinion` shape stored in the model file."""
    output = run["structured_output"]
    assert isinstance(output, dict)
    return {
        "model": run["model"],
        "attempts": run["attempts"],
        "list_price_usd": run.get("total_cost_usd", 0),
        "scenario_notes": output.get("scenario_notes", ""),
        "checkpoints": _checkpoints_list(run),
    }


def probe_doc(result: dict[str, object], second: dict[str, object] | None) -> dict[str, object]:
    """A minimal in-memory model document, just enough for `triage` to check
    whether the two runs agree, without writing anything to disk."""
    doc: dict[str, object] = {"checkpoints": _checkpoints_list(result)}
    if second is not None:
        doc["second_opinion"] = {"checkpoints": _checkpoints_list(second)}
    return doc


def status_flags(
    model_doc: dict[str, object], *, votes: dict[str, Vote] | None = None
) -> dict[str, list[str]]:
    """Checkpoints that still need a human: `triage`'s disagreements when there is
    no third run, or a vote's `no_majority` reasons once one has voted."""
    if model_doc.get("third_opinion") is not None:
        if votes is None:
            votes = vote_scenario(model_doc)
        return {
            checkpoint_id: vote.no_majority
            for checkpoint_id, vote in votes.items()
            if vote.no_majority
        }
    return scenario_flags(model_doc)


def write_annotation_files(
    scenario: Scenario,
    result: dict[str, object],
    *,
    annotations_dir: Path,
    effort: str,
    guidelines_text: str,
    force: bool,
    second: dict[str, object] | None = None,
    third: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Write the durable model draft and the review copy.

    `second` is an independent second annotator run; its labels are stored
    for triage and shown to the reviewer, never applied. `third` is a further
    independent run made only when the first two disagreed on content
    (`triage.checkpoint_flags`); when present, the review file votes the
    three runs per checkpoint instead of showing the first run's labels
    directly (see `vote.py`).
    """
    model_path = annotations_dir / "model" / f"{scenario.id}.yaml"
    review_path = annotations_dir / "review" / f"{scenario.id}.yaml"
    if model_path.exists() and not force:
        raise AnnotatorError(f"refusing to overwrite existing annotation: {model_path}")
    output = result["structured_output"]
    assert isinstance(output, dict)
    model_doc = {
        "scenario": scenario.id,
        "annotator": {
            "model": result["model"],
            "effort": effort,
            "date": date.today().isoformat(),
            "attempts": result["attempts"],
            "session_id": result.get("session_id"),
            "list_price_usd": result.get("total_cost_usd", 0),
            "prompt_sha256": hashlib.sha256(ANNOTATOR_PROMPT_PATH.read_bytes()).hexdigest(),
            "guidelines_sha256": hashlib.sha256(guidelines_text.encode()).hexdigest(),
            "scenario_sha256": hashlib.sha256(scenario_section(scenario).encode()).hexdigest(),
        },
        "scenario_notes": output.get("scenario_notes", ""),
        "checkpoints": _checkpoints_list(result),
    }
    if second is not None:
        model_doc["second_opinion"] = opinion_doc(second)
    if third is not None:
        model_doc["third_opinion"] = opinion_doc(third)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(dump_yaml(model_doc), encoding="utf-8")

    if review_path.exists() and not force:
        return model_path, review_path
    review_path.write_text(review_text(scenario, model_doc), encoding="utf-8")
    return model_path, review_path


def write_local_annotation(
    scenario: Scenario,
    result: dict[str, object],
    *,
    out_dir: Path,
    effort: str,
    guidelines_text: str,
) -> Path:
    """Write one local run in model-annotation format, without a review file."""
    output = result["structured_output"]
    assert isinstance(output, dict)
    model_doc = {
        "scenario": scenario.id,
        "annotator": {
            "model": result["model"],
            "backend": "ollama",
            "effort": effort,
            "date": date.today().isoformat(),
            "attempts": result["attempts"],
            "runner_attempts": result.get("runner_attempts", result["attempts"]),
            "attempt_temperatures": result.get("attempt_temperatures", []),
            "session_id": None,
            "list_price_usd": 0.0,
            "runtime_seconds": result.get("runtime_seconds", 0.0),
            "parse_validation_failures": result.get("parse_validation_failures", []),
            "normalization_changed": result.get("normalization_changed", False),
            "normalization_notes": result.get("normalization_notes", []),
            "prompt_sha256": hashlib.sha256(ANNOTATOR_PROMPT_PATH.read_bytes()).hexdigest(),
            "guidelines_sha256": hashlib.sha256(guidelines_text.encode()).hexdigest(),
            "scenario_sha256": hashlib.sha256(scenario_section(scenario).encode()).hexdigest(),
        },
        "scenario_notes": output.get("scenario_notes", ""),
        "checkpoints": _checkpoints_list(result),
    }
    path = out_dir / f"{scenario.id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(model_doc), encoding="utf-8")
    return path


def _clock(scenario: Scenario, seconds: float) -> str:
    start = datetime.combine(date.today(), scenario.start)
    return (start + timedelta(seconds=seconds)).strftime("%H:%M:%S")


def _event_text(event: TimelineEvent) -> str:
    if event.utterance is not None:
        return f'says "{event.utterance.text}"'
    person = event.person
    assert person is not None
    text = f"{person.state.replace('_', ' ')}, zone {person.zone}"
    if person.confidence != 0.9:
        text += f" (confidence {person.confidence:g})"
    if person.scene_note:
        text += f"; camera note: {person.scene_note}"
    return text


def _comment(text: str, indent: str = "", width: int = 92) -> list[str]:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, width - len(indent) - 2) or [""]
        lines.extend(f"{indent}# {line}".rstrip() for line in wrapped)
    return lines


_REVIEW_GUIDE = """HOW TO REVIEW
WHO WROTE WHAT: every label, rationale, doubt and note in this file was
written by the ANNOTATOR, {annotator}, working only from guidelines.md,
the profile and this scenario. The LOCAL MODEL (the agent's LLM, e.g.
gemma on Ollama) has not been run on this file; it is scored against these
labels later, once you have reviewed them.

Everything you need is here: the scripted night first, then each checkpoint
with its question, the annotator's proposed labels and its doubts. You are
judging the answer key, not the agent.

For each checkpoint, read the question and the timeline up to that moment,
then ask: do I agree with what the model says the agent should and must not
do?
  acceptable  good responses. The checkpoint passes if the agent does AT
              LEAST ONE of these inside the window. Empty = holding back is
              the right response.
  must_not    forbidden. Any of these inside the window is a critical failure.
  escalate_by the caregiver must be notified within this many seconds after
              `trigger` (a scenario second). Always a caregiver choice, never
              evidence.
  cites       guideline clauses (guidelines.md) the labels rest on. Only
              clauses you have ticked there may be cited.
Agree: leave it. Disagree: edit acceptable / must_not / escalate_by / trigger
/ cites and write a short `reason`. Rationale-only edits need no reason.
When done: set `reviewed: true` (just below this header) and tell Claude,
or run `python -m decision_bench apply <id>`.

Action values:
  phase     IDLE OBSERVING ENGAGED COOLDOWN ESCALATED
  goal      bed restroom comfort
  strategy  ambient_orient soft_greeting orient_time_place validate_and_redirect
            guided_return familiar_voice path_light escalate_phone
  notify    any info attention critical
  say       any conjunction_but avoid_terms states_clock_time invents_proper_noun
            addresses_by_name correction_of_reality memory_question
            blunt_refusal infantilising"""


def review_text(scenario: Scenario, model_doc: dict[str, object]) -> str:
    """The human review file: scenario, questions and model labels in one place.

    Everything except the labels is a YAML comment, so `apply` reads only the
    label fields and the reviewer never has to open the fixture.
    """
    rule = "# " + "=" * 78
    lines = [rule]
    annotator = model_doc.get("annotator") or {}
    annotator_name = f"Claude, {annotator.get('model', 'unknown model')}"
    lines += _comment(_REVIEW_GUIDE.format(annotator=annotator_name))
    lines += [rule, f"# SCENARIO {scenario.id} ({scenario.category})"]
    lines += _comment(scenario.summary)
    lines += ["#", f"# Night starts at {scenario.start.strftime('%H:%M')}."]
    if scenario.profile:
        lines += _comment(f"Profile changes for this scenario: {scenario.profile}")
    lines.append(f"# Consented caregiver voice clip: {'yes' if scenario.voice_clip else 'no'}")
    lines += ["#", "# TIMELINE (a camera reading holds until the next one)"]
    for event in scenario.timeline:
        lines.append(f"#   t={event.t:>5g}s  {_clock(scenario, event.t)}  {_event_text(event)}")
    notes = str(model_doc.get("scenario_notes") or "").strip()
    if notes:
        lines += ["#", f"# ANNOTATOR'S NOTES ON THE SCENARIO ITSELF ({annotator_name})"]
        lines += _comment(notes, "")

    votes = vote_scenario(model_doc) if model_doc.get("third_opinion") is not None else None
    flags = status_flags(model_doc, votes=votes)
    lines += ["#", "# TRIAGE"]
    if votes is not None:
        if flags:
            lines += _comment(
                "The two independent runs disagreed, so a third run voted. "
                f"{len(flags)} of {len(votes)} checkpoint(s) still need you: "
                f"{', '.join(flags)}. Look for NEEDS YOUR REVIEW below; the vote decided "
                "the rest. Set reviewed: true when done."
            )
        else:
            lines += _comment(
                "The two independent runs disagreed, so a third run voted. Every checkpoint "
                "reached a majority (a lone vote is recorded as doubtful, not dropped), so "
                "these labels were accepted as voted (reviewed_by: model). Edit and set "
                "reviewed_by: human to overrule."
            )
    elif flags:
        lines += _comment(
            f"{len(flags)} of {len(model_doc['checkpoints'])} checkpoint(s) need you: "
            f"{', '.join(flags)}. Look for NEEDS YOUR REVIEW below. The others were "
            "accepted as the model labelled them. Set reviewed: true when done."
        )
    else:
        lines += _comment(
            "Nothing flagged: the two runs agreed on content, so these labels were "
            "accepted as the model labelled them (reviewed_by: model). Edit and set "
            "reviewed_by: human to overrule."
        )
    reviewed = "false" if flags else "true"
    lines += [
        rule,
        "",
        f"scenario: {scenario.id}",
        f"reviewed: {reviewed}",
        f"reviewed_by: {'human' if flags else 'model'}",
        "",
        "checkpoints:",
    ]
    second_by_id = {
        str(item["id"]): item
        for item in (model_doc.get("second_opinion") or {}).get("checkpoints") or ()
    }
    third_by_id = {
        str(item["id"]): item
        for item in (model_doc.get("third_opinion") or {}).get("checkpoints") or ()
    }

    def fmt_actions(actions: object) -> str:
        return ", ".join(f"{k}: {v}" for a in actions or () for k, v in dict(a).items()) or "none"

    def fmt_run(label: str, run: dict[str, object] | None) -> str:
        if run is None:
            return f"{label} ({annotator_name}): no checkpoint"
        return (
            f"{label} ({annotator_name}): acceptable [{fmt_actions(run.get('acceptable'))}]; "
            f"must_not [{fmt_actions(run.get('must_not'))}]; escalate_by {run.get('escalate_by')} "
            f"from {run.get('trigger')}. {run.get('uncertain') or ''}"
        )

    by_id = {checkpoint.id: checkpoint for checkpoint in scenario.checkpoints}
    for item in model_doc["checkpoints"]:
        checkpoint = by_id[item["id"]]
        second = second_by_id.get(checkpoint.id)
        third = third_by_id.get(checkpoint.id)
        lines.append("")
        lines.append("  # " + "-" * 76)
        if checkpoint.window is not None:
            start, end = checkpoint.window
            lines.append(
                f"  # CHECKPOINT {checkpoint.id}: window t={start:g}-{end:g}s "
                f"({_clock(scenario, start)}-{_clock(scenario, end)})"
            )
        else:
            lines.append(f"  # CHECKPOINT {checkpoint.id}")
        if checkpoint.question:
            lines += _comment(f"QUESTION: {checkpoint.question}", "  ")
        if checkpoint.id in flags:
            lines.append("  # >>> NEEDS YOUR REVIEW <<<")
            for reason in flags[checkpoint.id]:
                lines += _comment(f"  why: {reason}", "  ")
        if votes is not None:
            if checkpoint.id in flags:
                lines += _comment(fmt_run("RUN 1, the annotator", item), "  ")
                lines += _comment(fmt_run("RUN 2, second opinion", second), "  ")
                lines += _comment(fmt_run("RUN 3, third opinion", third), "  ")
        elif second is not None and checkpoint.id in flags:
            lines += _comment(
                f"SECOND OPINION: acceptable [{fmt_actions(second.get('acceptable'))}]; must_not "
                f"[{fmt_actions(second.get('must_not'))}]; escalate_by {second.get('escalate_by')} "
                f"from {second.get('trigger')}. {second.get('uncertain') or ''}",
                "  ",
            )
        uncertain = str(item.get("uncertain") or "").strip()
        if uncertain:
            lines += _comment(f"ANNOTATOR'S DOUBTS ({annotator_name}): {uncertain}", "  ")
        if votes is not None:
            vote = votes[checkpoint.id]
            lines.append("  # Labels below: the 2-of-3 vote across all three runs. Edit freely.")
            lines.append("  # " + "-" * 76)
            review_item: dict[str, object] = {"id": item["id"]}
            if vote.acceptable:
                review_item["acceptable"] = vote.acceptable
            if vote.must_not:
                review_item["must_not"] = vote.must_not
            if vote.doubtful_acceptable:
                review_item["doubtful_acceptable"] = vote.doubtful_acceptable
            if vote.doubtful_must_not:
                review_item["doubtful_must_not"] = vote.doubtful_must_not
            review_item["escalate_by"] = vote.escalate_by
            review_item["trigger"] = vote.trigger
            review_item["rationale"] = vote.rationale
            review_item["cites"] = vote.cites
            lines.append("")
            lines.extend(_comment(f"vote: {vote.record}", "  "))
        else:
            lines.append(
                f"  # Labels below: proposed by the annotator, {annotator_name}. Edit freely."
            )
            lines.append("  # " + "-" * 76)
            review_item = {
                key: value
                for key, value in item.items()
                if key not in {"uncertain", "needs_review"}
            }
        review_item["acceptable"] = _action_list(review_item.get("acceptable"))
        review_item["must_not"] = _action_list(review_item.get("must_not"))
        if "doubtful_acceptable" in review_item:
            review_item["doubtful_acceptable"] = _action_list(review_item["doubtful_acceptable"])
        if "doubtful_must_not" in review_item:
            review_item["doubtful_must_not"] = _action_list(review_item["doubtful_must_not"])
        review_item["rationale"] = _Folded(str(review_item.get("rationale") or ""))
        review_item["cites"] = _FlowList(review_item.get("cites") or ())
        review_item["reason"] = ""
        lines.extend("  " + line for line in dump_yaml([review_item]).rstrip().splitlines())
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draft guideline-grounded scenario labels.")
    parser.add_argument("--scenario", action="append", dest="scenario_ids")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--model", default="opus")
    parser.add_argument("--backend", choices=("claude", "ollama"), default="claude")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Ollama thinking (thinking text is not stored)",
    )
    parser.add_argument("--effort", default="high")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="rebuild review files from existing model annotations without calling claude",
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument(
        "--no-second-opinion",
        action="store_true",
        help="skip the independent second annotator run used for triage",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="do not apply scenarios whose labels triage accepted without a human",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="run the Opus calls for up to N scenarios concurrently; writes stay serial",
    )
    parser.add_argument(
        "--tiebreak",
        action="store_true",
        help="run a third opinion (or apply directly) for scenarios that already have "
        "two runs and are not yet reviewed",
    )
    return parser


def _annotate_with_tiebreak(
    scenario: Scenario,
    *,
    profile: dict[str, object],
    guidelines_text: str,
    citable: list[str],
    model: str,
    effort: str,
    no_second_opinion: bool,
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    """Run the first (and, unless skipped, second and third) Opus calls for one
    scenario. Pure computation, no filesystem writes, so callers can run it in
    a thread pool and keep all writes serial."""
    result = annotate_scenario(
        scenario,
        profile=profile,
        guidelines_text=guidelines_text,
        citable=citable,
        model=model,
        effort=effort,
    )
    second = None
    third = None
    if not no_second_opinion:
        second = annotate_scenario(
            scenario,
            profile=profile,
            guidelines_text=guidelines_text,
            citable=citable,
            model=model,
            effort=effort,
        )
        doc = probe_doc(result, second)
        if scenario_flags(doc) and not scenario_blocked(doc):
            third = annotate_scenario(
                scenario,
                profile=profile,
                guidelines_text=guidelines_text,
                citable=citable,
                model=model,
                effort=effort,
            )
    return result, second, third


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.backend == "ollama":
        return _ollama_main(args)
    if args.tiebreak:
        from decision_bench.tiebreak import tiebreak_main

        return tiebreak_main(args)
    scenarios = load_scenarios(args.scenarios)
    if args.scenario_ids:
        wanted = set(args.scenario_ids)
        missing = sorted(wanted - {item.id for item in scenarios})
        if missing:
            raise SystemExit(f"unknown scenario(s): {', '.join(missing)}")
        scenarios = [item for item in scenarios if item.id in wanted]
    else:
        scenarios = [item for item in scenarios if item.noise_of is None and not item.labelled]
    profile = load_default_profile(args.profile)
    guidelines_text = GUIDELINES_PATH.read_text(encoding="utf-8")
    citable = citable_clauses()
    if args.review_only:
        for scenario in scenarios:
            model_path = args.annotations / "model" / f"{scenario.id}.yaml"
            review_path = args.annotations / "review" / f"{scenario.id}.yaml"
            if not model_path.exists():
                continue
            if review_path.exists() and not args.force:
                raise SystemExit(
                    f"refusing to overwrite review file without --force: {review_path}"
                )
            model_doc = yaml.safe_load(model_path.read_text(encoding="utf-8"))
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(review_text(scenario, model_doc), encoding="utf-8")
            print(f"{scenario.id}: {review_path}")
        return 0
    if args.dry_run:
        for scenario in scenarios:
            print(f"SYSTEM PROMPT: {ANNOTATOR_PROMPT_PATH}")
            print(annotator_input(scenario, profile | scenario.profile, guidelines_text, citable))
        return 0

    total = 0.0
    if not args.force:
        existing = [
            args.annotations / "model" / f"{scenario.id}.yaml"
            for scenario in scenarios
            if (args.annotations / "model" / f"{scenario.id}.yaml").exists()
        ]
        if existing:
            raise SystemExit(f"refusing to overwrite existing annotation: {existing[0]}")
    jobs = max(1, args.jobs)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            scenario.id: executor.submit(
                _annotate_with_tiebreak,
                scenario,
                profile=profile | scenario.profile,
                guidelines_text=guidelines_text,
                citable=citable,
                model=args.model,
                effort=args.effort,
                no_second_opinion=args.no_second_opinion,
            )
            for scenario in scenarios
        }
        for scenario in scenarios:
            try:
                result, second, third = futures[scenario.id].result()
                model_path, review_path = write_annotation_files(
                    scenario,
                    result,
                    annotations_dir=args.annotations,
                    effort=args.effort,
                    guidelines_text=guidelines_text,
                    force=args.force,
                    second=second,
                    third=third,
                )
            except AnnotatorError as exc:
                raise SystemExit(str(exc)) from exc
            cost = float(result.get("total_cost_usd", 0))
            if second is not None:
                cost += float(second.get("total_cost_usd", 0))
            if third is not None:
                cost += float(third.get("total_cost_usd", 0))
            total += cost
            flags = status_flags(yaml.safe_load(model_path.read_text(encoding="utf-8")))
            if flags:
                status = (
                    f"{len(flags)} checkpoint(s) need review: {', '.join(flags)} -> {review_path}"
                )
            elif args.no_apply:
                status = "accepted by triage, not applied (--no-apply)"
            else:
                from decision_bench.review import ReviewError, apply_scenario

                try:
                    applied, _ = apply_scenario(
                        scenario.id, annotations_dir=args.annotations, scenarios_dir=args.scenarios
                    )
                except ReviewError as exc:
                    raise SystemExit(str(exc)) from exc
                status = f"accepted by triage, {applied} checkpoint(s) applied"
            third_note = ", 3rd run voted" if third is not None else ""
            print(f"{scenario.id}: attempts={result['attempts']}{third_note} {status}")
    print(
        f"subscription usage, list-price equivalent ${total:.2f} (not billed: claude.ai login, "
        "no API key)"
    )
    return 0


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _ollama_main(args: argparse.Namespace) -> int:
    """Run one local first-pass annotation per scenario, with no review side effects."""
    if args.out_dir is None:
        raise SystemExit("--out-dir is required with --backend ollama")
    protected = (
        ANNOTATIONS_DIR / "model",
        ANNOTATIONS_DIR / "review",
        args.annotations / "model",
        args.annotations / "review",
    )
    if any(_inside(args.out_dir, path) for path in protected):
        raise SystemExit("--out-dir for Ollama must not be annotations/model or annotations/review")
    if args.review_only or args.tiebreak:
        raise SystemExit("--review-only and --tiebreak are not available with --backend ollama")

    scenarios = load_scenarios(args.scenarios)
    by_id = {scenario.id: scenario for scenario in scenarios}
    if args.scenario_ids:
        wanted = list(dict.fromkeys(args.scenario_ids))
    else:
        model_dir = args.annotations / "model"
        wanted = sorted(path.stem for path in model_dir.glob("*.yaml"))
    missing = sorted(set(wanted) - set(by_id))
    if missing:
        raise SystemExit(f"unknown scenario(s): {', '.join(missing)}")
    selected = [by_id[scenario_id] for scenario_id in wanted]
    if not args.force:
        for scenario in selected:
            path = args.out_dir / f"{scenario.id}.yaml"
            if path.exists():
                print(f"{scenario.id}: already present, skipped")
        selected = [
            scenario for scenario in selected if not (args.out_dir / f"{scenario.id}.yaml").exists()
        ]

    profile = load_default_profile(args.profile)
    guidelines_text = GUIDELINES_PATH.read_text(encoding="utf-8")
    citable = citable_clauses()
    if args.dry_run:
        for scenario in selected:
            print(f"SYSTEM PROMPT: {ANNOTATOR_PROMPT_PATH}")
            print(annotator_input(scenario, profile | scenario.profile, guidelines_text, citable))
        return 0

    runner: Runner = functools.partial(run_ollama, base_url=args.ollama_url, think=args.think)
    jobs = max(1, args.jobs)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            scenario.id: executor.submit(
                annotate_scenario,
                scenario,
                profile=profile | scenario.profile,
                guidelines_text=guidelines_text,
                citable=citable,
                model=args.model,
                effort=args.effort,
                runner=runner,
            )
            for scenario in selected
        }
        for scenario in selected:
            try:
                result = futures[scenario.id].result()
            except AnnotatorError as exc:
                raise SystemExit(f"{scenario.id}: {exc}") from exc
            path = write_local_annotation(
                scenario,
                result,
                out_dir=args.out_dir,
                effort=args.effort,
                guidelines_text=guidelines_text,
            )
            print(
                f"{scenario.id}: {path} ({float(result.get('runtime_seconds', 0)):.1f}s, "
                f"{len(result.get('parse_validation_failures', []))} parse/validation failures)"
            )
    return 0
