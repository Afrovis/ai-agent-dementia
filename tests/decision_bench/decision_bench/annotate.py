"""Generate model annotations for decision scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

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
        if not raw.get("acceptable") and not raw.get("must_not") and raw.get("escalate_by") is None:
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
    """Run Claude in an isolated empty directory and return its structured result."""
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
    }


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
        output = result.get("structured_output")
        if not isinstance(output, dict):
            raise AnnotatorError("annotator runner returned no structured_output mapping")
        checkpoints, problems = validate_annotation(scenario, output, citable)
        if not problems:
            return result | {
                "attempts": attempts,
                "checkpoints": checkpoints,
                "total_cost_usd": total_cost,
            }
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
    }


def _dump(value: object) -> str:
    return yaml.dump(value, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def write_annotation_files(
    scenario: Scenario,
    result: dict[str, object],
    *,
    annotations_dir: Path,
    effort: str,
    guidelines_text: str,
    force: bool,
) -> tuple[Path, Path]:
    """Write the durable model draft and human review copy."""
    model_path = annotations_dir / "model" / f"{scenario.id}.yaml"
    review_path = annotations_dir / "review" / f"{scenario.id}.yaml"
    if model_path.exists() and not force:
        raise AnnotatorError(f"refusing to overwrite existing annotation: {model_path}")
    output = result["structured_output"]
    assert isinstance(output, dict)
    raw_by_id = {item["id"]: item for item in output["checkpoints"]}
    checkpoints = result["checkpoints"]
    model_doc = {
        "scenario": scenario.id,
        "annotator": {
            "model": result["model"],
            "effort": effort,
            "date": date.today().isoformat(),
            "attempts": result["attempts"],
            "session_id": result.get("session_id"),
            "cost_usd": result.get("total_cost_usd", 0),
            "prompt_sha256": hashlib.sha256(ANNOTATOR_PROMPT_PATH.read_bytes()).hexdigest(),
            "guidelines_sha256": hashlib.sha256(guidelines_text.encode()).hexdigest(),
            "scenario_sha256": hashlib.sha256(scenario_section(scenario).encode()).hexdigest(),
        },
        "scenario_notes": output.get("scenario_notes", ""),
        "checkpoints": [_checkpoint_output(raw_by_id[item.id], item) for item in checkpoints],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(_dump(model_doc), encoding="utf-8")

    if review_path.exists() and not force:
        return model_path, review_path
    review_path.write_text(review_text(scenario, model_doc), encoding="utf-8")
    return model_path, review_path


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
    lines += [rule, "", f"scenario: {scenario.id}", "reviewed: false", "", "checkpoints:"]

    by_id = {checkpoint.id: checkpoint for checkpoint in scenario.checkpoints}
    for item in model_doc["checkpoints"]:
        checkpoint = by_id[item["id"]]
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
        uncertain = str(item.get("uncertain") or "").strip()
        if uncertain:
            lines += _comment(f"ANNOTATOR'S DOUBTS ({annotator_name}): {uncertain}", "  ")
        lines.append(f"  # Labels below: proposed by the annotator, {annotator_name}. Edit freely.")
        lines.append("  # " + "-" * 76)
        review_item = {key: value for key, value in item.items() if key != "uncertain"}
        review_item["acceptable"] = _action_list(review_item.get("acceptable"))
        review_item["must_not"] = _action_list(review_item.get("must_not"))
        review_item["rationale"] = _Folded(str(review_item.get("rationale") or ""))
        review_item["cites"] = _FlowList(review_item.get("cites") or ())
        review_item["reason"] = ""
        lines.extend("  " + line for line in _dump([review_item]).rstrip().splitlines())
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draft guideline-grounded scenario labels.")
    parser.add_argument("--scenario", action="append", dest="scenario_ids")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--model", default="opus")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="rebuild review files from existing model annotations without calling claude",
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
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
    for scenario in scenarios:
        try:
            result = annotate_scenario(
                scenario,
                profile=profile | scenario.profile,
                guidelines_text=guidelines_text,
                citable=citable,
                model=args.model,
                effort=args.effort,
            )
            model_path, _ = write_annotation_files(
                scenario,
                result,
                annotations_dir=args.annotations,
                effort=args.effort,
                guidelines_text=guidelines_text,
                force=args.force,
            )
        except AnnotatorError as exc:
            raise SystemExit(str(exc)) from exc
        cost = float(result.get("total_cost_usd", 0))
        total += cost
        print(f"{scenario.id}: attempts={result['attempts']} cost=${cost:.2f} {model_path}")
    print(f"total cost: ${total:.2f}")
    return 0
