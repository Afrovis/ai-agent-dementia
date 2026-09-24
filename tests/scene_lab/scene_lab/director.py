"""Choose synthetic scenes using the isolated Claude subscription runner."""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from itertools import product
from pathlib import Path

import yaml

from . import usage
from .bugs import harness_error, usage_limited
from .scene import Scene, load_scene

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
CARDS = REPO / "tests/scene_lab/scenes"
PROFILES = REPO / "tests/scene_lab/profiles"
PROMPT = (
    "Choose the next synthetic scene to maximise new coverage. If the previous scene "
    "has a critical or major finding, first choose a smaller or harsher variant to "
    "confirm and isolate it. Do not chase review findings: designed pacing silence "
    "with drop reason reassured_recently is known. Never touch code or config outside "
    "the scene card. Portray the person "
    "respectfully, without caricature. Return JSON with scene and rationale. Scene must "
    "have mind: claude, opening beats at 1–4 distinct times, and duration_s <= 600 and "
    "time left. Each beat is exactly one move, say, wait or end; to talk while moving, "
    "give a move beat and a say beat the same at. "
    "Use only the allowed profile and strategies files. Give it a unique id "
    "<category>-<slug>-<n>."
)


def noise_level(scene: Scene) -> str:
    if scene.noise.low_confidence:
        return "low_confidence"
    if scene.noise.flicker:
        return "flicker"
    return "none"


def persona_tag(scene: Scene) -> str:
    if scene.persona_tag:
        return scene.persona_tag
    return next(
        (
            tag
            for marker, tag in (
                ("question-in-gap", "retired-teacher-morning"),
                ("hidden-restroom", "restless-hidden-need"),
                ("interrupting-talker", "interrupting-talker"),
                ("silent-wanderer", "silent-wanderer"),
                ("repeated-question", "repeater"),
            )
            if marker in scene.id
        ),
        "restless-hidden-need",
    )


def coverage(scenes: list[dict]) -> dict:
    dimensions = yaml.safe_load((PACKAGE / "coverage.yaml").read_text())
    counts = Counter()
    for row in scenes:
        for stressor in row.get("stressors", []):
            key = (row["category"], row.get("persona_tag") or "unknown", stressor, row["noise"])
            counts[key] += 1
    return {
        "dimensions": dimensions,
        "counts": {
            "|".join(key): counts[key]
            for key in product(
                dimensions["categories"],
                dimensions["personas"],
                dimensions["stressors"],
                dimensions["noise"],
            )
        },
    }


def coverage_prompt(scenes: list[dict]) -> dict:
    """Coverage for the director's prompt: the dimensions and only the non-zero cells.

    The full grid has 1,008 cells; in 2026-09-24T1435-live 11 were non-zero and
    the zeros made up most of a ~63 KB prompt on every director call.
    """
    full = coverage(scenes)
    return {
        "dimensions": full["dimensions"],
        "counts": {key: n for key, n in full["counts"].items() if n},
        "note": "counts are category|persona|stressor|noise; a cell not listed is 0",
    }


def scenes_prompt(scenes: list[dict]) -> list[dict]:
    """Scenes so far with each scene's failures counted per fingerprint, not listed."""
    rows = []
    for row in scenes:
        failures: dict[str, dict] = {}
        for item in row.get("failures", []):
            entry = failures.setdefault(
                item["fingerprint"], {"severity": item["severity"], "count": 0}
            )
            entry["count"] += 1
        rows.append({**row, "failures": failures})
    return rows


def existing_cards() -> list[tuple[Path, Scene]]:
    return [(path, load_scene(path)) for path in sorted(CARDS.glob("*.yaml"))]


def allowed_files() -> list[str]:
    base = ["config/person.example.yaml", "config/strategies.example.yaml"]
    return base + [str(path.relative_to(REPO)) for path in sorted(PROFILES.rglob("*.yaml"))]


def _valid_config(scene: Scene) -> None:
    allowed = set(allowed_files())
    for field, default in (
        ("profile", "config/person.example.yaml"),
        ("strategies", "config/strategies.example.yaml"),
    ):
        value = getattr(scene, field)
        if value not in allowed | {"default"}:
            raise ValueError(f"{field} must be an allowed file: {value}")
        if value == "default":
            setattr(scene, field, default)


def split_combined_beats(opening: list) -> list:
    """Split a beat that holds several actions into one beat per action at the same time.

    Talking while walking is a move and a say with the same `at`; the director often
    writes them as one beat, which Scene rejects. The split is lossless: the move runs
    first so the body is already moving when the line starts, and `style` stays with
    its say.
    """
    beats = []
    for beat in opening:
        if not isinstance(beat, dict):
            beats.append(beat)
            continue
        actions = [key for key in ("move", "say", "wait", "end") if beat.get(key) is not None]
        if len(actions) < 2:
            beats.append(beat)
            continue
        for key in actions:
            part = {"at": beat.get("at"), key: beat[key]}
            if key == "say" and "style" in beat:
                part["style"] = beat["style"]
            beats.append(part)
    return beats


def output_schema() -> dict:
    """JSON schema for the director's reply. Scene's nested models live in `$defs`, which
    must sit at the root: `#/$defs/...` references resolve from the root document, and a
    schema nested under `properties.scene` made every Claude call fail."""
    scene = Scene.model_json_schema()
    defs = scene.pop("$defs", {})
    return {
        "type": "object",
        "properties": {"scene": scene, "rationale": {"type": "string"}},
        "required": ["scene", "rationale"],
        "$defs": defs,
    }


class Director:
    # Sonnet: a validated, rule-bound pick with a fallback card; a weaker pick costs
    # coverage, not correctness.
    def __init__(self, model: str = "sonnet", runner=None):
        self.model = model
        self.runner = runner
        # Why the last call fell back, if it did; the batch stops on a usage limit.
        self.last_error: str | None = None

    def next_scene(self, state: dict) -> Scene:
        from decision_bench.annotate import run_claude

        run = state["run"]
        scenes = state.get("scenes", [])
        left = state["time_left_s"]
        cards = existing_cards()
        summary = {
            "time_left_s": left,
            "coverage": coverage_prompt(scenes),
            "scenes_so_far": scenes_prompt(scenes),
            "bugs": state.get("bugs", {}),
            "existing_cards": [
                {"id": card.id, "summary": card.persona.summary} for _, card in cards
            ],
            "allowed_files": allowed_files(),
        }
        prompt = PROMPT + "\n\n" + json.dumps(summary, sort_keys=True)
        error = ""
        log = run.path / "usage.jsonl"
        for attempt in range(2):
            try:
                with tempfile.TemporaryDirectory() as directory:
                    system = Path(directory) / "system.txt"
                    system.write_text("Return only structured JSON for a synthetic scene card.")
                    try:
                        result = (self.runner or run_claude)(
                            system, prompt + error, output_schema(), self.model, "low"
                        )
                    except Exception as exc:
                        usage.record(log, "director", error=str(exc), requested_model=self.model)
                        raise
                usage.record(
                    log,
                    "director",
                    payload=result if isinstance(result, dict) else None,
                    requested_model=self.model,
                    prompt_chars=len(prompt + error),
                )
                raw = (
                    result.get("structured_output", result) if isinstance(result, dict) else result
                )
                if isinstance(raw, str):
                    raw = json.loads(raw)
                card = dict(raw["scene"])
                if isinstance(card.get("opening"), list):
                    card["opening"] = split_combined_beats(card["opening"])
                scene = Scene.model_validate(card)
                if not isinstance(raw["rationale"], str) or not raw["rationale"].strip():
                    raise ValueError("rationale required")
                if scene.mind != "claude" or not 1 <= len({b.at for b in scene.opening}) <= 4:
                    raise ValueError(
                        "mind must be claude and opening beats need 1–4 distinct times"
                    )
                if scene.duration_s > left:
                    raise ValueError("duration exceeds time left")
                dimensions = coverage(scenes)["dimensions"]
                if scene.category not in dimensions["categories"]:
                    raise ValueError("unknown category")
                if scene.persona_tag not in dimensions["personas"]:
                    raise ValueError("unknown persona_tag")
                if any(item not in dimensions["stressors"] for item in scene.stressors):
                    raise ValueError("unknown stressor")
                if scene.id in {row["id"] for row in scenes} or not re.fullmatch(
                    rf"{re.escape(scene.category)}-[a-z0-9-]+-[0-9]+", scene.id
                ):
                    raise ValueError("scene id must be unique and category-slug-n")
                _valid_config(scene)
                self._log(
                    run, {"attempt": attempt + 1, "scene": scene.id, "rationale": raw["rationale"]}
                )
                self.last_error = None
                return scene
            except Exception as exc:
                error = f"\nValidation error: {type(exc).__name__}: {exc}. Correct the JSON."
                self._log(run, {"attempt": attempt + 1, "error": str(exc)})
                if usage_limited(str(exc)):
                    break  # a retry fails the same way
        return self.fallback(state, error)

    def fallback(self, state: dict, error: str) -> Scene:
        """Least-covered existing card; used after invalid output or a timed-out call."""
        self.last_error = error
        run = state["run"]
        scenes = state.get("scenes", [])
        left = state["time_left_s"]
        cards = existing_cards()
        counts = coverage(scenes)["counts"]

        def card_count(card: Scene) -> int:
            matched = sum(
                row["category"] == card.category
                and row.get("persona_tag") == persona_tag(card)
                and row["noise"] == noise_level(card)
                for row in scenes
            )
            cells = sum(
                counts.get(
                    "|".join((card.category, persona_tag(card), stressor, noise_level(card))), 0
                )
                for stressor in card.stressors
            )
            return cells if card.stressors else matched

        ranked = sorted(
            cards,
            key=lambda item: (card_count(item[1]), item[1].id),
        )
        if not ranked:
            raise RuntimeError("director failed and no existing cards are available")
        scene = ranked[0][1].model_copy(deep=True)
        scene.persona_tag = persona_tag(scene)
        base = re.sub(r"[^a-z0-9]+", "-", scene.id.lower()).strip("-")
        n = 1
        used = {row["id"] for row in scenes}
        while f"{scene.category}-{base}-{n}" in used:
            n += 1
        scene.id = f"{scene.category}-{base}-{n}"
        scene.duration_s = min(scene.duration_s, left, 600)
        scene.mind = "claude"
        scene.opening = [beat for beat in scene.opening if beat.at <= scene.duration_s][:4]
        if not scene.opening:
            from .scene import Beat

            scene.opening = [Beat(at=0, wait=1)]
        _valid_config(scene)
        run.append([harness_error(run.id, scene.id, "director failure: " + error.strip())])
        self._log(run, {"scene": scene.id, "rationale": "least-covered existing card fallback"})
        return scene

    @staticmethod
    def _log(run, row: dict) -> None:
        with (run.path / "director.jsonl").open("a") as out:
            out.write(json.dumps(row) + "\n")
