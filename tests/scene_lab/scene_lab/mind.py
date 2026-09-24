"""Isolated Claude subscription mind for synthetic scene personas."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import usage
from .scene import Persona, State, Zone


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MindMove(_Strict):
    state: State
    zone: Zone
    over_s: float = Field(ge=0)


class MindBeat(_Strict):
    wait: float | None = Field(default=None, ge=0)
    say: str | None = None
    style: Literal["normal", "mumble", "trailing"] | None = None
    move: MindMove | None = None
    interrupt_if_agent_speaks: Literal[True] | None = None

    @model_validator(mode="after")
    def one_action(self) -> MindBeat:
        if (
            sum(
                x is not None
                for x in (self.wait, self.say, self.move, self.interrupt_if_agent_speaks)
            )
            != 1
        ):
            raise ValueError("each beat needs exactly one action")
        if self.say is not None and not self.say.strip():
            raise ValueError("say must contain text")
        if self.style is not None and self.say is None:
            raise ValueError("style requires say")
        return self


class MindOutput(_Strict):
    beats: list[MindBeat]
    end_scene: bool
    note: str


def _heard(text: str, hearing: str) -> str:
    if hearing != "poor":
        return text
    words = text.split()
    # Hash each word and position: stable across calls and processes.
    return (
        " ".join(
            word
            for i, word in enumerate(words)
            if hashlib.sha256(f"{i}:{word.lower()}".encode()).digest()[0] % 4 != 0
        )
        or "[could not make out the words]"
    )


def build_prompt(persona: Persona, context: dict) -> str:
    history = []
    for row in context.get("history", []):
        kind = row.get("kind")
        if kind == "heard":
            history.append(
                f"The bedside device said to you: {_heard(row['text'], persona.hearing)}"
            )
        elif kind == "say":
            history.append(f"You said: {row['text']}")
        elif kind == "move":
            history.append(f"You moved: {row['state']} in {row['zone']}")
    scene = "\n".join(history) or "Nothing has happened yet."
    return (
        "You play the person in the bed: a confused older adult awake at night. "
        "You are never the bedside device, a carer or a narrator; the device's words are "
        "listed as 'The bedside device said to you' and are not yours to repeat or continue. "
        "Play the person respectfully. "
        "Use plain speech and short lines. Never caricature dementia. Stay in persona. "
        "Pace like a real person: say one or two short lines, then usually wait 8 to 20 "
        "seconds for the device to answer before speaking again; only an impatient persona "
        "or one who interrupts speaks sooner. "
        "Express the hidden need in your own natural way; do not name its keyword unless natural. "
        "Return JSON only with beats, end_scene, and note. Each beat is exactly one of "
        '{"wait": seconds}, {"say": text, "style": "normal|mumble|trailing"}, '
        '{"move": {"state": state, "zone": zone, "over_s": seconds}}, or '
        '{"interrupt_if_agent_speaks": true}. '
        "Use only states in_bed, sitting_up, standing, walking, on_floor, absent; "
        "zones bed, door, bathroom_path, other. Keep the plan short.\n\n"
        f"Persona: {persona.summary}\nHidden need: {persona.hidden_need or 'none'}\n"
        f"Hearing: {persona.hearing}; patience: {persona.patience}; "
        f"interrupts: {persona.interrupts}\n"
        f"Scene so far:\n{scene}\n"
        f"On screen: {context.get('headline') or '[nothing]'}\n"
        f"Elapsed: {context.get('t', 0):.1f} s\n"
        f"Trigger: {context.get('trigger', context.get('reason', 'scene start'))}"
    )


class ClaudeMind:
    def __init__(
        self,
        persona: Persona,
        model: str = "sonnet",
        runner=None,
        timeout_s: float = 60,
        usage_log: Path | None = None,
        scene: str | None = None,
    ):
        self.persona = persona
        self.model = model
        self.runner = runner
        self.timeout_s = timeout_s
        # Where each call's token use goes (the run's usage.jsonl); None skips it.
        self.usage_log = usage_log
        self.scene = scene
        self.calls: list[dict] = []
        self.failures: list[str] = []

    async def decide(self, context: dict):
        from decision_bench.annotate import run_claude

        from .run import Plan

        runner = self.runner or run_claude
        prompt = build_prompt(self.persona, context)
        schema = MindOutput.model_json_schema()
        error = ""
        for attempt in range(2):
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory() as directory:
                    system = Path(directory) / "system.txt"
                    system.write_text("Return only valid JSON for the requested person plan.")
                    try:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(
                                runner, system, prompt + error, schema, self.model, "low"
                            ),
                            timeout=self.timeout_s,
                        )
                    except Exception as exc:
                        usage.record(
                            self.usage_log,
                            "mind",
                            error=f"{type(exc).__name__}: {exc}",
                            requested_model=self.model,
                            scene=self.scene,
                        )
                        raise
                usage.record(
                    self.usage_log,
                    "mind",
                    payload=result if isinstance(result, dict) else None,
                    requested_model=self.model,
                    scene=self.scene,
                    prompt_chars=len(prompt + error),
                )
                raw = (
                    result.get("structured_output", result) if isinstance(result, dict) else result
                )
                if isinstance(raw, str):
                    raw = json.loads(raw)
                output = MindOutput.model_validate(raw)
                beats = [beat.model_dump(exclude_none=True) for beat in output.beats]
                self.calls.append(
                    {
                        "t": context.get("t", 0),
                        "trigger": context.get("trigger", context.get("reason")),
                        "latency_s": round(time.monotonic() - started, 3),
                        "ok": True,
                        "beats": beats,
                        "note": output.note,
                    }
                )
                return Plan(beats=beats, end_scene=output.end_scene, note=output.note)
            except Exception as exc:
                error = (
                    f"\nPrevious output failed validation: {type(exc).__name__}: {exc}. "
                    "Return corrected JSON only."
                )
                self.calls.append(
                    {
                        "t": context.get("t", 0),
                        "trigger": context.get("trigger", context.get("reason")),
                        "latency_s": round(time.monotonic() - started, 3),
                        "ok": False,
                        "beats": [],
                        "note": str(exc),
                    }
                )
        self.failures.append(f"mind failure: {error.strip()}")
        return Plan(end_scene=True, note="mind failure")
