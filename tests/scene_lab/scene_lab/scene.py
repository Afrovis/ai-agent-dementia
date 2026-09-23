"""Validated scene cards and fixed scripted beats."""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

State = Literal["in_bed", "sitting_up", "standing", "walking", "on_floor", "absent"]
Zone = Literal["bed", "door", "bathroom_path", "other"]


class Persona(BaseModel):
    summary: str
    hidden_need: str | None = None
    hearing: Literal["normal", "poor"] = "normal"
    patience: Literal["low", "medium", "high"] = "medium"
    interrupts: bool = False


class Noise(BaseModel):
    flicker: float = Field(default=0, ge=0, le=1)
    low_confidence: float = Field(default=0, ge=0, le=1)


class Move(BaseModel):
    state: State
    zone: Zone
    over_s: float = Field(default=0, ge=0)


class Beat(BaseModel):
    at: float = Field(ge=0)
    move: Move | None = None
    say: str | None = None
    style: Literal["normal", "mumble", "trailing"] = "normal"
    wait: float | None = Field(default=None, ge=0)
    end: bool | None = None

    @model_validator(mode="after")
    def one_action(self) -> Beat:
        actions = sum(value is not None for value in (self.move, self.say, self.wait, self.end))
        if actions != 1 or self.end is False:
            raise ValueError("beat needs exactly one move, say, wait, or end: true")
        if self.say is not None and not self.say.strip():
            raise ValueError("say must contain text")
        if self.say is None and self.style != "normal":
            raise ValueError("style requires say")
        return self


class Scene(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    category: str
    persona_tag: str | None = None
    start: time
    duration_s: float = Field(gt=0, le=600)
    profile: str
    strategies: str = "default"
    persona: Persona
    opening: list[Beat] = Field(default_factory=list)
    stressors: list[str] = Field(default_factory=list)
    mind: Literal["claude", "script"] = "script"
    mind_silence_s: float = Field(default=20, gt=0)
    noise: Noise = Field(default_factory=Noise)

    @field_validator("opening")
    @classmethod
    def sorted_beats(cls, beats: list[Beat]) -> list[Beat]:
        if any(left.at > right.at for left, right in zip(beats, beats[1:])):
            raise ValueError("beats must be ordered by at")
        return beats

    @model_validator(mode="after")
    def within_duration(self) -> Scene:
        if any(beat.at > self.duration_s for beat in self.opening):
            raise ValueError("beat extends past scene duration")
        return self


def load_scene(path: str | Path) -> Scene:
    with Path(path).open(encoding="utf-8") as source:
        data = yaml.safe_load(source)
    return Scene.model_validate(data)
