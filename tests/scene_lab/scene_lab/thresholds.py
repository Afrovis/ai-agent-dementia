"""Single source of configurable timing thresholds."""

from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel


class Thresholds(BaseModel):
    reply_deadline_s: float = 5.0
    talk_over_grace_s: float = 1.0
    barge_in_s: float = 0.5
    silence_gap_s: float = 8.0
    direct_reply_window_s: float = 30.0
    settle_s: float = 30.0
    silent_session_s: float = 60.0
    escalated_silent_s: float = 150.0
    loop_lag_s: float = 3.0
    max_scene_s: float = 600.0
    active_phases: list[str] = ["OBSERVING", "ENGAGED", "ESCALATED"]
    estimate_tts_s: float = 0.0
    estimate_min_s: float = 1.0
    estimate_words_per_s: float = 2.5
    tail_s: float = 60.0
    # Mirrors AgentConfig.utterance_presence_seconds: recent speech counts as presence.
    utterance_presence_s: float = 30.0


def load(path: str | Path | None = None) -> Thresholds:
    source = Path(path) if path else files("scene_lab").joinpath("thresholds.yaml")
    return Thresholds.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))
