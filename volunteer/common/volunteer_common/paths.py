"""Layout under `VOLUNTEER_DATA_DIR` (HANDOFF.md section 5.2).

Every path helper takes the data root explicitly rather than reading the
environment itself, so `web` and `worker` tests can point it at a temp
directory without touching real configuration.
"""

from __future__ import annotations

from pathlib import Path

ID_PATTERN_LENGTH = 22


def db_path(data_root: Path) -> Path:
    return data_root / "volunteer.db"


def public_key_path(data_root: Path) -> Path:
    return data_root / "public.pem"


def incoming_dir(data_root: Path, submission_id: str) -> Path:
    return data_root / "incoming" / submission_id


def key_wrapped_path(data_root: Path, submission_id: str) -> Path:
    return incoming_dir(data_root, submission_id) / "key.wrapped"


def chunk_path(data_root: Path, submission_id: str, n: int) -> Path:
    return incoming_dir(data_root, submission_id) / f"chunk_{n:06d}.enc"


def submission_json_path(data_root: Path, submission_id: str) -> Path:
    return incoming_dir(data_root, submission_id) / "submission.json"


def raw_path(data_root: Path, submission_id: str, suffix: str = "mkv") -> Path:
    return data_root / "raw" / f"{submission_id}.{suffix}"


def clip_dir(data_root: Path, submission_id: str) -> Path:
    return data_root / "clips" / submission_id


def analysis_path(data_root: Path, submission_id: str) -> Path:
    return data_root / "analysis" / f"{submission_id}__pipeline.mp4"


def result_path(data_root: Path, submission_id: str) -> Path:
    return data_root / "results" / f"{submission_id}.mp4.enc"


def sample_dir(data_root: Path) -> Path:
    return data_root / "sample"


def prompts_dir(data_root: Path) -> Path:
    return data_root / "prompts"


def prompt_path(data_root: Path, step_id: str) -> Path:
    return prompts_dir(data_root) / f"{step_id}.wav"
