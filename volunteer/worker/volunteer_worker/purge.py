"""File deletion for a submission marked `deleted` (HANDOFF.md section 6, V8).

Removing files that are already gone is a no-op, so a sweep can run
repeatedly over every `deleted` row without tracking which ones it already
handled.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from volunteer_common import paths


def purge_submission(data_root: Path, submission_id: str) -> None:
    shutil.rmtree(paths.incoming_dir(data_root, submission_id), ignore_errors=True)
    for path in (data_root / "raw").glob(f"{submission_id}.*"):
        path.unlink(missing_ok=True)
    shutil.rmtree(paths.clip_dir(data_root, submission_id), ignore_errors=True)
    for path in (data_root / "analysis").glob(f"{submission_id}__*"):
        path.unlink(missing_ok=True)
    for path in (data_root / "results").glob(f"{submission_id}*"):
        path.unlink(missing_ok=True)


def purge_pending(data_root: Path, submission_ids: list[str]) -> None:
    for submission_id in submission_ids:
        purge_submission(data_root, submission_id)
