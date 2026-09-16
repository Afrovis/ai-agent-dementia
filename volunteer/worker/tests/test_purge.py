from __future__ import annotations

from pathlib import Path

from volunteer_common import paths

from volunteer_worker.purge import purge_pending, purge_submission


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_purge_submission_removes_every_layout_path(tmp_path: Path) -> None:
    submission_id = "sub-purge-0000000000001"
    _touch(paths.key_wrapped_path(tmp_path, submission_id))
    _touch(paths.chunk_path(tmp_path, submission_id, 0))
    _touch(paths.raw_path(tmp_path, submission_id))
    _touch(paths.clip_dir(tmp_path, submission_id) / "predictions" / "x.jsonl")
    _touch(paths.analysis_path(tmp_path, submission_id))
    _touch(paths.result_path(tmp_path, submission_id))

    purge_submission(tmp_path, submission_id)

    assert not paths.incoming_dir(tmp_path, submission_id).exists()
    assert not paths.raw_path(tmp_path, submission_id).exists()
    assert not paths.clip_dir(tmp_path, submission_id).exists()
    assert not paths.analysis_path(tmp_path, submission_id).exists()
    assert not paths.result_path(tmp_path, submission_id).exists()


def test_purge_submission_is_idempotent_on_missing_files(tmp_path: Path) -> None:
    purge_submission(tmp_path, "never-existed-0000000")  # must not raise


def test_purge_pending_only_touches_named_ids(tmp_path: Path) -> None:
    keep_id = "sub-keep-00000000000001"
    delete_id = "sub-delete-0000000000001"
    _touch(paths.raw_path(tmp_path, keep_id))
    _touch(paths.raw_path(tmp_path, delete_id))

    purge_pending(tmp_path, [delete_id])

    assert paths.raw_path(tmp_path, keep_id).exists()
    assert not paths.raw_path(tmp_path, delete_id).exists()
