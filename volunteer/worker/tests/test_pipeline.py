from __future__ import annotations

import pytest

from volunteer_worker import pipeline


def test_rejects_disallowed_subcommands() -> None:
    for bad in ("sheets", "label-codex", "label-local"):
        with pytest.raises(pipeline.DisallowedSubcommand):
            pipeline.run(bad, [], data_root="/data")


def test_allowed_subcommand_builds_the_expected_command(monkeypatch) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

        class Result:
            stdout = "ok"

        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    output = pipeline.run("predict", ["--clip", "abc"], data_root="/data", python="python3.12")
    assert output == "ok"
    assert captured["command"] == [
        "python3.12",
        "-m",
        "video_eval",
        "predict",
        "--data-root",
        "/data",
        "--clip",
        "abc",
    ]
