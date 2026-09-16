"""Allowlisted `video_eval` runner (HANDOFF.md rule 3: "Enforce it in code
... not by convention").

`worker` must never run `sheets`, `label-codex` or `label-local` on
volunteer clips, because those send data to Ollama or Codex. `run` is the
only way the job in section 5.8 invokes `video_eval`, and it raises before
touching a subprocess if the subcommand is not on the allowlist.
"""

from __future__ import annotations

import subprocess

ALLOWED_SUBCOMMANDS = frozenset({"prepare", "predict", "visualize"})


class DisallowedSubcommand(Exception):
    pass


def run(subcommand: str, args: list[str], *, data_root: str, python: str = "python") -> str:
    """Run `python -m video_eval <subcommand> --data-root <data_root> <args>`.

    Returns captured stdout. Raises `subprocess.CalledProcessError` (with
    stderr on the exception) on a non-zero exit, so the caller can put it in
    `error_detail` without a plaintext video ever reaching a message the
    volunteer sees.
    """
    if subcommand not in ALLOWED_SUBCOMMANDS:
        raise DisallowedSubcommand(
            f"{subcommand!r} is not on the allowlist {sorted(ALLOWED_SUBCOMMANDS)}"
        )
    command = [python, "-m", "video_eval", subcommand, "--data-root", data_root, *args]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout
