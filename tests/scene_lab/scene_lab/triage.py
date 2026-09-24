"""End-of-run triage: a ranked fix list with proposals, backed by quick tests.

After a director batch, Opus (the Claude Code CLI on the claude.ai
subscription, like the director and the mind) reads the run's `bugs.md`,
reports and traces next to the code, and writes `fixes.md` in the run folder:
what to fix first, the likely cause, a concrete proposal, and whether the
owner has to decide something.

When a proposal can be checked cheaply, the triage agent tries it in a
throwaway git worktree under the system temp directory and runs the relevant
tests or replays there. It can read everything, edit only that copy, and run
only Python through a wrapper pinned to the copy, plus read-only git. It
cannot run Docker, Ollama or anything networked, cannot commit, and never
touches a real checkout or service. Its edits come back as `fixes.patch` for a
person to review; the worktree is removed afterwards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

TRIAGE_MODEL = "opus"
TRIAGE_TIMEOUT_S = 2400
# Package roots the wrapper puts first on PYTHONPATH, so tests import the
# throwaway copy instead of the checkout the venv was installed from.
PACKAGE_ROOTS = (
    "shared",
    "services/agent",
    "tests/scene_lab",
    "tests/session_replay",
    "tests/decision_bench",
    "tests/dialogue_bench",
)

Runner = Callable[[list[str], str, Path, int], subprocess.CompletedProcess]

_INSTRUCTIONS = """\
You are triaging one scene_lab run of a bedside night companion for a person
living with dementia. Scene_lab plays a simulated person against the real
agent stack and checks label-free invariants; the findings are in bugs.md and
in each scene's report.md and trace.jsonl. Your job: a ranked list of things to
fix, each with a concrete proposal.

Read first: the run's bugs.md (below), the scene reports it links, the agent
code under services/agent/agent/ (main.py reply routing, session.py, veto.py,
strategies.py, rules.py, llm.py), tests/scene_lab/BASELINE.md (earlier
findings and the current open list, so you do not re-report what is already
known unless it got worse), tests/scene_lab/scene_lab/invariants.py (what each
check means) and tests/decision_bench/guidelines.md (the care guidelines every
spoken sentence must follow). Separate agent bugs from checker false positives
and harness problems.

The run directory is `{run_dir}` and is read-only. Use Read, Grep or Glob with
absolute paths there. Bash permissions cover single commands only: do not
prefix commands with `cd`, join them with `;` or `&&`, or use Bash to list the
run directory. Your working directory is the throwaway worktree. Do not read
`.env` or other local secrets; the example configuration contains the keys.
If you rescore, first copy the run into {scratch} with `{py} -c` and
`shutil.copytree`, then rescore that copy. Never rescore the run directory.

Quick tests. When a proposal can be checked cheaply (a few minutes), check it
in this working directory, which is a throwaway git worktree of the run's
commit:
- Run Python only as `{py} ...`, e.g. `{py} -m pytest -q
  services/agent/tests/test_main.py -k name`, `{py} -m scene_lab rescore
  {scratch}/run-copy`, `{py} -m scene_lab promote <run dir> --scene S --at T --before B
  --after A --to session_replay --out-dir {scratch} --no-claude` then
  `{py} -m session_replay run {scratch}/NAME.jsonl --llm recorded
  --llm-latency recorded --invariants --runs-root {scratch}/runs`.
  Recorded replays use the run's recorded interpretations; composed wording
  falls back to templates there.
- Edit files in this worktree freely; it is discarded. Your diff is saved as
  fixes.patch for review. Do not commit. Do not write outside this worktree
  and {scratch}.
- No Docker, no Ollama or other model calls, no network. At most five quick
  tests; stop a test that is not converging and report it as untested.

Output: reply with the complete fixes.md content and nothing else, in this
shape:

# Fix list: <run id>

One paragraph: what the run exercised, the headline, and how many items need
an owner decision.

## 1. <short title>
- **Severity / reach:** <check ids, counts, scenes affected>
- **Evidence:** <1-3 links to report.md#t=... in this run, with the utterance
  or sentence quoted>
- **Cause:** <file:line and why>
- **Proposal:** <the concrete change, small enough to review>
- **Quick test:** <what you ran and the result, with numbers; or "not tested:
  <why>">
- **Owner decision:** <none, or the exact question to answer first>

Order by harm to the person first, then reach. After the numbered items, add
"## Checker or harness issues" and "## Not worth fixing now" sections (short
bullets). Plain, direct English; no marketing words.
"""


def _claude_command(py: Path, run_dir: Path, scratch: Path, quick_tests: bool) -> list[str]:
    executable = shutil.which("claude")
    if executable is None:
        raise RuntimeError("claude executable was not found on PATH")
    allowed = ["Read", "Grep", "Glob", "Bash(git log:*)", "Bash(ls:*)"]
    if quick_tests:
        allowed += ["Edit", "Write", f"Bash({py}:*)", "Bash(git diff:*)", "Bash(git status:*)"]
    return [
        executable,
        "-p",
        "--model",
        TRIAGE_MODEL,
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--add-dir",
        str(run_dir),
        "--add-dir",
        str(scratch),
        "--allowedTools",
        *allowed,
        "--output-format",
        "json",
    ]


def _run_claude(command: list[str], prompt: str, cwd: Path, timeout: int):
    # Same subscription-only rule as the director and the mind.
    from decision_bench.annotate import _subscription_env

    return subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=cwd,
        check=False,
        env=_subscription_env(),
    )


def _prepare_workspace(repo_root: Path, commit: str, tmp: Path) -> tuple[Path, Path, Path]:
    worktree = tmp / "repo"
    scratch = tmp / "scratch"
    scratch.mkdir()
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree), commit],
        check=True,
        capture_output=True,
    )
    py = tmp / "py"
    pythonpath = ":".join(str(worktree / root) for root in PACKAGE_ROOTS)
    py.write_text(
        "#!/bin/sh\n"
        f'PYTHONPATH="{pythonpath}" SCENE_LAB_RUNS="{scratch}/runs" '
        f'exec "{sys.executable}" "$@"\n'
    )
    py.chmod(0o755)
    return worktree, scratch, py


def build_prompt(run_dir: Path, py: Path, scratch: Path) -> str:
    bugs = (run_dir / "bugs.md").read_text() if (run_dir / "bugs.md").exists() else "(none)"
    director = run_dir / "director.jsonl"
    rationale = []
    if director.exists():
        for line in director.read_text().splitlines():
            row = json.loads(line)
            if row.get("rationale"):
                rationale.append(f"- {row.get('scene')}: {row['rationale']}")
    scenes = sorted(p.name for p in run_dir.iterdir() if (p / "report.md").exists())
    return (
        _INSTRUCTIONS.format(py=py, scratch=scratch, run_dir=run_dir)
        + f"\nRun id: {run_dir.name}\nRun directory: {run_dir}\n"
        + f"Scenes ({len(scenes)}): {', '.join(scenes)}\n\n"
        + "Director rationale per scene:\n"
        + ("\n".join(rationale) or "(none)")
        + "\n\nbugs.md:\n\n"
        + bugs
    )


def triage(
    run_dir: Path,
    repo_root: Path,
    commit: str = "HEAD",
    *,
    quick_tests: bool = True,
    runner: Runner = _run_claude,
    timeout: int = TRIAGE_TIMEOUT_S,
) -> Path:
    """Write `fixes.md` (and `fixes.patch` when quick tests edited code)."""
    run_dir = Path(run_dir).resolve()
    tmp = Path(tempfile.mkdtemp(prefix=f"scene-lab-triage-{run_dir.name}-"))
    worktree = None
    try:
        worktree, scratch, py = _prepare_workspace(Path(repo_root), commit, tmp)
        prompt = build_prompt(run_dir, py, scratch)
        command = _claude_command(py, run_dir, scratch, quick_tests)
        if not quick_tests:
            prompt += "\nDo not run quick tests in this triage: analysis only.\n"
        completed = runner(command, prompt, worktree, timeout)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"claude exited with status {completed.returncode}: {detail}")
        payload = json.loads(completed.stdout)
        if payload.get("is_error"):
            raise RuntimeError(f"claude reported an error: {payload.get('result')}")
        fixes = run_dir / "fixes.md"
        fixes.write_text(str(payload.get("result", "")).strip() + "\n")
        diff = subprocess.run(
            ["git", "-C", str(worktree), "diff"], capture_output=True, text=True, check=False
        ).stdout
        if diff.strip():
            (run_dir / "fixes.patch").write_text(diff)
        return fixes
    finally:
        if worktree is not None:
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree)],
                capture_output=True,
                check=False,
            )
        shutil.rmtree(tmp, ignore_errors=True)


def git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
