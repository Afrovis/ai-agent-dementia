"""End-of-run triage: a ranked fix list with proposals, backed by quick tests.

After a director batch, triage turns the run's findings into `fixes.md` in
four steps, so no single Claude context has to hold the whole run. The
single-agent version it replaces read ~4.8M cached tokens in 44 turns on
2026-09-24T1435-live, re-reading a ~110K context on every turn.

1. Prep (Python, no model): group `bugs.jsonl` by fingerprint into clusters,
   each with severity, scenes, report links and a short evidence window from
   the first occurrence's report.md.
2. Plan (Opus, no tools): rank the clusters, merge those with one likely
   cause, set aside known and review-only ones, and write one question per
   investigation.
3. Investigate (Sonnet, one `claude -p` per investigation, in series): find
   the cause in the code, propose a change and, when cheap, check it in a
   throwaway git worktree. Before each one the GPU gate (`scene_lab.gpu`)
   waits for an idle host, so quick-test timings come from a quiet machine.
4. Write (Opus, read-only tools): check weak or conflicting findings against
   the cited code and write fixes.md. If this step fails, fixes.md is
   rendered from the investigations without it.

All calls run on the claude.ai subscription, like the director and the mind,
and each call's token use goes to the run's `usage.jsonl`. Intermediate
results stay in the run's `triage/` folder.

Sandbox: the worktree is a detached `git worktree add` of the run's commit
under the system temp directory. Workers can read everything, edit only that
copy, run only Python through a wrapper pinned to the copy, and use read-only
git. No Docker, Ollama or network, no commits, never a real checkout or
service. Each investigation's diff is saved as `triage/<n>.patch`, joined into
`fixes.patch` for a person to review, and the copy is reset in between. The
worktree is removed afterwards.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import gpu, usage
from .bugs import DESCRIPTIONS, RANK, usage_limited

PLAN_MODEL = "opus"
WORKER_MODEL = "sonnet"
WRITE_MODEL = "opus"
EFFORT = "medium"
MAX_INVESTIGATIONS = 6
PLAN_TIMEOUT_S = 600
WORKER_TIMEOUT_S = 900
WRITE_TIMEOUT_S = 900
# Evidence window around a cluster's first occurrence, in report.md seconds.
WINDOW_BEFORE_S = 15
WINDOW_AFTER_S = 10
WINDOW_LINES = 30
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

_CONTEXT = """\
You are helping triage one scene_lab run of a bedside night companion for a
person living with dementia. Scene_lab plays a simulated person against the
real agent stack and checks label-free invariants (the checks are in
tests/scene_lab/scene_lab/invariants.py). Harm to the person comes first.
"""

_PLAN = (
    _CONTEXT
    + """
Your job in this step: plan the investigation. You have no tools; everything
you need is below. Another model (Sonnet) investigates each item you choose,
one at a time, in a fresh context, so each question must stand on its own.

- Order by harm to the person first, then reach (count, scenes).
- Merge clusters that most likely share one cause into one investigation.
- At most {max_items} investigations. Prefer critical and major clusters.
- Set aside, with a one-line reason: review-only clusters that are designed
  behaviour (see the known fingerprints below), items already on the open
  list that did not get worse, and anything not worth a look now.
- A checker false positive or harness problem is worth an investigation only
  if it hides or invents harm; otherwise set it aside as checker or harness.
- For each investigation give a precise question and 1-4 starting files
  (repository paths) so the worker does not have to search widely.
"""
)

_SANDBOX = """\
The run directory is `{run_dir}` and is read-only. Use Read, Grep or Glob with
absolute paths there. Bash permissions cover single commands only: do not
prefix commands with `cd`, join them with `;` or `&&`, or pipe them; use Read,
Grep and Glob to look at files. Your working directory is a throwaway git
worktree of the run's commit. Do not read `.env` or other local secrets; the
example configuration contains the keys.
"""

_QUICK_TESTS = """\
Quick test: if your proposal can be checked in a few minutes, check it here.
- Run Python only as `{py} ...`, e.g. `{py} -m pytest -q
  services/agent/tests/test_main.py -k name`. To rescore, first copy the run
  into {scratch} with `{py} -c` and `shutil.copytree`, then `{py} -m scene_lab
  rescore {scratch}/run-copy`; never rescore the run directory. To replay a
  moment: `{py} -m scene_lab promote <run dir> --scene S --at T --before B
  --after A --to session_replay --out-dir {scratch} --no-claude`, then
  `{py} -m session_replay run {scratch}/NAME.jsonl --llm recorded
  --llm-latency recorded --invariants --runs-root {scratch}/runs`. Recorded
  replays use the run's recorded interpretations; composed wording falls back
  to templates there.
- Edit files in this worktree freely; it is discarded and your diff is saved
  for review. Do not commit. Do not write outside it and {scratch}.
- No Docker, no Ollama or other model calls, no network. At most two quick
  tests; stop one that is not converging and report it as not tested.
"""

_WORKER = (
    _CONTEXT
    + """
Your job: investigate one item from the run's fix list and report the cause
and a concrete, reviewable fix. Stay on this item. Start from the files named
below and read only what you need (aim for about fifteen tool calls). Useful
background if relevant: services/agent/agent/ (main.py reply routing,
session.py, veto.py, strategies.py, rules.py, llm.py),
tests/decision_bench/guidelines.md (care guidelines for every spoken
sentence), tests/scene_lab/BASELINE.md (earlier findings).

Say whether it is an agent bug, a checker false positive, a harness problem
or not a bug. Cite file:line for the cause, and quote the utterance or
sentence from the evidence. Put an exact question in owner_decision when the
fix needs the owner to choose (wording the person hears, a safety trade-off),
otherwise "none".

{sandbox}
{quick_tests}
{gpu_note}
Item {n}: {title}
Question: {question}
Starting files: {start_files}

Clusters (from bugs.jsonl, with the first occurrence's report.md lines):

{clusters}
"""
)

_WRITE = (
    _CONTEXT
    + """
Your job: write the run's fix list from the investigations below. Each was
done by another model in its own context; treat them as leads. For every item
with low confidence, and for items that conflict, open the cited file:line
(and the report.md link if needed) and correct or drop what does not hold. At
most about ten reads in total; do not re-investigate from scratch, and do not
claim quick tests that were not run.

{sandbox}
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
- **Quick test:** <what was run and the result, with numbers; or "not tested:
  <why>">
- **Owner decision:** <none, or the exact question to answer first>

Order by harm to the person first, then reach. After the numbered items, add
"## Checker or harness issues" and "## Not worth fixing now" sections (short
bullets; include what the plan set aside). Plain, direct English; no
marketing words. {patch_note}
"""
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "investigations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "clusters": {"type": "array", "items": {"type": "string"}},
                    "question": {"type": "string"},
                    "start_files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "clusters", "question", "start_files"],
            },
        },
        "set_aside": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clusters": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["clusters", "reason"],
            },
        },
    },
    "required": ["summary", "investigations", "set_aside"],
}

WORKER_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["agent_bug", "checker", "harness", "not_a_bug"]},
        "cause": {"type": "string"},
        "cause_location": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "proposal": {"type": "string"},
        "quick_test": {
            "type": "object",
            "properties": {
                "ran": {"type": "boolean"},
                "command": {"type": "string"},
                "result": {"type": "string"},
            },
            "required": ["ran", "result"],
        },
        "owner_decision": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "kind",
        "cause",
        "cause_location",
        "evidence",
        "proposal",
        "quick_test",
        "owner_decision",
        "confidence",
    ],
}


class LimitReached(RuntimeError):
    """The subscription's usage limit: every later call fails the same way."""


# --- 1. prep ---------------------------------------------------------------

_ANCHOR = re.compile(r'^<a id="t=([0-9.]+)"></a>\s*')


def evidence_window(run_dir: Path, entry: dict) -> list[str]:
    """report.md lines from shortly before to shortly after an occurrence."""
    report = run_dir / entry["report"].split("#", 1)[0]
    if not report.exists():
        return []
    lines = []
    for line in report.read_text().splitlines():
        match = _ANCHOR.match(line)
        if not match:
            continue
        t = float(match.group(1))
        if entry["t"] - WINDOW_BEFORE_S <= t <= entry["t"] + WINDOW_AFTER_S:
            lines.append(line[match.end() :][:220])
    return lines[:WINDOW_LINES]


def prepare_clusters(run_dir: Path) -> list[dict]:
    """One cluster per fingerprint, worst first (severity, then count)."""
    bugs = run_dir / "bugs.jsonl"
    if not bugs.exists():
        return []
    groups: dict[str, list[dict]] = {}
    for line in bugs.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            groups.setdefault(row["fingerprint"], []).append(row)
    clusters = []
    for fingerprint, rows in groups.items():
        first = rows[0]
        severity = min((r["severity"] for r in rows), key=lambda s: RANK.get(s, 99))
        worst = next(r for r in rows if r["severity"] == severity)
        clusters.append(
            {
                "fingerprint": fingerprint,
                "check": first["check"],
                "meaning": DESCRIPTIONS.get(first["check"], "see invariants.py"),
                "origin": first["origin"],
                "severity": severity,
                "count": len(rows),
                "scenes": sorted({r["scene"] for r in rows}),
                "summary": worst["summary"],
                "evidence": worst.get("evidence", [])[:4],
                "occurrences": [r["report"] for r in rows[:8]],
                "window": evidence_window(run_dir, worst),
            }
        )
    clusters.sort(key=lambda c: (RANK.get(c["severity"], 99), -c["count"], c["fingerprint"]))
    for n, cluster in enumerate(clusters, 1):
        cluster["id"] = f"c{n}"
    return clusters


def baseline_notes(repo: Path) -> str:
    """The known-by-design fingerprints and the latest open list from BASELINE.md."""
    path = repo / "tests/scene_lab/BASELINE.md"
    if not path.exists():
        return "(no BASELINE.md)"
    text = path.read_text()
    parts = []
    known = re.search(r"Known review fingerprint.*?(?:\n\n|\Z)", text, re.S)
    if known:
        parts.append(known.group(0).strip())
    start = text.rfind("Still open")
    if start >= 0:
        parts.append(text[start:])
    return "\n\n".join(parts)[:8000] or "(nothing marked open)"


def _cluster_text(clusters: list[dict], *, windows: bool) -> str:
    rows = []
    for c in clusters:
        head = (
            f"[{c['id']}] {c['fingerprint']}: {c['severity']}, {c['count']}x in "
            f"{len(c['scenes'])} scene(s) ({', '.join(c['scenes'])}); {c['origin']}\n"
            f"  check: {c['meaning']}\n  first: {c['summary']}\n"
            + "".join(f"  evidence: {e}\n" for e in c["evidence"])
            + f"  links: {', '.join(c['occurrences'])}\n"
        )
        if windows and c["window"]:
            head += "  report.md around it:\n" + "".join(f"    {w}\n" for w in c["window"])
        rows.append(head)
    return "\n".join(rows) or "(no findings)"


# --- sandbox and calls -----------------------------------------------------


def _claude_command(
    model: str,
    tools: list[str],
    add_dirs: list[Path],
    schema: dict | None = None,
) -> list[str]:
    executable = shutil.which("claude")
    if executable is None:
        raise RuntimeError("claude executable was not found on PATH")
    command = [
        executable,
        "-p",
        "--model",
        model,
        "--effort",
        EFFORT,
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]
    for directory in add_dirs:
        command += ["--add-dir", str(directory)]
    if schema is not None:
        command += ["--json-schema", json.dumps(schema)]
    command += ["--allowedTools", *tools] if tools else ["--tools", ""]
    return command


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


def _call(
    runner: Runner,
    command: list[str],
    prompt: str,
    cwd: Path,
    timeout: int,
    log: Path,
    role: str,
    **extra,
) -> dict:
    """One Claude call; logs its usage and returns the CLI's JSON result."""
    model = command[command.index("--model") + 1]
    try:
        completed = runner(command, prompt, cwd, timeout)
    except subprocess.TimeoutExpired:
        usage.record(log, role, error=f"timed out after {timeout} s", requested_model=model)
        raise RuntimeError(f"{role}: claude timed out after {timeout} s") from None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = None
    detail = completed.stderr.strip() or completed.stdout.strip()
    failed = completed.returncode != 0 or not isinstance(payload, dict) or payload.get("is_error")
    usage.record(
        log,
        role,
        payload=payload if isinstance(payload, dict) else None,
        error=detail[:2000] if failed else None,
        requested_model=model,
        prompt_chars=len(prompt),
        **extra,
    )
    if failed:
        message = f"{role}: claude failed (status {completed.returncode}): {detail}"
        if usage_limited(detail):
            raise LimitReached(message)
        raise RuntimeError(message)
    return payload


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


def _git(worktree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(worktree), *args], capture_output=True, text=True, check=False
    ).stdout


def _read_tools() -> list[str]:
    return ["Read", "Grep", "Glob", "Bash(git log:*)", "Bash(ls:*)"]


def _worker_tools(py: Path, quick_tests: bool) -> list[str]:
    tools = _read_tools()
    if quick_tests:
        tools += ["Edit", "Write", f"Bash({py}:*)", "Bash(git diff:*)", "Bash(git status:*)"]
    return tools


# --- 2. plan ----------------------------------------------------------------


def _director_rationale(run_dir: Path) -> str:
    director = run_dir / "director.jsonl"
    rows = []
    if director.exists():
        for line in director.read_text().splitlines():
            row = json.loads(line)
            if row.get("rationale"):
                rows.append(f"- {row.get('scene')}: {row['rationale']}")
    return "\n".join(rows) or "(none)"


def build_plan_prompt(run_dir: Path, repo: Path, clusters: list[dict], max_items: int) -> str:
    # No report.md windows here: they made up ~46K of a 65K-character plan
    # prompt on 2026-09-24T1435-live, and only the workers need them.
    agent_files = sorted(
        f"{p.relative_to(repo)} ({len(p.read_text().splitlines())} lines)"
        for p in (repo / "services/agent/agent").glob("*.py")
    )
    return (
        _PLAN.format(max_items=max_items)
        + f"\nRun id: {run_dir.name}\n"
        + f"Agent code: {', '.join(agent_files) or '(not found)'}\n\n"
        + "From tests/scene_lab/BASELINE.md:\n\n"
        + baseline_notes(repo)
        + "\n\nDirector rationale per scene:\n"
        + _director_rationale(run_dir)
        + "\n\nClusters, worst first:\n\n"
        + _cluster_text(clusters, windows=False)
    )


def fallback_plan(clusters: list[dict], max_items: int, reason: str) -> dict:
    """Without a planner: one investigation per worst non-review cluster."""
    chosen = [c for c in clusters if c["severity"] != "review"][:max_items]
    ids = {c["id"] for c in chosen}
    return {
        "summary": f"Planned without the planner ({reason}): worst clusters first.",
        "investigations": [
            {
                "title": f"{c['check']} {c['severity']}: {c['summary'][:80]}",
                "clusters": [c["id"]],
                "question": f"Why does {c['fingerprint']} happen, and what is the fix?",
                "start_files": ["services/agent/agent/main.py"],
            }
            for c in chosen
        ],
        "set_aside": [
            {"clusters": [c["id"]], "reason": "not planned (fallback plan)"}
            for c in clusters
            if c["id"] not in ids
        ],
    }


def clean_plan(plan: dict, clusters: list[dict], max_items: int) -> dict:
    """Drop unknown cluster ids, cap the list, and set aside what the plan left out."""
    known = {c["id"] for c in clusters}
    items = []
    for item in plan.get("investigations", []):
        ids = [i for i in item.get("clusters", []) if i in known]
        if ids and len(items) < max_items:
            items.append({**item, "clusters": ids})
    aside = [
        {**a, "clusters": [i for i in a.get("clusters", []) if i in known]}
        for a in plan.get("set_aside", [])
    ]
    covered = {i for x in items + aside for i in x["clusters"]}
    missing = sorted(known - covered, key=lambda i: int(i[1:]))
    if missing:
        aside.append({"clusters": missing, "reason": "not placed by the plan"})
    return {"summary": plan.get("summary", ""), "investigations": items, "set_aside": aside}


# --- 3. investigate -----------------------------------------------------------


def build_worker_prompt(
    n: int,
    item: dict,
    clusters: list[dict],
    run_dir: Path,
    py: Path,
    scratch: Path,
    quick_tests: bool,
    gpu_state: dict,
) -> str:
    chosen = [c for c in clusters if c["id"] in item["clusters"]]
    gpu_note = (
        ""
        if gpu_state.get("idle", True)
        else f"The host GPU was busy when you started ({gpu_state.get('busy')}): say that "
        "any timing you measure is contended.\n"
    )
    return _WORKER.format(
        sandbox=_SANDBOX.format(run_dir=run_dir),
        quick_tests=(
            _QUICK_TESTS.format(py=py, scratch=scratch)
            if quick_tests
            else "Analysis only: do not run or edit anything.\n"
        ),
        gpu_note=gpu_note,
        n=n,
        title=item["title"],
        question=item["question"],
        start_files=", ".join(item.get("start_files", [])) or "(none given)",
        clusters=_cluster_text(chosen, windows=True),
    )


# --- 4. write ------------------------------------------------------------------


def render_fixes(run_dir: Path, plan: dict, results: list[dict], clusters: list[dict], note: str):
    """fixes.md straight from the investigations, when the write step cannot run."""
    by_id = {c["id"]: c for c in clusters}
    lines = [f"# Fix list: {run_dir.name}", "", f"_{note}_", "", plan.get("summary", ""), ""]
    for n, result in enumerate(results, 1):
        item = result["item"]
        found = result.get("result") or {}
        chosen = [by_id[i] for i in item["clusters"] if i in by_id]
        reach = "; ".join(f"{c['fingerprint']} ({c['severity']}, {c['count']}x)" for c in chosen)
        lines.append(f"## {n}. {item['title']}")
        if not found:
            lines += [
                f"- **Severity / reach:** {reach}",
                f"- **Not investigated:** {result.get('error')}",
                "",
            ]
            continue
        test = found.get("quick_test", {})
        lines += [
            f"- **Severity / reach:** {reach}",
            f"- **Kind / confidence:** {found.get('kind')} / {found.get('confidence')}",
            "- **Evidence:** " + "; ".join(found.get("evidence", [])),
            f"- **Cause:** {found.get('cause_location')}: {found.get('cause')}",
            f"- **Proposal:** {found.get('proposal')}",
            "- **Quick test:** "
            + (
                f"{test.get('command', '')}: {test.get('result')}"
                if test.get("ran")
                else f"not tested: {test.get('result')}"
            ),
            f"- **Owner decision:** {found.get('owner_decision')}",
            "",
        ]
    lines.append("## Not worth fixing now")
    for aside in plan.get("set_aside", []):
        names = ", ".join(by_id[i]["fingerprint"] for i in aside["clusters"] if i in by_id)
        lines.append(f"- {names}: {aside['reason']}")
    return "\n".join(lines).rstrip() + "\n"


def triage(
    run_dir: Path,
    repo_root: Path,
    commit: str = "HEAD",
    *,
    quick_tests: bool = True,
    runner: Runner = _run_claude,
    gpu_gate: Callable[[], dict] = gpu.wait_idle,
    max_items: int = MAX_INVESTIGATIONS,
) -> Path:
    """Write `fixes.md` (and `fixes.patch` when quick tests edited code)."""
    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root)
    work = run_dir / "triage"
    work.mkdir(exist_ok=True)
    log = run_dir / "usage.jsonl"
    tmp = Path(tempfile.mkdtemp(prefix=f"scene-lab-triage-{run_dir.name}-"))
    worktree = None
    try:
        worktree, scratch, py = _prepare_workspace(repo_root, commit, tmp)
        clusters = prepare_clusters(run_dir)
        (work / "clusters.json").write_text(json.dumps(clusters, indent=2) + "\n")

        # 2. plan
        prompt = build_plan_prompt(run_dir, worktree, clusters, max_items)
        try:
            payload = _call(
                runner,
                _claude_command(PLAN_MODEL, [], [], PLAN_SCHEMA),
                prompt,
                scratch,
                PLAN_TIMEOUT_S,
                log,
                "triage-plan",
            )
            plan = clean_plan(payload.get("structured_output") or {}, clusters, max_items)
        except LimitReached:
            raise
        except Exception as exc:  # noqa: BLE001 - a planner failure has a fallback
            plan = fallback_plan(clusters, max_items, f"{type(exc).__name__}: {exc}"[:200])
        (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

        # 3. investigate, one at a time on a quiet GPU
        results: list[dict] = []
        patches: list[str] = []
        stopped = None
        for n, item in enumerate(plan["investigations"], 1):
            if stopped:
                results.append({"item": item, "error": stopped})
                continue
            gpu_state = gpu_gate()
            prompt = build_worker_prompt(
                n, item, clusters, run_dir, py, scratch, quick_tests, gpu_state
            )
            entry: dict = {"item": item, "gpu": gpu_state}
            try:
                payload = _call(
                    runner,
                    _claude_command(
                        WORKER_MODEL,
                        _worker_tools(py, quick_tests),
                        [run_dir, scratch],
                        WORKER_SCHEMA,
                    ),
                    prompt,
                    worktree,
                    WORKER_TIMEOUT_S,
                    log,
                    "triage-worker",
                    item=n,
                    gpu_idle=gpu_state.get("idle"),
                    gpu_utilization_pct=gpu_state.get("device_utilization_pct"),
                )
                entry["result"] = payload.get("structured_output")
            except LimitReached as exc:
                stopped = f"usage limit reached: {exc}"[:300]
                entry["error"] = stopped
            except Exception as exc:  # noqa: BLE001 - one failed item does not stop the rest
                entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
            diff = _git(worktree, "diff")
            if diff.strip():
                (work / f"{n}.patch").write_text(diff)
                patches.append(f"# item {n}: {item['title']}\n{diff}")
                entry["patch"] = f"triage/{n}.patch"
            _git(worktree, "reset", "-q", "--hard")
            _git(worktree, "clean", "-fdq")
            (work / f"{n}.json").write_text(json.dumps(entry, indent=2) + "\n")
            results.append(entry)
        if patches:
            (run_dir / "fixes.patch").write_text("\n".join(patches))

        # 4. write
        fixes = run_dir / "fixes.md"
        if stopped or not any(r.get("result") for r in results):
            reason = stopped or "no investigation returned a result"
            fixes.write_text(
                render_fixes(
                    run_dir, plan, results, clusters, f"Written without the final pass: {reason}"
                )
            )
            return fixes
        patch_note = (
            "Mention per item that its diff is in triage/<n>.patch (joined in fixes.patch)."
            if patches
            else ""
        )
        prompt = (
            _WRITE.format(sandbox=_SANDBOX.format(run_dir=run_dir), patch_note=patch_note)
            + f"\nRun id: {run_dir.name}\n\nPlan summary: {plan['summary']}\n\n"
            + "Set aside by the plan:\n"
            + json.dumps(plan["set_aside"], indent=1)
            + "\n\nClusters:\n\n"
            + _cluster_text(clusters, windows=False)
            + "\n\nInvestigations:\n\n"
            + json.dumps(
                [
                    {"n": n, **{k: r.get(k) for k in ("item", "result", "error", "patch")}}
                    for n, r in enumerate(results, 1)
                ],
                indent=1,
            )
        )
        try:
            payload = _call(
                runner,
                _claude_command(WRITE_MODEL, _read_tools(), [run_dir]),
                prompt,
                worktree,
                WRITE_TIMEOUT_S,
                log,
                "triage-write",
            )
            fixes.write_text(str(payload.get("result", "")).strip() + "\n")
        except Exception as exc:  # noqa: BLE001 - fall back to the investigations as they are
            fixes.write_text(
                render_fixes(
                    run_dir,
                    plan,
                    results,
                    clusters,
                    f"Written without the final pass: {type(exc).__name__}: {exc}"[:300],
                )
            )
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
