"""End-of-run triage with a fake Claude: workspace, prompt, outputs, cleanup."""

import json
import subprocess

from scene_lab.triage import triage


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "services/agent/agent").mkdir(parents=True)
    (repo / "services/agent/agent/main.py").write_text("MAX_REASSURANCES = 2\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    return repo


def _run(tmp_path):
    run = tmp_path / "runs" / "2026-09-23T2000-live"
    (run / "restroom-1").mkdir(parents=True)
    (run / "restroom-1" / "report.md").write_text("# restroom-1\n")
    (run / "bugs.md").write_text("## TT-1|ENGAGED|restroom|path_light|- (3)\n")
    (run / "director.jsonl").write_text(
        json.dumps({"scene": "restroom-1", "rationale": "probe the path"}) + "\n"
    )
    return run


def test_triage_writes_fix_list_and_patch_then_cleans_up(tmp_path):
    repo, run = _repo(tmp_path), _run(tmp_path)
    seen = {}

    def runner(command, prompt, cwd, timeout):
        seen.update(command=command, prompt=prompt, cwd=cwd)
        # A quick test edits the throwaway copy only.
        (cwd / "services/agent/agent/main.py").write_text("MAX_REASSURANCES = 3\n")
        payload = {"is_error": False, "result": "# Fix list: 2026-09-23T2000-live\n\n## 1. x"}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    fixes = triage(run, repo, "HEAD", runner=runner)

    assert fixes.read_text().startswith("# Fix list: 2026-09-23T2000-live")
    assert "MAX_REASSURANCES = 3" in (run / "fixes.patch").read_text()
    assert "TT-1|ENGAGED|restroom" in seen["prompt"] and "probe the path" in seen["prompt"]
    assert "restroom-1" in seen["prompt"]
    assert "Edit" in seen["command"] and not any("docker" in c for c in seen["command"])
    # The real checkout is untouched and the throwaway worktree is gone.
    assert (repo / "services/agent/agent/main.py").read_text() == "MAX_REASSURANCES = 2\n"
    assert not seen["cwd"].exists()
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True
    ).stdout
    assert len(listed.strip().splitlines()) == 1


def test_triage_without_quick_tests_is_read_only(tmp_path):
    repo, run = _repo(tmp_path), _run(tmp_path)
    seen = {}

    def runner(command, prompt, cwd, timeout):
        seen.update(command=command, prompt=prompt)
        payload = {"is_error": False, "result": "# Fix list"}
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    triage(run, repo, "HEAD", quick_tests=False, runner=runner)
    assert "Edit" not in seen["command"] and "Write" not in seen["command"]
    assert "analysis only" in seen["prompt"]
    assert not (run / "fixes.patch").exists()
