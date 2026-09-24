"""End-of-run triage with a fake Claude: plan, serial workers, write, cleanup."""

import json
import subprocess

from scene_lab import gpu
from scene_lab.triage import prepare_clusters, triage

IDLE = {"idle": True, "device_utilization_pct": 0, "waited_s": 0}


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


def _bug(scene, t, fingerprint, severity):
    return {
        "run": "2026-09-23T2000-live",
        "scene": scene,
        "t": t,
        "check": fingerprint.split("|")[0],
        "severity": severity,
        "origin": "agent",
        "fingerprint": fingerprint,
        "summary": f"{fingerprint} at {t}",
        "evidence": [f"Utterance@{t}: where is the loo"],
        "report": f"{scene}/report.md#t={t:g}",
        "commit": "-",
        "model": "-",
    }


def _run(tmp_path):
    run = tmp_path / "runs" / "2026-09-23T2000-live"
    (run / "restroom-1").mkdir(parents=True)
    (run / "restroom-1" / "report.md").write_text(
        "# restroom-1\n\n"
        '<a id="t=1"></a> 1s PersonState {"state": "standing"}\n'
        '<a id="t=40"></a> 40s Utterance {"text": "where is the loo"}\n'
        '<a id="t=45"></a> 45s Say {"text": "The bathroom is through the door."}\n'
        '<a id="t=90"></a> 90s PersonState {"state": "in_bed"}\n'
    )
    rows = [
        _bug("restroom-1", 40, "TT-1|ESCALATED|wait_for_caregiver|x|reassured_recently", "review"),
        _bug("restroom-1", 40, "SM-3|ENGAGED|restroom|acknowledge_progress|-", "major"),
        _bug("restroom-1", 45, "SM-3|ENGAGED|restroom|acknowledge_progress|-", "major"),
        _bug("restroom-1", 45, "TT-3|ENGAGED|restroom|path_light|-", "critical"),
    ]
    (run / "bugs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "director.jsonl").write_text(
        json.dumps({"scene": "restroom-1", "rationale": "probe the path"}) + "\n"
    )
    return run


def _done(command, result=None, structured=None, **usage):
    payload = {
        "is_error": False,
        "result": result or "",
        "usage": {"input_tokens": 10, "output_tokens": 5, **usage},
        "modelUsage": {command[command.index("--model") + 1]: {}},
        "total_cost_usd": 0.01,
        "num_turns": 3,
    }
    if structured is not None:
        payload["structured_output"] = structured
    return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


FINDING = {
    "kind": "agent_bug",
    "cause": "progress remarks are not routed",
    "cause_location": "services/agent/agent/main.py:1",
    "evidence": ["restroom-1/report.md#t=40 'where is the loo'"],
    "proposal": "raise MAX_REASSURANCES",
    "quick_test": {"ran": True, "command": "pytest -k x", "result": "3 passed"},
    "owner_decision": "none",
    "confidence": "high",
}


def _plan(ids):
    return {
        "summary": "restroom path",
        "investigations": [
            {
                "title": "Path light",
                "clusters": [ids[0], "c99"],
                "question": "Why no light?",
                "start_files": ["services/agent/agent/main.py"],
            },
            {
                "title": "Progress remarks",
                "clusters": [ids[1]],
                "question": "Why silence?",
                "start_files": [],
            },
        ],
        "set_aside": [{"clusters": [ids[2]], "reason": "designed pacing"}],
    }


def test_prepare_clusters_ranks_and_cuts_a_window(tmp_path):
    clusters = prepare_clusters(_run(tmp_path))
    assert [c["check"] for c in clusters] == ["TT-3", "SM-3", "TT-1"]
    assert [c["id"] for c in clusters] == ["c1", "c2", "c3"]
    sm3 = clusters[1]
    assert sm3["count"] == 2 and sm3["severity"] == "major"
    # 40 s occurrence: lines from 25 s to 50 s only.
    assert any("where is the loo" in line for line in sm3["window"])
    assert not any("in_bed" in line or "standing" in line for line in sm3["window"])


def test_triage_plans_then_investigates_in_series_then_writes(tmp_path):
    repo, run = _repo(tmp_path), _run(tmp_path)
    calls = []
    gates = []

    def gate():
        gates.append(len(calls))
        return IDLE

    def runner(command, prompt, cwd, timeout):
        model = command[command.index("--model") + 1]
        calls.append({"command": command, "prompt": prompt, "cwd": cwd, "model": model})
        if len(calls) == 1:
            assert "--tools" in command and "--json-schema" in command
            return _done(command, structured=_plan(["c1", "c2", "c3"]), cache_read_input_tokens=7)
        if "--json-schema" in command:
            # A quick test edits the throwaway copy only.
            (cwd / "services/agent/agent/main.py").write_text(
                f"MAX_REASSURANCES = {len(calls) + 8}\n"
            )
            return _done(command, structured=FINDING)
        return _done(command, result="# Fix list: 2026-09-23T2000-live\n\n## 1. Path light")

    fixes = triage(run, repo, "HEAD", runner=runner, gpu_gate=gate)

    assert [c["model"] for c in calls] == ["opus", "sonnet", "sonnet", "opus"]
    plan_prompt, first, second, write = (c["prompt"] for c in calls)
    assert "SM-3|ENGAGED|restroom" in plan_prompt and "probe the path" in plan_prompt
    assert "where is the loo" in plan_prompt  # evidence window for a major cluster
    # Each worker sees only its own clusters.
    assert "TT-3|ENGAGED|restroom|path_light" in first and "SM-3" not in first
    assert "SM-3|ENGAGED|restroom" in second and "path_light|-:" not in second
    assert "`.env`" in first and str(run.resolve()) in first
    assert "Edit" in calls[1]["command"] and not any("docker" in c for c in calls[1]["command"])
    # The GPU gate runs right before each worker, never before plan or write.
    assert gates == [1, 2]
    assert "raise MAX_REASSURANCES" in write and "designed pacing" in write
    assert "Edit" not in calls[3]["command"]
    assert fixes.read_text().startswith("# Fix list: 2026-09-23T2000-live")
    # One patch per item, from a reset copy each time.
    first_patch = (run / "triage/1.patch").read_text()
    assert "-MAX_REASSURANCES = 2" in first_patch and "+MAX_REASSURANCES = 10" in first_patch
    # The second item starts from a clean copy, not from the first item's edit.
    assert "-MAX_REASSURANCES = 2" in (run / "triage/2.patch").read_text()
    assert "# item 2: Progress remarks" in (run / "fixes.patch").read_text()
    plan = json.loads((run / "triage/plan.json").read_text())
    assert plan["investigations"][0]["clusters"] == ["c1"]  # unknown c99 dropped
    usage = [json.loads(line) for line in (run / "usage.jsonl").read_text().splitlines()]
    assert [u["role"] for u in usage] == [
        "triage-plan",
        "triage-worker",
        "triage-worker",
        "triage-write",
    ]
    assert usage[0]["cache_read"] == 7 and usage[1]["gpu_idle"] is True
    # The real checkout is untouched and the throwaway worktree is gone.
    assert (repo / "services/agent/agent/main.py").read_text() == "MAX_REASSURANCES = 2\n"
    assert not calls[1]["cwd"].exists()
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list"], capture_output=True, text=True
    ).stdout
    assert len(listed.strip().splitlines()) == 1


def test_triage_without_quick_tests_is_read_only(tmp_path):
    repo, run = _repo(tmp_path), _run(tmp_path)
    workers = []

    def runner(command, prompt, cwd, timeout):
        if "--tools" in command:
            return _done(command, structured=_plan(["c1", "c2", "c3"]))
        if "--json-schema" in command:
            workers.append((command, prompt))
            return _done(command, structured=FINDING)
        return _done(command, result="# Fix list")

    triage(run, repo, "HEAD", quick_tests=False, runner=runner, gpu_gate=lambda: IDLE)
    command, prompt = workers[0]
    assert "Edit" not in command and "Write" not in command
    assert "Analysis only" in prompt
    assert not (run / "fixes.patch").exists()


def test_triage_usage_limit_stops_workers_and_renders_without_write(tmp_path):
    repo, run = _repo(tmp_path), _run(tmp_path)
    calls = []

    def runner(command, prompt, cwd, timeout):
        calls.append(command)
        if "--tools" in command:
            return _done(command, structured=_plan(["c1", "c2", "c3"]))
        payload = {
            "is_error": True,
            "result": "You've hit your session limit · resets 6pm (America/New_York)",
            "usage": {"input_tokens": 1, "cache_read_input_tokens": 4000, "output_tokens": 9},
            "total_cost_usd": 0.5,
        }
        return subprocess.CompletedProcess(command, 1, json.dumps(payload), "")

    fixes = triage(run, repo, "HEAD", runner=runner, gpu_gate=lambda: IDLE)
    assert len(calls) == 2  # plan and the first worker; no second worker, no write
    text = fixes.read_text()
    assert "Written without the final pass: usage limit" in text
    assert "Not investigated" in text and "designed pacing" in text
    usage = [json.loads(line) for line in (run / "usage.jsonl").read_text().splitlines()]
    assert usage[1]["ok"] is False and usage[1]["cache_read"] == 4000


def test_triage_falls_back_when_the_planner_fails(tmp_path):
    repo, run = _repo(tmp_path), _run(tmp_path)
    prompts = []

    def runner(command, prompt, cwd, timeout):
        if "--tools" in command:
            return subprocess.CompletedProcess(command, 1, "", "boom")
        prompts.append(prompt)
        if "--json-schema" in command:
            return _done(command, structured=FINDING)
        return _done(command, result="# Fix list")

    triage(run, repo, "HEAD", runner=runner, gpu_gate=lambda: IDLE)
    plan = json.loads((run / "triage/plan.json").read_text())
    # Non-review clusters only, worst first; review goes to set aside.
    assert [i["clusters"] for i in plan["investigations"]] == [["c1"], ["c2"]]
    assert plan["set_aside"][0]["clusters"] == ["c3"]
    assert len(prompts) == 3


def test_gpu_gate_waits_for_idle_then_gives_up_flagged():
    readings = iter([80, 50, 10])
    now = [0.0]

    def snap():
        return {"device_utilization_pct": next(readings), "ollama_models": []}

    def sleep(s):
        now[0] += s

    state = gpu.wait_idle(timeout_s=300, poll_s=15, snap=snap, sleep=sleep, clock=lambda: now[0])
    assert state["idle"] and state["waited_s"] == 30

    busy = gpu.wait_idle(
        timeout_s=30,
        poll_s=15,
        snap=lambda: {"device_utilization_pct": 90},
        sleep=sleep,
        clock=lambda: now[0],
    )
    assert not busy["idle"] and "90%" in busy["busy"]
