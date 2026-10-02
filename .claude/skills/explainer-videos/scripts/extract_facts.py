"""Collect every fact the scene_lab loop explainer shows into loop_facts.json.

Reads the real run folders, the fix list and git; nothing quoted in the video is typed by hand.
Usage: python extract_facts.py [out.json]  (default WORK/scene-lab-loop-2026-10-01/loop_facts.json)
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from motion import DATA, WORK  # noqa: E402

RUNS = DATA / "analysis/scene-lab/runs"
REPO = HERE  # git finds the repository this skill lives in
RUN = "2026-09-23T1854-live"
SCENE = "fall-poor-hearing-quiet-floor-2"
FIX_COMMIT = "862537b"
BEFORE = ["2026-09-23T0552-live", "2026-09-23T1854-live"]
AFTER = ["2026-09-23T2238-live", "2026-09-24T1311-live", "2026-09-24T1435-live", "2026-09-25T0600-live"]
TRIAGED = [RUN, *AFTER]
RETURN_PROMPTS = {"acknowledge_return", "guided_return"}
GUIDE_SOURCES = [("NICE", "NICE guideline NG97"), ("AA", "Alzheimer's Association guidance"),
                 ("VAL", "Validation (Cochrane review)"), ("PCC", "Person-centred care (Kitwood)"),
                 ("DICE", "DICE approach"), ("FALL", "Falls study (BMJ 2008)"),
                 ("TOIL", "Night-time toileting")]


def jl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def floor_prompts(run):
    scenes = hits = affected = 0
    for tr in glob.glob(str(RUNS / run / "*/trace.jsonl")):
        state, seen, n = None, False, 0
        for e in jl(tr):
            if e["type"] == "PersonState":
                state = e["data"]["state"]
                seen |= state == "on_floor"
            if e["type"] == "Say" and state == "on_floor" and e["data"].get("strategy") in RETURN_PROMPTS:
                n += 1
        scenes += seen
        hits += n
        affected += n > 0
    return {"floor_scenes": scenes, "prompts": hits, "scenes_affected": affected}


def main():
    import yaml

    run_dir = RUNS / RUN
    card = yaml.safe_load(open(run_dir / SCENE / "scene.yaml"))
    director = next(d for d in jl(run_dir / "director.jsonl") if d.get("scene") == SCENE)
    trace = jl(run_dir / SCENE / "trace.jsonl")
    lines, state = [], None
    for e in trace:
        d = e["data"]
        if e["type"] == "PersonState":
            state = d["state"]
        if e["type"] == "Utterance":
            lines.append({"t": round(e["t"], 1), "who": "person", "text": d["text"], "state": state})
        elif e["type"] == "Say":
            lines.append({"t": round(e["t"], 1), "who": "agent", "text": d["text"],
                          "strategy": d.get("strategy"), "state": state})
        elif e["type"] == "Notify":
            lines.append({"t": round(e["t"], 1), "who": "notify", "level": d["level"], "state": state})
    flags = [{"t": round(b["t"], 1), "check": b["check"], "severity": b["severity"], "summary": b["summary"],
              "evidence": b.get("evidence", [])}
             for b in jl(run_dir / "bugs.jsonl") if b["scene"] == SCENE]

    fixes = open(run_dir / "fixes.md").read()
    items = re.findall(r"^## (\d+)\. (.+)$", fixes, re.M)
    item2 = re.search(r"^## 2\. .*?(?=^## 3\.)", fixes, re.M | re.S).group(0)
    field = lambda name: re.search(rf"\*\*{name}[^*]*:\*\*(.*?)(?=\n- \*\*|\Z)", item2, re.S).group(1).strip()
    owner = {n: re.search(rf"^## {n}\. .*?\*\*Owner decision:\*\*\s*(.*?)(?=^## |\Z)", fixes, re.M | re.S)
             .group(1).strip().split("\n")[0] for n, _ in items}

    veto_src = git("show", f"{FIX_COMMIT}:services/agent/agent/veto.py").splitlines()
    i = next(k for k, ln in enumerate(veto_src) if "no_return_prompt_from_floor" in ln)
    snippet = veto_src[i - 2:i + 4]

    totals = {"director_hours": len(TRIAGED), "scenes": 0, "bug_entries": 0, "fix_items": 0}
    for r in TRIAGED:
        totals["scenes"] += len(glob.glob(str(RUNS / r / "*/report.md")))
        totals["bug_entries"] += len(jl(RUNS / r / "bugs.jsonl"))
        totals["fix_items"] += len(re.findall(r"^## \d+\. ", open(RUNS / r / "fixes.md").read(), re.M))

    run_index = {d["id"]: d for d in jl(RUNS / "index.jsonl")}[RUN]
    facts = {
        "run": RUN, "run_commit": run_index["commit"][:7], "agent_model": run_index["model"],
        "run_scenes": run_index["scene_count"], "run_bug_entries": len(jl(run_dir / "bugs.jsonl")),
        "roles": {"director": "Claude Opus", "person": "Claude Sonnet", "triage": "Claude Opus",
                  "agent": run_index["model"] + " (local)"},
        "scene": {"id": card["id"], "category": card["category"], "persona_tag": card["persona_tag"],
                  "stressors": card["stressors"], "summary": card["persona"]["summary"],
                  "hearing": card["persona"]["hearing"], "director_rationale": director["rationale"]},
        "transcript": lines, "flags": flags,
        "fix_items": [{"n": int(n), "title": t, "owner": owner[n]} for n, t in items],
        "fix2": {"title": items[1][1], "evidence": field("Evidence"), "cause": field("Cause"),
                 "proposal": field("Proposal"), "quick_test": field("Quick test"), "owner": owner["2"]},
        "commit": {"sha": FIX_COMMIT, "subject": git("show", "-s", "--format=%s", FIX_COMMIT).strip(),
                   "date": git("show", "-s", "--format=%ad", "--date=short", FIX_COMMIT).strip(),
                   "veto_snippet": snippet},
        "before": {"runs": BEFORE, **{k: sum(floor_prompts(r)[k] for r in BEFORE)
                                     for k in ("floor_scenes", "prompts", "scenes_affected")}},
        "after": {"runs": AFTER, **{k: sum(floor_prompts(r)[k] for r in AFTER)
                                   for k in ("floor_scenes", "prompts", "scenes_affected")}},
        "totals": totals,
        "invariants": ["TT-1", "TT-2", "TT-3", "TT-4", "TT-5", "TT-6", "TT-7",
                       "SM-1", "SM-2", "SM-3", "SM-4", "SM-5", "TM-1", "TM-2", "TM-3"],
    }
    guide = git("show", "origin/main:tests/decision_bench/guidelines.md")
    clause_ids = re.findall(r"^### ([A-Z]+-\d+)", guide, re.M)
    facts["guidelines"] = {
        "clauses": len(clause_ids),
        "checked": len(re.findall(r"\*\*Checked:\*\* \[[xX]\]", guide)),
        # Short display names for the source families, keyed by clause prefix.
        "sources": [name for prefix, name in GUIDE_SOURCES
                    if any(c.startswith(prefix + "-") for c in clause_ids)],
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else WORK / "scene-lab-loop-2026-10-01/loop_facts.json"
    json.dump(facts, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
