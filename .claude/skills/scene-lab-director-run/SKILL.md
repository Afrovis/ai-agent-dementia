---
name: scene-lab-director-run
description: Run a scene_lab director batch (simulated nights with an Opus director and a Claude-played person against the real agent stack), keep the Mac mini safe while it runs, and report the result with the auto-generated fix list (fixes.md). Use whenever the user asks for a director hour/run/batch, a simulated night, "run scene_lab", "try a new director hour", to re-run or triage a finished scene_lab run, or to analyse what went wrong in one.
---

# scene_lab director run

`python -m scene_lab run --hours N` plays simulated nights against the real
agent stack (`nightsim` compose project). An Opus director picks each scene,
and a Claude "mind" plays the person through real audio. At the end, an Opus
triage agent writes `fixes.md`: a ranked fix list with proposals, backed by
quick tests it runs in a throwaway worktree. Background: `tests/scene_lab/README.md`,
the measured `tests/scene_lab/BASELINE.md`, and the Ollama note in `AGENTS.md` ("Operational gotchas").

All commands run from the checkout under test, with a Python environment
where scene_lab is installed with its `live` extra (README, top).

## Safety rules (the host is a 16 GB Mac mini)

- **Never kill, signal or restart Ollama.** A killed `ollama serve` leaves
  its old MLX runner holding GPU memory, and the next load kernel-panicked
  the machine on 2026-09-23. Free the model only through the API:
  `curl -s localhost:11434/api/generate -d '{"model":"gemma4:e4b-mlx","keep_alive":0}'`.
  If Ollama hangs, report what you see and ask.
- **One agent stack at a time.** Other compose projects with an `-agent-1`
  container (for example `majority-vote`) share Ollama and memory. Ask
  before stopping someone else's stack. Use `docker compose -p <name> stop`,
  never `down`, and say how to bring it back (`start`).
- **Subscription budget.** The director, the mind and the triage all run on
  the claude.ai subscription (`claude -p`, never the API). About an hour of
  talkative scenes can hit the session limit. The batch then stops by itself
  with one harness entry, and triage is skipped. Don't start a second batch
  on top of a limited one. Tell the user the reset time.

## Steps

1. **Pre-flight.** Check each of these and fix or ask before starting:
   - `docker compose ls`: no other agent stack running. Ask about any you find.
   - Flush the model: `curl -s localhost:11434/api/ps` shows no model
     loaded, or unload it through the API.
   - Docker disk space. Docker Desktop's VM disk is capped at 24 GiB, and
     every stack rebuild leaves untagged images behind. A full disk made the
     sim's Redis refuse writes (MISCONF) and cost the whole
     2026-09-23T2238-live batch. Check the free space with
     `docker run --rm alpine df -h / | tail -1`, then always clean up
     dangling data before a 1 h session:
     `docker image prune -f` (untagged images only) and
     `docker builder prune -f` (unused build cache). Never prune volumes,
     tagged images or other projects' containers, and never use
     `docker system prune -a`. Leave the VM's disk size and location as they
     are. If less than ~6 GB is free after the cleanup, stop and ask.
   - A `.env` exists in the checkout. Copy it from the main checkout if
     missing. Set `TZ` to the user's zone (America/New_York), because the sim
     otherwise speaks UTC hours.
   - `git log -1` shows the commit to be measured. The stack is built from
     this checkout.
   - Piper voices exist in `../data-ai-agent-dementia/models/piper-sim` (README).
2. **Start** in the background, logging to a file:
   `python -m scene_lab run --hours 1 > <log> 2>&1`. Options:
   `--max-scenes N`, `--no-triage`, and `--triage-no-tests` (analysis only).
   The run folder is the newest `*-live` under
   `../data-ai-agent-dementia/analysis/scene-lab/runs/`.
3. **Watch without polling hard.** Every ~10 minutes check the log tail and
   the scenes finished (folders with `report.md`). Don't watch memory or swap
   during the run: swap growth is normal and not a reason to stop. Memory is
   handled only by flushing the model at the start (step 1) and the end
   (step 7). If you must stop a batch for another reason, use Ctrl-C
   semantics: `kill -INT` on the `scene_lab` Python process only, never
   Ollama.
4. **After the batch** the run folder has `bugs.md`, per-scene
   `report.md`/`trace.jsonl`, `director.jsonl`, and `fixes.md` (plus
   `fixes.patch` if the triage edited code in its throwaway worktree). If
   triage failed or was skipped (`triage-error.txt`, usage limit), run it
   later with `python -m scene_lab triage <run-id>`, adding `--no-tests` for
   analysis only.
5. **Check the fix list before relaying it.** It is model output. For each
   item you pass on, open the linked `report.md#t=...` and the cited
   `file:line`, and confirm them. Say which quick tests were really run, and
   their numbers. `fixes.patch` is a suggestion from a throwaway copy: never
   apply it without the user asking.
6. **Compare with earlier runs.** `python -m scene_lab bugs <old-run> <new-run>`
   labels clusters gone, new or persisting. Check `BASELINE.md`'s open list so
   known items aren't reported as new. Note `contention` from `report.json`;
   latency counts only on an uncontended run.
7. **Report to the user.** Say what the run exercised: scenes, how many the
   director chose versus fallback, and any harness errors. Give the headline
   changes against the previous run and the top fix-list items with their
   evidence. List the owner decisions the list asks for as questions. Then
   offer to implement the chosen fixes. Flush the model: confirm
   `curl -s localhost:11434/api/ps` is empty (the batch unloads it at the
   end; if not, unload it through the API) and restart any stack you stopped
   if the user wants it back.

## What the triage agent may do

`scene_lab/triage.py` runs `claude -p --model opus` in a `git worktree add
--detach` copy of the run's commit, under the system temp folder. Allowed
tools: Read/Grep/Glob, Edit/Write inside that copy, Python only through a
wrapper pinned to the copy (pytest, `scene_lab rescore`, `scene_lab promote`
plus `session_replay run --llm recorded`), and read-only git. It has no
Docker, no Ollama, no network, and cannot commit. It runs at most five quick
tests. Its diff is saved as `fixes.patch`, and the worktree is removed. If you
widen these permissions, say so in the PR.
