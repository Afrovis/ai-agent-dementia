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
   - `curl -s localhost:11434/api/ps` shows no model loaded, or unload it
     through the API.
   - `memory_pressure | tail -1` shows at least 50% free.
     `sysctl vm.swapusage` is not near its total.
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
3. **Watch without polling hard.** Every ~10 minutes check the log tail, the
   scenes finished (folders with `report.md`), `memory_pressure`, and swap.
   If swap climbs past about 80% or memory stays under 10% free, stop the
   batch with Ctrl-C semantics: `kill -INT` on the `scene_lab` Python process
   only, never Ollama. Report it.
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
   offer to implement the chosen fixes. Put the model back to its normal
   state (the batch unloads it at the end) and restart any stack you stopped
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
