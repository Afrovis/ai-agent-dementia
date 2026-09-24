# Handoff: scene_lab fixes still to finish and deploy (2026-09-23, evening)

Branch `scene-lab-fixes`, worktree
`/Users/mathiasserver/Documents/ai-agent-dementia/.claude/worktrees/scene-lab-fixes`,
PR #90. A Python env with everything installed is at `.venv/` in that worktree. Run tests with
`SCENE_LAB_RUNS=$PWD/.rescore/runs .venv/bin/python -m pytest -q --import-mode=importlib services/agent/tests tests/scene_lab/tests tests/session_replay tests/decision_bench/tests`.
`tests/dialogue_bench/tests/test_scenarios.py` already fails on main (`wants_bed` has no
scenario). It is unrelated.

## Done and pushed (in PR #90)

- Reassurance: two per escalation, then paced once a minute. A question, distress ≥ 2 or a
  stated need is urgent. The wording rotates, with no composed text while escalated.
- Check-ins every 120 s while escalated, if the person is present and not settled.
- Pain: one clear statement → "I'm sorry it hurts, Jean; I'm letting someone know now." plus an
  immediate alert.
- Restroom path: progress is acknowledged once. There are no directions on the floor, and
  escalated directions need toilet words.
- Vetoes `no_directions_from_floor` and `no_return_prompt_from_floor` (FALL-01).
- Reply routes for unmapped remarks: `validate_and_redirect` / `acknowledge_feeling`, `ask_need`
  and `caregiver_alerted`.
- Spoken time uses rotating night phrases, never clock times.
- "Speak up" repeats the last sentence louder and slower with its text on screen
  (`Say.emphasis="loud"`, embodiment soft-limit, −18.5 → −13.2 dBFS).
- Model-written text with quote marks is rejected.
- Ollama prefix-cache guard: the agent unloads daily while idle and re-warms 15 min before
  night. scene_lab resets the model per scene. See CLAUDE.md.
- scene_lab end-of-run triage (`fixes.md`, `fixes.patch`, `scene_lab triage RUN`), plus the
  skill `.claude/skills/scene-lab-director-run/`.
- Commit 862537b: the triage patch from run `2026-09-23T1854-live` (fix-list items 1a, 2, 3, 4,
  10 and the TT-3/TT-4 checker fixes).

## In flight when this was written (uncommitted, check first)

Two Codex jobs were running in the worktree. Their reports land in `.codex/out-*.md` and their
briefs are next to them. Check `git status` and `git diff`, review, run the tests, and commit
if they pass.

1. `.codex/brief-agent-round3.md` → `.codex/out-agent-round3.md`. Owner decisions, all approved:
   - Tests for 862537b.
   - **1b:** one extra `critical` Notify when distress ≥ 2 comes twice more, at least 60 s into
     an escalation.
   - **5:** agreeing with the directions is not `wants_bed`, and the hallway light stays on
     until the camera sees bed or `in_bed`.
   - **7:** place questions are `confused_time`. While escalated, "how much longer" gets a
     reassurance.
   - **9:** a direct reply waits 2 s after the previous sentence's estimated end, not 8 s from
     its start (`AGENT_REPLY_GAP_SECONDS`).
   - **11:** `AGENT_FLOOR_LIMIT_SECONDS=10`. A real-time critical alert comes only after 10 s on
     the floor. A shorter floor episode sends one `info` Notify, "Brief floor reading" (ntfy
     low priority, logged). Many tests used immediate floor escalation; the brief says to set
     `floor_limit_seconds=0` in those tests and to report any bench expectation that breaks.
2. `.codex/brief-checker-round3.md` → `.codex/out-checker-round3.md`, scene_lab only:
   - TT-1 labels "reply cancelled at start by barge-in" (major, or info if the conversation
     recovered).
   - Designed silence to distress ≥ 2 or a stated need is `review`, not `info`.

## Not started yet

3. `.codex/brief-talkover.md`, to run after job 1 is committed because it touches the same
   files. Scope:
   - Hold any non-terminal Say while the person is speaking: record `SpeechStarted`, and clear
     it on the Utterance or after 15 s.
   - A queued reply is superseded by the newer utterance's reply.
   - Fix false self-echo drops: content words only, overlap ≥ 0.6, at least 3 matching words.

   Run it with
   `codex exec -m gpt-6-sol -c model_reasoning_effort="medium" -C <worktree> -s workspace-write -o .codex/out-talkover.md - < .codex/brief-talkover.md`.
4. Update `tests/scene_lab/BASELINE.md` "Triage after the director hours" with the run
   `2026-09-23T1854-live` results: 13 scenes, no contention, 7 critical, 29 major, 37 review.
   Its fix list is at
   `../data-ai-agent-dementia/analysis/scene-lab/runs/2026-09-23T1854-live/fixes.md`. Mark
   items 1–11 fixed or open. Push, and update the PR #90 description.
5. Verify with a new director hour using the skill (`.claude/skills/scene-lab-director-run/`).
   It will be the first run with "speak up", the floor timing and the talk-over hold. Watch the
   poor-hearing and pain scenes in particular.

## Deploying (after the PR is reviewed and merged)

- Rebuild and restart `agent` and `embodiment`. Embodiment needs the rebuild for the loud
  rendering; `shared` changed (`Say.emphasis`), so rebuild every service that installs
  `nc_shared`, or just run `docker compose up --build`.
- `.env` in the main checkout is from 10 Sep and has no `TZ`, `AGENT_NIGHT_START/END` or
  `STRATEGIES_PATH`. Add `TZ=America/New_York`, or the agent runs its night window and spoken
  time on UTC. Copy the new keys from `.env.example`: `AGENT_FLOOR_LIMIT_SECONDS=10`, and
  `AGENT_REPLY_GAP_SECONDS` if job 1 lands.
- A caregiver `config/strategies.yaml`, if one exists, keeps working. The new reply-only
  strategies fall back to their code defaults. Restart `embodiment` so it pre-renders their
  phrases.
- The `majority-vote` stack (the other worktree) was stopped with
  `docker compose -p majority-vote stop` for the director hour. Bring it back with
  `docker compose -p majority-vote start` if it is still needed. It shares host Ollama, so keep
  only one agent stack up during scene_lab runs.
- Never kill or restart Ollama processes. Unload through the API:
  `curl -s localhost:11434/api/generate -d '{"model":"gemma4:e4b-mlx","keep_alive":0}'`. On
  2026-09-23 a killed server's orphaned runner plus a reload kernel-panicked the 16 GB Mac.

## Still open (not scheduled)

- Latency: re-measure on a clean run; the old numbers were taken while the host swapped.
- Claims in composed ENGAGED replies: a TT-2 judge pass over a live run.
- `validate_and_redirect` fallback on the restroom path.
- Restroom goal from the camera (perceive cadence).
- `compose.sim.yml` `TZ`.
- `plan` calls publish no Activity.
- Option 2 for Ollama: `mlx_lm.server` with a capped prompt cache.
