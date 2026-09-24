@AGENTS.md

## Claude Code specifics

`AGENTS.md` above is shared with Codex; put anything that applies to both tools
there. This file holds only what is specific to Claude Code.

Project skills in `.claude/skills/` (keep this list in sync with the directory):

- `run-stack`: start the local stack, check the bridge and pages are alive, and
  tear it down. Use it before claiming a dashboard or embodiment change works.
- `decision-bench-annotate`: label decision_bench scenarios with the isolated
  Opus annotator, hand flagged ones to the human, apply reviewed labels.
- `scene-lab-director-run`: run a scene_lab director batch safely on the Mac
  mini and report it with the generated fix list (`fixes.md`).
- `demo-creation-video`: polished 30 fps demo clips from the bedroom recordings
  (not the 2 fps `tools/video_eval` review renders).

Tooling that calls Claude (the decision_bench annotator, the scene_lab person
and director) runs `claude -p` on the claude.ai subscription, never the API.
Long scene_lab runs can exhaust the subscription session limit, so check
before starting several in a row.

## Keeping instructions and skills current

Follow "Keeping these instructions current" in `AGENTS.md`; the same applies to
this file and to skills. In addition:

- If a workflow in a skill changes (a command, path, flag or output location),
  update the skill's `SKILL.md` in the same change. A skill with a stale
  command is followed confidently and fails.
- When you repeat a multi-step procedure with traps for the second time and no
  skill covers it, propose a skill for it rather than growing `AGENTS.md`.
- Remove a skill when the tool it drives is deleted or replaced, and drop it
  from the list above.
- Check what actually loads with `/context`, and prune with `/doctor` or
  `/memory` when this file or `AGENTS.md` drifts past about 200 lines.
