---
name: explainer-videos
description: Make and publish the silent, diagram-style explainer videos for the README and website of ai-agent-dementia (Night Companion) — "One night moment, as events" (agent ≠ LLM, events on Redis streams), "Simulated nights" (scene_lab: Opus directs, checks, triage, owner decides, fix lands) and any new one in the same paper-and-ink style — and put demo videos into the GitHub README with real inline players. Use this whenever the user asks for an explainer, architecture or "how it works" video, an animated diagram, a scene_lab / director / autonomous-testing video, wants an existing explainer re-rendered with new data, or wants any demo video (including the pose reels) added to the README, GitHub or the website, even if they don't say "skill" or "explainer".
---

# Explainer videos, and getting demo videos onto GitHub

Two kinds of video live in this project:

- **Pose demo reels** of real recordings (skeleton + state HUD). Those are made with
  the `demo-creation-video` skill; this skill only covers **publishing** them.
- **Explainers**: 1080p, 30 fps, silent, captioned animated diagrams drawn frame by
  frame with PIL. Every label, quote and number on screen is read from real data
  (a captured agent run, scene_lab run folders, git), never typed into the renderer.

**This skill maintains itself.** Before finishing a task that used it, update the
log and pitfalls below (see "Keeping this skill current").

## Where things live

| What | Path |
|---|---|
| Renderer scripts (canonical, in git) | `.claude/skills/explainer-videos/scripts/` |
| Shared drawing kit (palette, fonts, Canvas, wires, chips, encode) | `scripts/motion.py` |
| Work/output root (`DEMO_WORK`) | `<data>/analysis/demo-videos/` |
| Event explainer data + output | `<work>/event-system-2026-10-01/` (`script_case.json`, `case_try*.json`, `case.json`, mp4, `poster.png`) |
| scene_lab explainer data + output | `<work>/scene-lab-loop-2026-10-01/` (`loop_facts.json`, mp4, `poster.png`) |
| Thumbnail frames for the event explainer | `<work>/pipeline-case-2026-09-18/video-02/_clip_frames/{up,floor}` (`EXPLAINER_CLIPS`) |
| Fonts (Newsreader, Instrument Sans, JetBrains Mono; OFL) | `<work>/fonts/` |
| scene_lab runs | `<data>/analysis/scene-lab/runs/<run-id>/` |
| Published copies | repo `docs/media/` (`*.mp4`, `*.webp`, `*.png`) |
| Python env | `.venv-video-eval/bin/python` in the main checkout (PIL 12 with WebP) |

`<data>` = `../data-ai-agent-dementia` beside the main checkout; `motion.py` finds it by
walking up from the script (works from worktrees) or takes `NC_DATA`.

## The house style

- Warm paper ground `GROUND`, paper cards with thin `LINE` outlines, INK text; WARM
  (amber) marks attention, ALERT (brick) marks an alert or a wrong answer, COOL (blue)
  marks live input, the model and good outcomes. All constants are in `motion.py`.
- Type: Newsreader serif 46 for the beat heading at (77, 84); mono 15 kicker above it
  (`NIGHT COMPANION  ·  <clock or run>`); Instrument Sans for body; JetBrains Mono for
  ids, code and event names; serif italic for anything spoken to the person.
- Motion: ghost outlines first, then staggered `ease(prog(t, a, b))` reveals, word-by-word
  quotes, crossfading headings per beat; global 0.4 s fade in, 0.6 s fade out.
- Footer y=1030: left says what is real and what is scripted/simulated, right says
  `not a medical device`. Keep it honest; that line is what makes the video credible.
- Events are pills (`motion.chip`) that slide along a wire, then **hold ≥ 1 s** so they
  can be read (`travelling(..., stop=0.5)` parks a pill mid-gap on short wires).

## Rules for what goes on screen

1. Every quote, id, number and code line comes from a data file the renderer loads
   (`case.json`, `loop_facts.json`). Choose, shorten by whole sentences, never paraphrase
   inside quotation marks. Labels you write yourself are fine ("caregiver's fallback").
2. Show behaviour as it is on the code being described. The 2026-09-18 hub video said a
   reply was "written by the LLM"; on today's code that reply is a fixed line the agent
   picks. Re-capture when the agent changes.
3. The agent is **not** the LLM: draw the state machine as the hero box with the model
   nested inside it; safety paths go *through* the agent box with the model dimmed
   (`not called`), never around it.
4. Never say "clinically validated". Where rules come from: the guideline pack
   (`tests/decision_bench/guidelines.md`, clauses ticked by a person), human-reviewed
   labels, caregiver settings, owner decisions. Don't tie a fix to a source it doesn't
   support (the floor veto is filed under FALL-01, but FALL-01's evidence is about
   alerting, not wording).
5. No pronouns for real people unless they're stated; "the person" works.
6. Any quote must stay fully visible ≥ 2 s before it changes; nothing overlaps; text
   wraps or ellipsizes inside its card.

## Workflow: the event explainer ("One night moment, as events")

1. **Capture a case on the code being shown** (Ollama must be idle; never restart it):
   ```sh
   W=<work>/event-system-2026-10-01; WT=<checkout of the commit to show>
   cd $WT && PYTHONPATH=$WT/services/agent:$WT/shared \
     <main>/.venv-dialogue/bin/python <skill>/scripts/capture_case.py \
     --script $W/script_case.json --out $W/case_tryN.json
   ```
   It drives the real `Session` + `run_once` + `OllamaLLM` (gemma4:e4b-mlx) on a
   `FakeBus` and logs inputs, phases, every LLM call and every published event.
2. **Pick a take.** The model is stochastic; run 6–8 tries and keep one that shows all
   three beats. The rejected-draft beat (draft contains "but" → `validate_composition`
   rejects → caregiver fallback) appeared in 2 of 8 runs. Copy it to `case.json` and say
   in the README that it is one real run.
3. **Render stills and look at them** (`render_events.py case.json out.png SECONDS`),
   then the full video (`render_events.py case.json out.mp4`, a few minutes on 8 workers).

Script timings that matter: the second utterance must come > 60 s after the first, or
the agent stays silent on purpose (`validated_recently`); the floor reading must come
before the ladder runs out (≈t 92 s in a 110 s script), or the escalation reason is
`strategies_exhausted` instead of rule 5. Rule 5 holds `on_floor` 10 s before escalating.

## Workflow: the scene_lab explainer ("Simulated nights")

1. `python <skill>/scripts/extract_facts.py [out.json]` reads the run folder (scene card,
   director rationale, trace, bugs.jsonl, fixes.md), git (fix commit, veto snippet,
   `origin/main:tests/decision_bench/guidelines.md`) and counts before/after floor
   prompts across runs. Edit the constants at its top (`RUN`, `SCENE`, `FIX_COMMIT`,
   `BEFORE`, `AFTER`) to tell a different bug's story; check the counts by hand once.
2. `render_loop.py loop_facts.json out.png SECONDS` for stills, then the mp4 (74 s).

Roles must match the run shown. On `main` (2026-09-23 runs) the scene **director is
Opus** (`director.py` `model="opus"`), the person is Sonnet, triage is one Opus agent in
a throwaway worktree. The Sonnet director and split triage exist only on the
`scene-lab-fixes` branch (PR #92) — check before reusing the labels.

## Workflow: a new explainer

1. Read the relevant code and data yourself first; decide the beats and the one real
   example that carries them. Ask the user what was wrong with any earlier version.
2. Write a fact extractor or capture so the renderer only reads JSON.
3. Large renderers can go to Codex (`codex exec -m gpt-6.1-sol ... --skip-git-repo-check
   -C <work>`; the work dir is not a git repo) with a brief giving layout coordinates,
   per-beat timings, exact data fields and a list of stills to check. Copy the brief style
   from `<work>/*/BRIEF.md`.
4. **Review the stills yourself.** Codex's first drafts here needed: wires that detoured
   around boxes (use straight wires and hub-style beziers into a box's top edge); a
   gate lit up that never acted in the data; an output box that went blank between beats
   (keep the last line, dimmed); event pills on screen < 1 s; pills parking on text.
5. Encode with `motion.encode` (x264 CRF 16, yuv420p, faststart); `ffprobe` the result
   and pull frames from the **encoded file** for a final contact sheet.

## Publishing to the README (GitHub)

GitHub plays inline video only from `user-attachments` uploads; an MP4 in the repo
opens as a download. So each video gets both:

1. **Repo copies** in `docs/media/`: `<name>.mp4` (≤ 10 MB; re-encode a big reel with
   `ffmpeg -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -movflags +faststart -an`),
   `<name>.webp` (`scripts/to_webp.py in.mp4 out.webp 1280 12 70`, 4–6 MB, an animated
   fallback that GitHub shows inline from the repo) and `<name>.png` poster.
2. **An attachment**: open the PR in Claude in Chrome (the user must be logged in), find
   the comment box's hidden file input, `file_upload` the MP4 (< 10 MB per call), read the
   `https://github.com/user-attachments/assets/<uuid>` link from the textarea with
   `javascript_tool`, then clear the textarea. The link 404s for anonymous visitors until
   it appears in a **posted** comment: post it with `gh pr comment <n>` (ask the user
   first) and check `curl -sL -o /dev/null -w "%{http_code} %{size_download}"` gives 200 and
   the file's exact size.
3. In the README the bare link on its own line becomes a player; follow it with a line
   linking the repo `.webp` and `.mp4`. Merge the PR: the repo homepage shows `main`.
   Verify logged out: `curl -sL https://github.com/Afrovis/ai-agent-dementia | grep -c '<video'`.

Footage of a person may only be published with that person's approval (the owner
approved bedroom-sample-02 on 2026-10-01). Keep the README's privacy paragraph true.

## Pitfalls met so far

- `capture_case.py` imports the agent from `PYTHONPATH`; check `agent.__file__` points at
  the checkout you mean to show, not an editable install of another branch.
- "I have to go and look for him" matches the toilet regex (`have to go`) and vetoes
  `guided_return`; avoid that phrasing in scripts (and fix it in the agent some day).
- zsh: `echo ===` fails (equals expansion); quote it.
- ffmpeg here has no `drawtext` and no WebP encoder; use PIL for both.
- Unposted `user-attachments` uploads work only for the uploader.

## Log

| Video | Data | Published |
|---|---|---|
| What the camera sees (perception reel, 35 s) | `2026-09-13_bedroom-sample-02/reel-30fps.mp4` (demo-creation-video), CRF 22 copy | README, PR #95 |
| One night moment, as events (62 s) | `event-system-2026-10-01/case.json` = try 10, agent at 62c9c51 | README, PR #94 |
| Simulated nights (74 s) | `scene-lab-loop-2026-10-01/loop_facts.json`, run 2026-09-23T1854-live, fix 862537b | README, PR #94 |

## Keeping this skill current

At the end of a task that used this skill, edit this file on a branch and merge it like
any other change: add rows to **Log**, add new failure modes to **Pitfalls**, edit the
canonical scripts in `scripts/` (never only the copies in `<work>`), and update **Where
things live** first if paths moved. Keep it under ~250 lines. Commit scripts and this
file only; frames, case captures and rendered videos stay in `<data>` (published copies
go to `docs/media/` only when the user asks for them on GitHub).
