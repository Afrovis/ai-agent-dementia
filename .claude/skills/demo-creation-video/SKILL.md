---
name: demo-creation-video
description: Make polished 30 fps demo videos from the bedroom recordings of the ai-agent-dementia project — two-stage yolo11x-pose on the 4K source, smoothed skeleton, "Direction A" overlay (state label, truth-vs-seen timeline, bed zone, on-the-floor chip) — and save them in the right place. Use this whenever the user asks for a demo video, demo clip, snippet, reel, showcase or "nice-looking" pose-estimation video of a bedroom sample (video 1–5, bedroom-sample-0N), wants to re-render or restyle existing demo clips, or wants to add a new snippet, even if they don't say "skill" or "Direction A". Not for the evaluation review videos in tools/video_eval (those are 2 fps diagnostic renders).
---

# Demo videos of the bedroom samples

This skill turns a time range of a bedroom recording into a demo clip: the real
footage, dimmed, with a clean yolo11x-pose skeleton and a quiet HUD showing what
the perception pipeline concluded. It exists because the review videos from
`tools/video_eval` are 2 fps diagnostics on 320×240 bridge frames; demos need the
full source and 30 fps.

**This skill maintains itself.** Before finishing any task that used it, update
this file (see "Keeping this skill current"). Paths, pitfalls and the snippet log
below are only trustworthy if every run writes back what it learned.

## Where things live

| What | Path |
|---|---|
| Scripts (canonical, in git) | `.claude/skills/demo-creation-video/scripts/{infer,render}.py` |
| Work/output root (`DEMO_WORK`) | `<data>/analysis/demo-videos/` |
| Per-clip outputs | `<work>/<clip_id>/<snippet>.pose.jsonl`, `<snippet>-30fps.mp4`, `reel-30fps.mp4` |
| Fonts (Newsreader, Instrument Sans, JetBrains Mono; OFL, Google Fonts) | `<work>/fonts/` |
| Source videos | `<data>/raw/<clip.yaml source>` |
| Clip metadata, predictions, reference labels | `<data>/clips/<clip_id>/{clip.yaml,predictions/,labels/reference.yaml}` |
| Calibrated bed zones (bridge-letterbox coords) | `<data>/analysis/walking-bed-2026-09-16/zones/<clip_id>.yaml` |
| Models | `<data>/models/yolo11m-pose.pt`, `yolo11x-pose.pt` |
| Python env | `.venv-video-eval/bin/python` in the main checkout, not in worktrees (ultralytics, torch MPS, PIL, cv2, yaml) |
| Design mockups (canvases) | Direction A on real footage: https://claude.ai/artifact/XykN9d1DpLMBEy4eWSB1NP · first exploration: https://claude.ai/artifact/SzRRdv2hNUJx22Ro7FJyBk · generators in `<data>/analysis/demo-overlay-mockup/` |

`<data>` = `../data-ai-agent-dementia` beside the main checkout; the scripts find
it by walking up from their own location, or take `NC_DATA`. Everything under
`<data>` is private footage: it lives outside the git repo and never gets
committed, and only the skill, never a frame or a rendered clip, belongs in git.
Uploading frames anywhere (artifacts, Codex) needs approval from the person in
the footage; the video 2 mockups were approved by them on 2026-09-18.

## Workflow

1. **Pick snippets.** Read `clips/<clip>/labels/reference.yaml` (and the `script`
   in `clip.yaml`) for what happens when. Good demo snippets are 7–15 s, contain a
   state change, and show the full body. Before committing to a moment, look at a
   contact sheet of candidate frames (cv2 → tiles → Read the image): walking
   footage is often motion-blurred, and legs leave the frame near the camera.
2. **Infer** (≈0.3 s/frame on MPS; 15 s ≈ 2.5 min). Run in the background for
   several snippets:
   ```sh
   REPO=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")  # main checkout
   PY=$REPO/.venv-video-eval/bin/python
   SK=$(git rev-parse --show-toplevel)/.claude/skills/demo-creation-video/scripts
   $PY $SK/infer.py 2026-09-13_bedroom-sample-02 floor 95 110
   ```
3. **Render** (≈1 min per 15 s; restyling only needs this step):
   ```sh
   $PY $SK/render.py 2026-09-13_bedroom-sample-02 walk bed floor
   ```
   It prints which prediction file, reference labels and bed zone it found. It
   picks `predictions/replay_*improved*.jsonl`, else the newest `*yolo*-g.jsonl`
   with a `state` field; override with `--pred`. `--label` changes the title.
4. **Check before handing over.** Pull 3–6 stills across each clip into a contact
   sheet and look at them: skeleton on the body (not the mirror), HUD legible, state
   label plausible. `ffprobe` for 30/1 fps and frame count.
5. **Reel** (optional):
   ```sh
   cd <work>/<clip>; printf "file '%s'\n" walk-30fps.mp4 bed-30fps.mp4 > r.txt
   ffmpeg -f concat -safe 0 -i r.txt -c copy reel-30fps.mp4 && rm r.txt
   ```
6. Send the reel/clips with SendUserFile if the user may be away, and update this
   skill.

## How the render looks, and why

- **Pose:** yolo11m at 1280 px finds the person on the full frame; yolo11x-pose at
  960 px runs on that box padded 25%. On 4K this beats yolo11s on bridge frames
  by a wide margin (floor-sit mean joint confidence ~0.95 vs visibly misplaced).
- **Person choice = largest box**, not most confident: the video 2 mirror gives a
  confident reflection.
- **Smoothing:** One Euro filter per coordinate (min_cutoff 1.2, beta 0.02, px
  units); confidence EMA; joints under 0.2 hold position; the skeleton fades in
  0.25 s / out 0.3 s when detection drops.
- **Look (Direction A):** video dimmed ~38% plus vignette and a bottom scrim; warm
  = person's left, cool = right; dark under-stroke so limbs read on bright walls;
  opacity follows confidence, <0.35 becomes dotted; neck + nose mark instead of a
  head ring (a ring over a real face looks odd); 0.5 s wrist/ankle trails from
  joints >0.6 only; bed zone dotted and faint when idle, filled in the state colour
  when the pipeline says `zone: bed`.
- **HUD is honest:** state, confidence, zone and the SEEN lane are the real 2 fps
  pipeline output, so labels lag transitions slightly and may disagree with what
  you see (e.g. "Standing" at the start of a walk). TRUTH is `reference.yaml`. The
  floor chip only states the duration; don't add agent speech or caregiver alerts
  unless they come from a real agent replay.
- **Drawing:** PIL on a 4K canvas, downsampled with LANCZOS → anti-aliased 1080p;
  x264 CRF 16, preset slow, 0.4 s fades.
- **Coordinates:** pose JSONL is normalised to the source frame. Bed zones and
  bridge predictions are normalised to the 4:3 letterboxed bridge frame; the
  renderer undoes that from the source aspect. Non-16:9 sources are fitted with
  side bars, not stretched.

## Pitfalls met so far

- The bridge-letterbox offset: forgetting it draws zones/bridge keypoints ~10% too
  high on 16:9 video (it went unnoticed on a schematic background).
- Highest-confidence detection picked the mirror reflection (video 2, 1:34).
- Motion blur during walking collapses pose; sharpest-frame search barely helps
  on this footage, choose moments where the person is side-on and fully in frame.
- ffmpeg here has no `drawtext`; do text with PIL/cv2.
- Keep the 60 fps source → 30 fps by reading one frame and `grab()`-ing the rest;
  seeking per frame is slow on HEVC.

## Snippet log

| Clip | Snippet | Range (s) | Notes |
|---|---|---|---|
| 2026-09-13_bedroom-sample-02 | walk | 29.5–37 | crossing the room; label reads Standing → Walking at 33.5 |
| 2026-09-13_bedroom-sample-02 | bed | 63–75 | getting into bed; bed zone lights up |
| 2026-09-13_bedroom-sample-02 | floor | 95–110 | sits on floor at ~98, up at ~107.5; floor chip |
| 2026-09-13_bedroom-sample-02 | reel | — | walk + bed + floor, 34.5 s; in the README since PR #95 (publishing: `explainer-videos` skill) |
| 2026-09-20_living-room-sample-01 | floor_sequence | 13–27 | local VLM says on-floor while the live pipeline reports sitting/absent |
| 2026-09-20_living-room-sample-01 | walk_return | 31–44 | return and crossing sequence with live walking/standing transitions |
| 2026-09-20_living-room-sample-01 | sit_depart | 52–64 | standing to sitting and leaving view; expected pose fade after departure |
| 2026-09-20_living-room-sample-01 | reel | — | floor_sequence + walk_return + sit_depart, 39 s |

Smoke-tested also on 2026-09-14_bedroom-sample-04 (1080×720, fitted with side
bars); output deleted.

## Open problems

- Portrait sources (sample-05 is vertical) stop with a message: the HUD layout is
  landscape-only. Needs a portrait layout.
- Agent lines and caregiver notifications aren't shown; they need an agent replay
  of the clip's events.
- Not yet ported into the repo as `tools/video_eval` `render_demo`; if that
  happens, point this skill at the repo command instead of the bundled scripts.

## Keeping this skill current

At the end of every task that used this skill, edit this file:

- Add rendered snippets to the **Snippet log** (clip, name, range, one-line note);
  remove rows whose files were deleted.
- Add any new failure mode and its fix to **Pitfalls**; close items in **Open
  problems** that got solved.
- If a script changed, edit the copy in `scripts/` (it is canonical) and adjust
  **How the render looks** if the behaviour changed. Keep commands in
  **Workflow** runnable as written.
- If paths moved (work dir, models, venv, zones, a repo port), update **Where
  things live** first; stale paths are the most expensive mistake here.
- Keep the file under ~250 lines: fold old detail into a sentence rather than
  letting the log sprawl.
- The skill lives in git, so these edits are repo changes: make them on a branch
  (a worktree if you're a background session), commit, push and open a PR like any
  other change. Commit only the skill; clip ids and time ranges are fine to log,
  frames and videos never are.
