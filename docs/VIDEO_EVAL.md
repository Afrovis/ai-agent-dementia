# Video evaluation plan

How to test perception and the agent against recorded bedroom videos, using
strong vision models as reference labellers and comparing our own pose
pipeline against them. Written 2026-09-13 against the first sample clip.

This document has two audiences. Phase A is a one-time build that turns the
plan into tooling. Phase B is the runbook an agent follows every time a new
video arrives. Read "Ground rules" and "Verified facts" regardless of which
phase you are executing.

Terminology: what we need is not OCR (text recognition). The reference
labellers are vision-language models (VLMs) that look at a frame and answer
questions about it, plus a stronger pose model where useful.

## 1. What we want to learn

For each recorded clip, answer these in a report:

1. Does `perceive` detect the person at all, per frame, at the resolution the
   real pipeline uses? Where does it lose them (in bed under a blanket, at the
   frame edge, in the dark)?
2. When it detects them, is the posture right? Per-state recall and precision
   for `in_bed`, `sitting_up`, `standing`, `walking`, `on_floor`, `absent`.
   The PLAN.md gate is 95 % recall on `standing` and `on_floor`.
3. How long after a real transition (bed exit, floor, room exit) does
   `PersonState` change? PLAN.md target is under 2 s for `on_floor`.
4. Is the zone right (`bed`, `door`, `bathroom_path`, `other`) when the person
   is in a drawn zone?
5. End to end: does the agent enter `OBSERVING` and `ENGAGED` when it should,
   go to `restroom` on the door or bathroom path, and never `ESCALATED`
   without a floor or absence event?
6. Which backend and settings do best: MediaPipe vs YOLOv8-pose, confirm
   frames, min confidence, squash vs letterbox?

## 2. Ground rules

Privacy first. The videos are of a real person in their bedroom.

- Raw video, unblurred frames, labels and reports live outside the
  repository in `../data-ai-agent-dementia/` (sibling of this repo, already
  exists). Nothing under it is ever committed. Do not copy frames into
  `tests/`, `data/` or a PR, and never attach frames to an issue.
- The only thing that may leave the Mac mini is a face-blurred, downscaled
  frame sent to Codex for labelling. The blur tool must fail closed: when it
  cannot locate a head confidently it blurs the whole person box, and a frame
  is only eligible for upload after a second face detector finds nothing in
  it. Local models (Ollama, MediaPipe, YOLO) may see unblurred frames.
- The sample has an audio track. Never send audio anywhere. Strip it from
  anything derived.
- Report results as numbers, never as frames. If a report needs to point at
  a frame, use the frame index and the local path.

Fidelity second. As of issue #52, the bedside browser produces a 320 by 240
JPEG at quality 0.6 and 2 fps, fitting the complete camera image into that
4:3 canvas with black letterbox bars. A 16:9 camera therefore occupies 320 by
180 pixels at y=30 with its proportions intact. `bridge-letterbox/` is the
current runtime-equivalent evaluation variant; `bridge/` retains the old
horizontally squashed rendering for comparison. Higher-resolution frames are
for the reference labellers and for humans only.

Reproducibility third. Every prediction and report is tagged with the git
commit of this repo, the backend name and its package version, and the
labeller model names. Reference labels are versioned separately from
predictions so that re-running perceive after a code change reuses them.

## 3. Verified facts (2026-09-13)

Machine: Mac mini, Apple M4, 16 GB RAM, 42 GB free disk, macOS 15.3.
Tooling present: ffmpeg and ffprobe at `/opt/homebrew/bin`, Docker, Ollama
0.32 with `qwen3-vl:8b` (6.1 GB), `gemma4:e4b-mlx` (8.8 GB) and
`mistral-nemo:12b` (7.5 GB), Codex CLI 0.154 configured for model
`gpt-5.6-sol`, Python 3.12 at `/opt/homebrew/bin/python3.12`. Only one
Ollama model fits in RAM alongside the Docker stack; they load one at a
time and that is fine. The `ollama` CLI is newer than the server and shows
an empty model list; use the HTTP API (`/api/tags`, `/api/pull`) instead.
Sandboxed shells cannot write to `~/.ollama`, so pull with
`curl -X POST localhost:11434/api/pull -d '{"name":"...","stream":false}'`.

Sample clip `../data-ai-agent-dementia/Bedroom Recording.mov`:

| Property | Value |
| --- | --- |
| Duration | 244 s |
| Resolution, rate, codec | 3840 by 2160, 60 fps, HEVC, from OBS Studio |
| Size | 37.6 MB |
| Audio | AAC track present, must be stripped |
| Lighting | Dim, one lamp, colour camera, no infrared |
| Content | Person walks in, lies in bed, sits up, stands, walks around, leaves; bed is empty for a stretch |
| Door | Out of frame, to the left. Leaving the room means walking out of the left edge. |

Scope decision (2026-09-13): this is test footage, not the deployment
placement. Start with RGB in lamp light because it is the easiest case to
get right; infrared and dark-room clips come later, once the RGB numbers
are stable.

Smoke test results on five frames from that clip, converted to the bridge
format (320 by 240, squashed, quality 60):

| Check | Result |
| --- | --- |
| MediaPipe pose 0.10.21, model_complexity 1 | Detected 2 of 5 (both standing mid-room). Missed one standing at the frame edge and both in-bed frames. 0.06 s per frame. |
| YOLOv8n-pose | Detected 4 of 5 at 320 by 240, with confidence 0.30 and 0.32 on the in-bed frames and one spurious extra box. 0.04 s per frame. |
| MediaPipe face detector (full-range) on 640-wide frames | Found the face in 1 of 4 frames. Face detection alone is not enough for blurring. |
| Ollama `qwen3-vl:8b` (6.1 GB, pulled 2026-09-13), JSON mode, one 640-wide frame | Correct on 6 of 6 including both under-the-blanket frames and a crouch beside the bed labelled `on_floor`. 16 to 24 s per frame, one outlier at 55 s. Once returned an out-of-enum `location`, so validate fields, not just JSON. |
| Ollama `gemma4:e4b-mlx`, same prompt | Correct on 3 of 3 easy frames, not tested on in-bed frames. About 9 s per frame once loaded, 37 s for the first call. |
| Codex `exec` with `-i` image | Works. Prompt must come on stdin (`echo prompt \| codex exec ... -`). About 6 s per call. |

Three defects found while planning. File each as an issue and fix them
before trusting any numbers.

1. **MediaPipe 1.0 removed the API `perceive` uses.** `services/perceive`
   pins `mediapipe>=0.10`, which resolves to 1.0.1 on both macOS and Linux
   as of today, and 1.0 has no `mediapipe.solutions`. A fresh
   `docker compose build perceive` yields a backend that crashes on first
   use. Pin `mediapipe>=0.10,<1.0` now; migrate to the tasks API later.
   MediaPipe 1.0 also aborts on macOS in a headless process with a Metal
   initialisation failure, so host-side venvs need the same pin.
2. **Resolved in issue #52: the bridge squashed 16:9 to 4:3.** The old
   five-argument `drawImage` compressed horizontal proportions by 25% and
   reported only the canvas size. The selected behavior letterboxes the full
   image and reports both the 320 by 240 encoded dimensions and the camera's
   intrinsic dimensions. The eval retains both variants for comparison.
3. **The running stack is stale.** The containers named `test-m0-*` were
   built on 2026-09-09 from `.claude/worktrees/test-m0`, and their
   `perceive` is a placeholder that logs "perceive is not implemented yet".
   Any end-to-end run must first rebuild from this checkout with
   `docker compose build && docker compose up -d`.

## 4. Pipeline design

```
raw video (4K, 60 fps, audio)
   |
   |  prepare: ffprobe, drop audio, sample at 2 fps
   v
frames.jsonl + two frame sets per clip
   |-- bridge/   320x240 squashed q60   -> what perceive sees
   |-- review/   640-wide, correct aspect -> labellers and humans
   |
   |--> blur      -> blurred/  (review frames, head-blurred, verified)  -> Codex
   |
   |--> label-local  (Ollama VLM on review/)     -> labels/local.jsonl
   |--> label-codex  (Codex on blurred/ sheets)  -> labels/codex.jsonl
   |--> reconcile    (+ the actor's script)      -> labels/reference.yaml
   |                                                 + disagreements.md
   |--> predict   (perceive backend + StateTracker + MotionGate on bridge/)
   |                                              -> predictions/<tag>.jsonl
   |--> score     (reference vs predictions)      -> reports/<tag>.md, .json
   |
   '--> replay    (bridge/ as RawFrame events onto the live stack)
                                                  -> e2e/<date>/bus.jsonl, report
```

Reference labels are a timeline of intervals, the same shape
`tests/perception_bench/fixtures/ir_manifest.example.yaml` already defines
(`from_s`, `to_s`, `state`), extended with `zone`. That lets the existing
tier 3 of `perception_bench` consume them once `infrared.py` is filled in.

### Why three sources for the reference

- **The actor's script.** The person in the video is the person recording
  it. Before recording, they write what they will do and roughly when. That
  card is the cheapest and most reliable ground truth and settles most
  disagreements. Make it part of the recording protocol (section 8).
- **Local VLM** labels every sampled frame. It is free and private but a
  4-billion-parameter model will confuse sitting on the bed edge with
  standing next to it, and cannot tell walking from standing in one frame.
- **Codex** sees blurred contact sheets, nine frames per call with the frame
  index printed on each tile. It is the strongest labeller we have and
  cheap enough to run on every frame, so run it on all of them rather than
  only on disagreements. Blur is verified before upload; see section 5, A3.

`walking` is never asked of a labeller. It is derived from person-centroid
displacement between consecutive review frames, exactly the way
`classify.py` derives it, so the labellers answer a five-way question:
`in_bed`, `sitting_up`, `upright`, `on_floor`, `absent`, plus a location.
The scorer collapses `standing` and `walking` into `upright` for the
primary metric and reports the split separately.

## 5. Phase A: one-time build

Everything lives in a new package `tools/video_eval/` in this repository,
host-side, Python 3.12, with a pinned lock. It never imports from the
services except `perceive` and `capture` (installed editable) so that
`predict` runs the real classifier. Each item below is one issue and one PR.
Order matters for A1 to A4; the rest can run in parallel.

Common conventions:

- One CLI, `python -m video_eval <command> --clip <clip_id>`, with
  `VIDEO_EVAL_DATA` defaulting to `../data-ai-agent-dementia`.
- Every command is idempotent and skips work whose output exists unless
  `--force`. Every command writes a `*.meta.json` next to its output with
  the git sha, package versions, model names, parameters and wall time.
- JSON line per frame everywhere, keyed by `frame_index` and `t_s`.
- No command needs the Docker stack except `replay`.

Data layout, created by `prepare`:

```
../data-ai-agent-dementia/
  raw/                         original recordings, never modified
  clips/<clip_id>/
    clip.yaml                  scenario card: recorded_at, camera, light, script, notes
    zones.yaml                 normalised polygons on the *bridge* frame
    frames.jsonl               frame_index, t_s, bridge_path, review_path
    bridge/f_000001.jpg        320x240 squashed, quality 60, 2 fps
    review/f_000001.jpg        640 wide, true aspect, quality 85
    blurred/f_000001.jpg       review frames with heads blurred, verified
    sheets/s_0001.jpg          3x3 contact sheets of blurred frames for Codex
    labels/local.jsonl
    labels/codex.jsonl
    labels/reference.draft.yaml
    labels/reference.yaml      confirmed by a human
    labels/disagreements.md
    predictions/<tag>.jsonl    tag = <backend>-<variant>-<gitsha>
    reports/<tag>.md, <tag>.json
    e2e/<date>/bus.jsonl, report.md
  index.yaml                   one line per clip, status per stage
```

Clip ids are `YYYY-MM-DD_<slug>`, e.g. `2026-09-13_bedroom-sample-01` for
the current file.

### A1. `prepare`

Input: a path to a video and a clip id. Steps:

1. `ffprobe` and record duration, size, rate, codec, rotation into
   `clip.yaml` (create the card if missing, with empty `script:`).
2. Copy the source into `raw/<clip_id>.<ext>` if not already there.
3. Extract review frames with ffmpeg at 2 fps, 640 wide, `-an`, quality 85.
4. Produce bridge frames from the review frames in Python with Pillow:
   resize to exactly 320 by 240 with bilinear filtering ignoring aspect,
   save JPEG quality 60. That reproduces the canvas squash. Add
   `--variant letterbox` that pads to 4:3 first, written to
   `bridge-letterbox/`, for the experiment in section 7.
5. Write `frames.jsonl`. `t_s` is `frame_index / 2`.

Acceptance: for the sample, 488 frames in each set, bridge files each under
20 KB, and a `prepare.meta.json`.

### A2. `predict`

Runs the real perception code on bridge frames, offline, no Redis.

1. Build the backend with `perceive.backends.build_backend(kind)`
   (`mediapipe`, `yolo`), the zones with `perceive.zones.load_zones(path)`
   from the clip's `zones.yaml`, and a `StateTracker` from
   `perceive.classify` with the same env-derived
   settings `perceive.main.PerceiveConfig` would use. Expose
   `--confirm-frames`, `--min-confidence`, `--walk-threshold` overrides.
2. Replicate the capture gate: instantiate `capture.gate.MotionGate` with
   the `.env` defaults and call `admit(jpeg, t_s)` per frame. Frames it
   drops are recorded with `gated: true` and not passed to the tracker.
   `--no-gate` disables this for the pure-perception number.
3. For each admitted frame call `backend.detect`, compute the zone from the
   pose centroid, call `tracker.update(pose, zone, t_s)` using the frame
   time as `now`, then `tracker.snapshot()`. Write one line per frame:
   `frame_index, t_s, gated, detected, detect_confidence, bbox, state,
   state_confidence, zone, published (bool: update returned a change),
   backend_ms`.
4. Tag output `<backend>-<variant>-<gitsha>` where variant is `squash` or
   `letterbox` and `-g` for gated.

Run it inside the `perceive` image when in doubt about package parity:
`docker compose run --rm --no-deps -v "$PWD/../data-ai-agent-dementia:/eval" perceive python -m video_eval predict ...`
needs the tool installed in that image; simplest is a `tools` extra in the
perceive Dockerfile guarded by a build arg. On the host, the venv must pin
`mediapipe<1.0` (see defect 1).

Acceptance: runs on the sample in under a minute per backend, and the
`gated` count is nonzero while the room is still.

### A3. `blur`

Head blurring with fail-closed verification, on review frames.

1. Detect people with YOLOv8n-pose on the review frame (it found people the
   face detector missed). For each person, define the head region as the
   union of: a circle around the nose keypoint with radius 1.2 times the
   shoulder width; the top 25 % of the person box; and any face box from
   the MediaPipe face detector padded by 60 %. Apply a Gaussian blur with
   sigma at least a tenth of the region size, then pixelate to 8 px cells.
2. If a person is detected but neither nose nor shoulders have visibility
   above 0.3, blur the entire person box. If no person is detected but the
   local VLM said `person_visible: true` for that frame, blur the whole
   frame and mark it `blur_mode: full`.
3. Verify: run the face detector on the result at two scales. If it finds
   anything, blur harder and re-verify; after three attempts, mark the
   frame `upload_ok: false`.
4. Write `blurred/` and `blur.jsonl` with `blur_mode` and `upload_ok`.
5. `sheets`: tile eligible frames nine per 3 by 3 sheet, each tile 426 by
   240 with a black label strip reading the frame index and `t_s`. Sheets
   containing any `upload_ok: false` frame are not built; those frames get
   local labels only. Sheets are the only files `label-codex` may read.

Acceptance: a test with a synthetic face image and a test that a frame
marked `upload_ok: false` never appears in a sheet. A human spot-checks the
first clip's sheets before the first Codex call.

### A4. `label-local` and `label-codex`

Both produce the same record per frame:
`frame_index, t_s, labeller, person_visible, posture (in_bed | sitting_up |
upright | on_floor | absent), location (bed | door | bathroom_path | other),
confidence, note`.

`label-local` posts each review frame to Ollama `/api/chat` with
`format: "json"`, temperature 0, model from `--model` defaulting to
`qwen3-vl:8b` (`--fast` switches to `gemma4:e4b-mlx`), and the prompt
below. Validate every field against its enum; retry once on invalid JSON
or an out-of-enum value, then record `posture: null`. Budget about 20 s per
frame with Qwen, so a 4-minute clip at 2 fps is close to three hours.
Default to `--adaptive`: label every frame while the review frames show
motion (reuse `capture.motion.frame_signature` and `motion_score` with the
gate's threshold) and every tenth frame during still stretches, then
propagate a still stretch's label across it. That brings the sample to
roughly 40 minutes. Run it in the background. Unload the model when done
with `curl localhost:11434/api/generate -d '{"model":"qwen3-vl:8b","keep_alive":0}'`
so the Docker stack has its memory back.

```
You label bedroom monitoring frames for a fall and wandering safety system.
Answer with JSON only:
{"person_visible": true|false,
 "posture": "in_bed"|"sitting_up"|"upright"|"on_floor"|"absent",
 "location": "bed"|"door"|"bathroom_path"|"other",
 "confidence": 0.0-1.0,
 "note": "at most 15 words"}
in_bed = lying on the bed, even under a blanket. sitting_up = torso upright
while on the bed or its edge. upright = standing or walking anywhere.
on_floor = lying, kneeling or crawling on the floor. absent = no person.
Do not describe identity, clothing or the room.
```

`label-codex` runs, per sheet:

```
echo "$PROMPT" | codex exec -s read-only --skip-git-repo-check -i sheets/s_0001.jpg -
```

with a prompt that explains the tile labels and asks for a JSON array with
one object per tile in reading order. Parse, validate the count, and write
one record per frame. The default `codex` model in `~/.codex/config.toml`
is used unless `--model` is given. About 6 s and 5k tokens per call, so a
4-minute clip costs about 55 calls.

### A5. `reconcile`

Turns per-frame labels plus the scenario card into a reference timeline.

1. Per frame, decide `posture` and `location`: agree when the two
   labellers agree; otherwise prefer Codex if its confidence is at least
   0.7, otherwise mark `disputed`.
2. Derive `walking` from review-frame centroid displacement over the last
   five frames using the same threshold `classify.py` uses, only for frames
   labelled `upright`.
3. Smooth into intervals: merge consecutive frames with the same state,
   then absorb any interval shorter than 2 s into its neighbours, except
   `on_floor`, which is never absorbed.
4. Align with the scenario card: if the card lists an event (`bed_exit`,
   `floor`, `room_exit`, `return`) and the timeline shows the matching
   transition within 10 s of the card's time, keep the timeline's precise
   time; otherwise add an entry to `disagreements.md`.
5. Write `reference.draft.yaml` in the `ir_manifest` interval format with an
   added `zone` per interval, plus `disagreements.md` listing every disputed
   run with frame indices and local review paths for a human to look at.
6. `reconcile --confirm` copies the draft to `reference.yaml` once a human
   has edited or accepted it, and stamps the confirmer and date in the
   file. Predictions are never scored against a draft.

### A6. `score`

1. Expand `reference.yaml` to a per-frame label at 2 fps.
2. Per-frame metrics using `perception_bench.scoring.score_predictions`:
   confusion matrix, per-state recall and precision, overall accuracy, and
   the same with `standing` and `walking` collapsed to `upright`. Report
   separately on all frames and on admitted (non-gated) frames only, and
   detection rate per reference state (how often the person was found at
   all while `in_bed`, and so on).
3. Event metrics: for each reference transition into `sitting_up`,
   `upright`, `on_floor`, `absent` and each zone entry into `door` or
   `bathroom_path`, the delay until the prediction first shows that state,
   or `missed` if it never does within 30 s. Also false transitions per
   minute: predicted transitions with no reference transition within 10 s.
4. Gate evaluation against PLAN.md: recall for `standing` and `on_floor`
   at least 0.95, `on_floor` delay at most 2 s. Print which gates were
   measured, met, or not measurable in this clip.
5. Write `reports/<tag>.md` (human) and `reports/<tag>.json` (machine), and
   update `index.yaml`.

### A7. `replay` (end to end)

1. Rebuild and start the stack from this checkout. Set
   `PERCEIVE_VISION_ENABLED=false` for the run unless the scene-note path
   is what is being tested, and use a temporary `DASHBOARD_PASSWORD`.
2. Convert `bridge/` into a replay JSONL of `RawFrame` events on stream
   `frames_raw`, `source_kind: browser`, spaced 0.5 s apart, and play it
   with `python -m nc_shared.replay play` at speed 1 while
   `python -m nc_shared.replay record` captures `person`, `session` (which
   also carries `GoalChanged`), `say`, `notify` and `light` into
   `e2e/<date>/bus.jsonl`. The agent only reacts inside its night window
   (`AGENT_NIGHT_START` 21:00 to `AGENT_NIGHT_END` 07:00 by default);
   setting both to the same value makes the window always on, which is
   the intended way to run this in daytime. Note the setting in the report.
3. Compare: `PersonState` from the bus against `reference.yaml` (should
   match the offline `predict` result; a mismatch means the container and
   the host disagree on packages or settings), and `SessionState`
   transitions against expectations derived from the reference: `OBSERVING`
   within 5 s of the first `sitting_up` or `upright`, `ENGAGED` after 20 s
   up, `restroom` goal on `door` or `bathroom_path`, no `ESCALATED` unless
   the reference has `on_floor` or a long `absent`.
4. Write `e2e/<date>/report.md`. Stop the stack, and delete
   `frames_raw`, `frames` and any speech cache afterwards.

### A8. Wire into `perception_bench` tier 3

Fill in `tests/perception_bench/perception_bench/infrared.py` so that
`python -m perception_bench --ir-manifest ../data-ai-agent-dementia/manifest.yaml`
scores every confirmed clip using the `predict` and `score` code above, and
`reconcile --confirm` appends the clip to that manifest. Tier 3 still skips
cleanly when the manifest is absent, so CI stays green.

### A9. Zones per camera placement

Zones are drawn on the bridge frame, because that is what `perceive` sees
and the squash moves things. Two options; do the first now:

- `python -m video_eval zones --clip <id>` prints the path of one bridge
  frame and expects a `zones.yaml` written by hand in the same normalised
  format `config/zones.example.yaml` uses. For the sample, the bed
  occupies roughly the left half of the frame and the door is out of frame
  to the left, so `door` is a narrow strip along the left edge (x from 0.0
  to about 0.06, full height) that a person crosses on the way out. It
  must not overlap the bed polygon, because `ZoneMap` gives `bed`
  precedence. There is no bathroom path in this placement; leave it out.
- Later, load a bridge frame into the dashboard Zones editor and save.

A clip records the camera placement in `clip.yaml`; clips from the same
placement share a `zones.yaml`.

### A10. Fix the three defects

Separate PRs: pin `mediapipe<1.0`; decide squash vs letterbox for the
bridge after the section 7 experiment and fix `script.js` accordingly,
sending true dimensions; tear down and rebuild the stale `test-m0` stack.

## 6. Phase B: runbook for each new video

An agent runs this from the repository root with the venv at
`tools/video_eval/.venv` created per its README. Replace `<clip>` with a
new id. Report numbers only, never frames.

1. **Intake.** Confirm the file is under `../data-ai-agent-dementia/`. Run
   `python -m video_eval prepare --video <path> --clip <clip>`. Open
   `clips/<clip>/clip.yaml` and copy in the scenario card the recorder
   wrote (section 8). If there is no card, write `script: unknown` and
   expect more disagreements. Copy `zones.yaml` from the most recent clip
   with the same camera placement, or ask for one.
2. **Predict first.** `predict --backend mediapipe` and
   `predict --backend yolo`, both gated. This needs no labels and takes
   about a minute. Check the meta files carry the current git sha.
3. **Local labels.** `label-local --clip <clip>` in the background. Expect
   about 10 s per model call. Running this before blur gives the fail-closed
   path a second signal when YOLO cannot find a person.
4. **Blur and sheets.** `blur --clip <clip>` then `sheets --clip <clip>`.
   Note how many frames are `upload_ok: false`. On the first clip from a
   new camera placement, a human looks at three sheets, then explicitly runs
   `sheets --clip <clip> --confirm-reviewed` before step 5.
5. **Codex labels.** `label-codex --clip <clip>`. Expect about 6 s per
   sheet. Stop and report if any call returns an error mentioning
   content or images; do not retry with different frames.
6. **Reconcile.** `reconcile --clip <clip>`. Read `disagreements.md`. If it
   is empty and the scenario card aligned, run `reconcile --confirm`.
   Otherwise stop and hand the list to the recorder to confirm; the
   recorder edits `reference.draft.yaml` and reruns `reconcile --confirm`.
7. **Score.** `score --clip <clip>` for every prediction tag. Read the two
   reports. If both backends miss the same stretch, look at the review
   frames for that stretch locally and describe in words what the scene
   was (blanket, edge of frame, darkness).
8. **End to end.** Only when the reference is confirmed and the perception
   score is not obviously broken: `replay --clip <clip>`. Requires the
   stack rebuilt from the current checkout.
9. **Record the outcome.** Append a row to
   `../data-ai-agent-dementia/RESULTS.md`: clip id, git sha, backend,
   per-frame accuracy, `upright` recall, `on_floor` recall or "not
   measurable", bed-exit delay, false transitions per minute, e2e verdict,
   one line of notes. Copy the same row into a comment on the tracking
   issue in this repository, numbers only.
10. **Clean up.** Unload the Ollama model (see A4). Confirm nothing under
    `../data-ai-agent-dementia/` was staged in git: `git status` must not
    list it.

Failure handling: a step that fails is reported with its stderr and the
runbook stops; never skip to a later step, and never lower a threshold to
make a gate pass.

## 7. Experiments to run once the tooling exists

Run each on every confirmed clip and keep the results in `RESULTS.md`.

1. **Backend**: MediaPipe vs YOLOv8n-pose on identical bridge frames. The
   smoke test suggests YOLO finds low-confidence people in bed that
   MediaPipe drops; measure whether that helps `in_bed` or adds false
   states.
2. **Squash vs letterbox**: `predict` on `bridge/` and `bridge-letterbox/`.
   If letterbox is clearly better, fix the page.
3. **Resolution**: 320 by 240 vs 480 by 360 vs 640 by 480 at the same
   quality. The page constant is one line; this tells us the cost of the
   current choice.
4. **Hysteresis**: `--confirm-frames` 1, 2, 3 against the 2 s latency gate
   and false transitions per minute.
5. **Gate**: gated vs `--no-gate`. Frames dropped by the motion gate cannot
   contribute to latency; measure the added delay.
6. **Vision scene notes**: with `PERCEIVE_VISION_ENABLED=true` during
   `replay`, compare `scene_note` text against the reference state for the
   frames where it fired. Qualitative only.

### Aspect-ratio decision (issue #52, 2026-09-13)

The first sample's 488 frames were run through YOLOv8n-pose 8.4.150 without
the motion gate on both prepared variants at commit `f208d4a1`:

| Variant | Detected frames | Detection rate | Mean confidence when detected |
| --- | ---: | ---: | ---: |
| Squash | 344/488 | 70.5% | 0.646 |
| Letterbox | 310/488 | 63.5% | 0.741 |

Letterboxing produced 34 fewer detections (6.97 percentage points) but raised
mean confidence by 0.095 (14.7% relative). More importantly, it removes a
deterministic 25% horizontal compression from every body ratio used by the
classifier, retains the entire camera view, and makes normalised geometry
portable across camera aspect ratios. The bridge therefore selects
**letterbox**. This is a detector-only comparison because the sample still
lacks a recorder-confirmed reference timeline; it is not presented as
state-accuracy evidence. Re-run the normal scorer when that reference exists.

## 8. Recording protocol for new clips

The person recording is also the ground truth. Before each recording, write
a scenario card and save it as `clip.yaml` next to the clip:

```yaml
clip_id: 2026-09-20_night-bed-exit-01
recorded_at: 2026-09-20T23:10:00-04:00
camera: obs-webcam-dresser   # same string for the same placement
light: lamp-dim | dark-ir | daylight
script:
  - {t_s: 0,   action: in_bed}
  - {t_s: 40,  action: sitting_up}
  - {t_s: 55,  action: bed_exit}         # stands up
  - {t_s: 70,  action: walk_to_door}
  - {t_s: 90,  action: room_exit}
  - {t_s: 150, action: return}
  - {t_s: 170, action: in_bed}
notes: blanket over legs only; lamp on the left
```

`room_exit` means the person walks out of the frame through the door
side (left edge in the sample placement) and stays out; `return` is the
moment they re-enter the frame.

Scenarios worth recording, each as its own 3 to 5 minute clip, in the order
of value to the project. Record 1 to 4 in RGB with the lamp on first; 5 is
deferred until the RGB numbers are stable.

1. Night bed exit and return: lie still under a blanket 60 s, sit up, sit
   on the edge, stand, walk to the door, leave, come back, lie down.
2. Restless in bed: roll over, sit up briefly and lie back down, several
   times. This is the false-alarm test.
3. On the floor, safely: kneel and then lie on the floor beside the bed
   for 30 s, in and out of the bed zone. No actual falling. This is the
   only way to measure the `on_floor` gate.
4. Partial visibility: sit on the far edge of the bed, stand at the frame
   edge, stand half behind furniture.
5. Dark, later: the same as scenario 1 with the lamp off and, once an
   infrared camera exists, in IR. Expect the current models to fail here;
   that is the point. Not part of the first round.
6. Long still: 10 minutes in bed with no motion, to exercise the gate's
   idle rate and the heartbeat.

Record in the same placement as the intended deployment camera, at the
resolution the deployment camera will use if known. 1080p at 30 fps is
plenty; 4K just slows `prepare`. Keep clips under 10 minutes so labelling
stays under two hours each.

## 9. Resource budget on the Mac mini

| Stage | Cost per 4-minute clip |
| --- | --- |
| prepare (ffmpeg, 4K source) | about 1 to 2 minutes |
| predict, one backend | under 1 minute |
| blur and sheets | about 1 minute |
| label-local, qwen3-vl 8b | about 3 hours on every frame, about 40 minutes with `--adaptive`; gemma4 e4b is roughly twice as fast and less accurate |
| label-codex | about 55 calls, roughly 6 minutes, about 300k tokens |
| replay | real time, 4 minutes plus stack rebuild |

RAM: the Docker stack, one Ollama model and the YOLO or MediaPipe venv fit
together in 16 GB. Do not run `label-local` and `replay` at the same time.
Ollama unloads a model after five minutes idle; `ollama stop` forces it.

## 10. Decisions and open questions

Answered by the recorder on 2026-09-13:

- The sample placement is for testing only, not the deployment camera.
  Results describe this placement until a deployment camera exists.
- RGB in lamp light first. Infrared and dark clips are a later round.
- The door is out of frame to the left. Zones and the scenario card treat
  the left edge as the way out. No bathroom path in this placement.
- A stronger local vision model is approved for `label-local` as long as
  it runs on the Mac mini. `qwen3-vl:8b` was pulled and verified the same
  day and is the default. Nothing beyond blurred sheets goes to Codex.

Still open:

1. Which camera and placement will the real deployment use, and at what
   resolution does it deliver frames to the browser page?
