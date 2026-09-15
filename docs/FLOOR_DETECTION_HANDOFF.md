# Floor detection: findings and handoff (2026-09-14)

Status: written as an investigation handoff; its main recommendations
have since shipped in PR #55 (see the status note at the top of
section 9). This document records what was measured on the two bedroom
recordings and what it means for `on_floor` detection. Read it together
with [VIDEO_EVAL.md](VIDEO_EVAL.md).

Privacy: every number here comes from local processing on the Mac mini.
No frame was viewed outside the machine or sent to a hosted model; the
vision-model test used local Ollama only.

## 1. The problem

A person sitting, kneeling or lying on the floor is the event this system
most needs to catch, and on the 2026-09-13 recordings the real-time
pipeline reports it in **0 of 55** floor-labelled frames on the current
defaults.

Headline result: a floor-relative height rule (section 4) on YOLOv8s with
640x480 input catches 40 of 55 floor frames and all three floor events,
with no false floor frames on these clips. The rule alone fixes the
sitting-on-the-floor case; the larger model and input are what let it see
the person beside the bed. Rotation test-time augmentation was a false
lead (section 6), and a local vision model is precise but misses most
floor frames (section 7).

## 2. Data and ground truth

| Clip | Floor event (recording script) | Floor-labelled frames | Setting |
| --- | --- | --- | --- |
| `2026-09-13_bedroom-sample-01` | 120-128 s `floor`, 135-144 s `floor` (then getting up) | 35 | on the floor beside the bed, partly behind it |
| `2026-09-13_bedroom-sample-02` | 99-108 s `sitting_on_floor` | 20 | sitting on the floor by the dresser, in open view |

Frames are the 2 fps, 320x240 letterboxed bridge frames produced by
`video_eval prepare` (`bridge-letterbox/`), plus a 640x480 variant
(`bridge-letterbox-640/`).

Caveats that bound every result below:

- Labels (`labels/local.jsonl`) were produced by `qwen3-vl:8b`, not by a
  person. At least one floor-labelled frame is described as "bent over
  near bed". Where it matters, positives were restricted to frames that
  are labelled `on_floor` *and* fall in a scripted floor window.
- Three floor events, one actor, two camera placements. These numbers
  show direction, not rates to quote.
- Metrics are per frame with no hysteresis unless stated. Non-floor
  frames within 2 s of a floor label boundary are excluded from false
  positive counts, since label timing is approximate.

## 3. Baseline: what the pipeline does today

States reported on floor-labelled frames by the existing
`video_eval predict` runs (`clips/<clip>/predictions/`):

| Run | Clip 01 (35) | Clip 02 (20) |
| --- | --- | --- |
| MediaPipe, 320 letterbox | 10 detected; absent 30, sitting_up 5 | 20 detected; sitting_up 16, walking 2, on_floor 2 |
| YOLOv8n, 320 letterbox | 14 detected; absent 30, sitting_up 4, standing 1 | 20 detected; sitting_up 17, walking 3 |
| YOLOv8n, 640 letterbox | 15 detected; absent 28, sitting_up 7 | 20 detected; sitting_up 17, walking 3 |
| YOLOv8s, 320 letterbox | 11 detected; absent 29, sitting_up 6 | 20 detected; sitting_up 17, walking 3 |
| YOLOv8s, 640 letterbox | **25 detected**; sitting_up 18, absent 17 | 20 detected; sitting_up 17, standing 3 |
| YOLO11s, 320 letterbox | 18 detected; walking 18, sitting_up 14, absent 3 | 20 detected; sitting_up 17, walking 3 |
| `yolo-letterbox-rules7cc-g` (earlier rule experiment) | on_floor 18, walking 6, absent 11; **9 false on_floor elsewhere** | on_floor 19; 3 false elsewhere |

The `rules7cc` run was made at commit `7cc4ff2` ("Add floor, dropout and
bed-vanish rules behind flags; offline detection cache and replay"). At
that commit `floor_top_y` defaults to disabled, so the run must have set
flags through the environment, and its `.meta.json` does not record them.
Treat its numbers as "a low-box-top rule catches floor frames but costs
false alarms", not as a reproducible configuration. The same commit added
`tools/video_eval/scripts/detection_cache.py` and `rule_replay.py`, which
overlap with the scratch scripts in section 11; check them before building
new tooling.

Two distinct failures:

1. **Clip 02 is a geometry failure.** The person is detected in every
   floor frame and classified `sitting_up`. The lying rule needs a box
   twice as wide as tall; a seated person's box is taller than wide. The
   low-box-top rule (`PERCEIVE_FLOOR_TOP_Y`) is off by default.
2. **Clip 01 is a detection failure.** The person is found in 10-18 of 35
   floor frames at 320 px, and `absent` swallows the rest. No
   classification rule can fix frames with no detection.

## 4. Fix 1: height relative to the floor (tested, works)

### Method

With a fixed camera, the image row where a person's feet touch the floor
determines how tall a standing person looks there. Fit that line from
frames the pipeline itself already calls `standing` (self-calibration),
then score every detection:

```
ground_y       = lowest visible ankle (keypoint visibility >= 0.3)
height         = ground_y - bbox top
expected       = a * ground_y + b          # Theil-Sen fit on standing frames
height_ratio   = height / expected
```

Fitted lines were stable and backend-specific (box conventions differ):
YOLOv8n clip 01 `a=2.45 b=-1.50` (n=210 standing frames), clip 02
`a=2.06 b=-1.09` (n=109); MediaPipe clip 01 `a=1.95 b=-1.12`. Standing
frames come out at a median ratio of 0.99-1.00, as they should.

### Separation (YOLOv8n, 320)

| Label | Clip 01 height_ratio p10 / p50 / p90 | Clip 02 |
| --- | --- | --- |
| on_floor | 0.27 / 0.45 / 0.66 (n=7) | 0.46 / 0.48 / 0.49 (n=17) |
| sitting_up (chair, bed edge) | 0.78 / 1.06 / 1.15 | 0.69 / 0.88 / 1.11 |
| upright | 0.90 / 0.99 / 1.12 | 0.75 / 1.00 / 1.07 |

### Rule simulation: `height_ratio <= T`, outside bed zone, confidence >= 0.5

| Setup | Rule | Clip 01 recall / false | Clip 02 recall / false |
| --- | --- | --- | --- |
| YOLOv8n 320 | current rules | 0/35 / 0 | 0/20 / 0 |
| YOLOv8n 320 | ratio <= 0.6 | 3/35 / 0 of 435 | **17/20 / 0 of 200** |
| YOLOv8n 320 | ratio <= 0.6, knee fallback when ankles hidden | 4/35 / 0 | 17/20 / 0 |
| MediaPipe 320 | current rules | 0/35 / 0 | 2/20 / 10 |
| MediaPipe 320 | ratio <= 0.6 | 7/35 / 0 | 19/20 / 12 |
| MediaPipe 320 | ratio <= 0.6, knee fallback | 8/35 / 0 | 20/20 / 14 |
| YOLOv8s 320 | ratio <= 0.6 | 5/35 / 0 | 19/20 / 0 |
| YOLOv8s 320 | ratio <= 0.6, knee fallback | 7/35 / 0 | **20/20 / 0** |
| YOLOv8n 640 | ratio <= 0.6 | 5/35 / 0 | 18/20 / 0 |
| YOLOv8n 640 | ratio <= 0.6, knee fallback | 7/35 / 0 | 18/20 / 0 |
| YOLOv8s 640 | ratio <= 0.6 | **18/35 / 0** | 18/20 / 0 |
| YOLOv8s 640 | ratio <= 0.6, knee fallback | **20/35 / 0** | **20/20 / 0** |

Floor-labelled frames with any detection, from the same dumps: clip 01
MediaPipe 10, YOLOv8n 320 14, YOLOv8s 320 11, YOLOv8n 640 15,
**YOLOv8s 640 25**; clip 02 20 in every setup. The ratio separates floor
from chair sitting in every setup: floor p50 0.30-0.48, chair-sitting p10
0.52-0.82 (the 0.52 is one YOLOv8s 640 clip 01 frame; p50 0.95).

Event view for YOLOv8s 640 on clip 01, which is what an alert depends on:

- First floor event (label 120.0-127.0 s): flagged from the first frame,
  120.0 s, and in 14 of 15 frames (127.0 s, as the person rises, reads
  0.66). Flagged ratios 0.25-0.47.
- Second floor event (label 135.0-144.5 s): the detector finds the person
  in 10 of 20 frames, three of them at confidence 0.38-0.49, below
  `min_confidence`. Confident ratio hits at 137.5, 140.0, 141.5, 143.5
  and 144.5 s. The first hit is 2.5 s after onset. Counting detected
  frames only, which is how `StateTracker` already treats gaps, a
  two-frame confirmation reports at 140.0 s. Letting 0.25-0.5
  detections outside the bed zone count as floor candidates (their
  ratios are 0.21-0.39) would confirm at 137.5 s.

Thresholds 0.55, 0.6 and 0.65 give the same recall within one frame;
0.6 sits in the gap between floor (<= 0.50) and chair sitting (>= 0.69).

MediaPipe's clip 02 false positives are 11 `in_bed`-labelled frames whose
box centre lies outside that clip's small bed zone
(`x 0.239-0.444, y 0.384-0.654`); the current rules already produce 10
of them. That is a zone-drawing issue, not a ratio issue.

### Limits

- About a third of detected floor frames have no visible ankle, so the
  ratio is undefined there. Using knees as the ground point recovers one
  frame per clip at no false-positive cost.
- At 320 px, clip 01 recall stays low because the detector misses the
  person, not because the rule misreads them. With YOLOv8s on 640x480
  input it rises to 18-20 of 35 (section 6).
- Frames mid-descent already read low: clip 01 at 119.0-119.5 s is
  labelled upright with ratio 0.46-0.48. Combined with `on_floor`
  bypassing hysteresis and `AGENT_FLOOR_LIMIT_SECONDS=0`, a person
  bending to pick something up could raise a critical alert. None of the
  555 non-floor frames triggered it here, but these clips contain almost
  no crouching; see section 9.

## 5. Fix 2: leg geometry (tested, weak)

`leg_ratio = (ankle_y - hip_y) / torso length` (aspect-corrected).

| Label | Clip 02 p10 / p50 / p90 |
| --- | --- |
| on_floor | 0.22 / 0.37 / 0.71 |
| sitting_up | 0.36 / 1.11 / 1.57 |
| upright | 1.15 / 1.37 / 2.03 |

It excludes standing but overlaps chair sitting, and it needs ankles just
as fix 1 does. As a tie-breaker (`ratio <= 0.6 or (leg <= 0.5 and ratio
<= 0.7)`) it changed nothing on either clip. `shin_ratio` (knee to
ankle, the kneeling signature) could not be evaluated: neither clip
contains kneeling. Verdict: do not ship fix 2 on this evidence.

## 6. Fix 5: detection (partly tested)

### Rotation test-time augmentation: harmful as implemented

Running YOLO on the frame rotated 90 degrees each way and mapping the
result back looked like the largest single gain: clip 01 floor frames
with a detection went from 14/35 to 33/35. **That gain is false.**

- In clip 01 the rotated passes detect a static "lying person" at almost
  exactly `(0.00, 0.61, 0.60, 0.87)`: YOLOv8n reports it in 143 of 240
  upright-labelled frames, 44 of 122 `in_bed`, 13 of 20 `absent`, and 33
  of 35 floor frames. In 120 frames it co-occurs with a confident upright
  person elsewhere in the frame. It is scenery, most likely bedding or
  the bed edge.
- The rotated passes find the actual person on the floor in 1 of 35
  frames (YOLOv8n) and 4 of 35 (YOLOv8s).
- The coordinate mapping is correct: when the upright and 270-degree
  passes both detect the person, boxes overlap at median IoU 0.69 (n=23,
  YOLOv8s).

Do not add rotation TTA without suppressing persistent static boxes
first, and expect little from it in this room even then.

### Larger model and input resolution

YOLOv8s on 640x480 input is the only setup that finds the person in most
of clip 01's floor frames (25/35, in both the earlier prediction run and
this session's dump), and combined with fix 1 it is the only setup that
catches both clip 01 floor events. Neither the larger model alone
(YOLOv8s 320: 11/35 detected) nor the larger input alone (YOLOv8n 640:
15/35) does it; it takes both.

Backend latency on the Mac mini, median per frame:

| Setup | Earlier prediction runs | This session's dump |
| --- | --- | --- |
| MediaPipe 320 | 12-18 ms | 18 ms |
| YOLOv8n 320 / 640 | 31 ms | 29-31 ms |
| YOLO11s 320 | 57 ms | not dumped |
| YOLOv8s 320 | 64 ms | 56-78 ms |
| YOLOv8s 640 | 64 ms | 64-83 ms (p90 85-103 ms) |

The dump timings overlap with gemma4 runs on the same GPU, so the upper
ends are inflated. At 2 fps even 100 ms per frame is well inside budget;
the open question is memory alongside the Docker stack and a vision
model, which was not measured. The 320 and 640 inputs cost about the same
because ultralytics resizes to its 640 default either way; what changes
is how much detail reaches it.

Shipping 640 input means changing the bridge capture size in
`services/embodiment/embodiment/static/script.js` (`FRAME_WIDTH`,
`FRAME_HEIGHT`, lines ~160-162), which roughly quadruples JPEG bytes on
`frames_raw`/`frames` (both capped at 50 entries).

## 7. Fix 3: local vision model as a floor check (tested)

Prompt, answering JSON only: "Is a person sitting, kneeling, or lying ON
THE FLOOR? A person on a bed, chair or sofa, standing, walking, or bending
over while standing is NOT on the floor." Sample: 48 positive frames
(labelled `on_floor` inside a scripted window) and 53 negatives
stratified across sitting_up, upright, in_bed and absent, from both
clips. `gemma4:e4b-mlx` through local Ollama, temperature 0, `think`
off.

| Input | Recall | False yes | Unclear on negatives | Latency median / p90 |
| --- | --- | --- | --- | --- |
| 320x240 bridge frame | 16/48 (clip 01 5/33, clip 02 11/15) | 0/53 | 9 | 1.6 s / 1.6 s |
| 640x480 | 6/48 (clip 01 2/33, clip 02 4/15) | 0/53 | 6 | 1.6 s / 2.1 s |

Reading:

- It never said yes when the person was not on the floor, in either run,
  so a yes is strong confirmation.
- It said no to most real floor frames, so a no must never cancel a floor
  detection from pose geometry. Use it to upgrade, not to veto.
- More pixels made recall worse, not better (16/48 to 6/48). The model is
  conservative with this prompt, and resolution is not what limits it.
  Before relying on it, try prompt variants (for example asking where the
  person's hips are: floor, bed, chair) and `qwen3-vl:8b` on
  human-verified labels.
- Both latency runs shared the GPU with pose dumps; treat the numbers as
  upper bounds for this model on the Mac mini.
- perceive's existing vision client is a different thing: it asks
  `moondream` (not installed on this machine) for a one-sentence
  description on state changes, via `PERCEIVE_VISION_MODEL`. A floor
  check would be a new, narrow request, not a reuse of that prompt.
- `qwen3-vl:8b` was not tested as a verifier: it produced the labels, so
  scoring it against them would be circular.

## 8. Object recognition of the floor itself

Question asked: would it be worth detecting the floor with object
recognition?

Not as a per-frame detector. The camera is fixed, so the floor does not
move; running segmentation on every frame costs GPU time and adds
nothing a one-time map does not already give.

As a **setup-time scene map**, yes, and it targets real gaps found here:

1. **Accurate bed and furniture outlines.** Zones are caregiver-drawn
   rectangles. Clip 02's bed zone is small enough that 11 in-bed frames
   fall outside it (MediaPipe), and two of clip 01's real floor
   detections beside the bed (125.5 s, 144.5 s at YOLOv8n 320) have their
   centre inside the bed rectangle. A semantic segmentation model
   trained on indoor scenes (ADE20K classes include floor, bed, chair,
   sofa, cabinet), run on a few empty-room frames and median-combined,
   would propose the mattress outline and a floor mask for the caregiver
   to confirm in the Zones editor.
2. **Ground contact on floor vs furniture.** Checking whether the hips or
   the lowest body point sit on floor pixels or bed/chair pixels
   separates "on the floor next to the bed" from "lying in bed" better
   than a rectangle does.
3. **Floor-plane calibration without waiting for standing frames.** The
   floor mask plus a monocular depth model gives the ground plane at
   setup, which fix 1 currently learns from the person walking around.

It does not fix clip 01's main failure: the person is not detected at
all in most of those frames, and knowing where the floor is cannot place
someone the detector never found.

Not tested in this session. `transformers` is not installed in
`.venv-video-eval`, and a model would need downloading.

## 9. Recommended next steps

Status after PR #55 (merged 2026-09-15):

- Item 1: done. Online Theil-Sen ground line in `perceive`, height-ratio
  rule confirmed over two detections, plus a fall-drop cue and a short
  `floor_suspect` hold that suppresses a false `absent` after a fall.
- Item 2: done, with a different model. Bridge frames are 640x480 and the
  default backend is YOLO11s-pose at 640 input, not YOLOv8s, which locked
  onto a static object on the bed in video 3.
- Item 3: partly. A pose now counts as in the bed zone only when both the
  landmark centroid and the box centre are inside it; the mattress-outline
  drawing guidance still applies.
- Item 4: done as an optional Ollama floor check outside the bed and door
  zones, requiring two consecutive positives (settings in `.env.example`).
- Item 5: open.
- Item 6: done for the live detector as the calibrated phantom-box filter
  (`perceive.calibrate_phantoms`, `PERCEIVE_PHANTOMS_FILE`); rotation TTA
  was not pursued.

The original recommendations follow, in order.

1. **Implement fix 1 in `perceive`.**
   - Online ground-line calibration inside `StateTracker` from frames
     confirmed `standing`/`walking` with confidence >= 0.5 and at least
     one ankle visible (>= 0.3). Robust Theil-Sen fit over a ring buffer;
     abstain until enough samples span enough of the frame height.
   - Optional install-time override, e.g. `PERCEIVE_GROUND_LINE=a,b`,
     so a restart does not lose calibration.
   - In `classify_pose`: outside the bed zone, confidence >= 0.5, ankles
     visible (knee fallback), `height_ratio <= 0.6` gives `on_floor`.
     Keep the existing rules.
   - Require two consecutive detected frames (1 s at 2 fps) for this rule
     before reporting, to absorb mid-descent and bending frames. Frames
     with no detection neither count nor reset, matching `StateTracker`.
     Existing floor rules keep their immediate path.
   - Consider letting 0.25-0.5 confidence detections outside the bed zone
     count towards that confirmation when their ratio is <= 0.6. On clip
     01's second event this moves the report from 140.0 s to 137.5 s.
     Measure false alarms before enabling it by default.
   - Unit tests with hand-built `PoseResult`s as in
     `tests/test_floor_and_dropout_rules.py`; then re-run
     `video_eval predict` for both clips and compare with section 3.
2. **Switch the bridge and backend to YOLOv8s on 640x480.** It is the one
   setup that, with fix 1, catches every floor event in both clips
   (frames: 20/35 and 20/20, no false floor frames). Change `FRAME_WIDTH`/
   `FRAME_HEIGHT` in `services/embodiment/embodiment/static/script.js`,
   set `PERCEIVE_POSE_BACKEND=yolo` and `PERCEIVE_YOLO_MODEL=yolov8s-pose.pt`,
   and check container memory with the full stack plus the Ollama model
   running. Recalibration is automatic, since the ground line is learned
   per setup.
3. **Tighten the bed zone guidance.** Draw the mattress outline, not a
   box around the bed. Re-draw both clips' zones and re-score.
4. **Vision check as an upgrade path.** When pose sees a person outside
   the bed zone with a ratio in an ambiguous band (for example 0.6-0.75),
   or the person vanishes low in the frame, ask the local model the
   floor question and treat a yes as `on_floor`. Never let a no cancel
   a pose-based floor.
5. **Setup-time floor and furniture segmentation** (section 8) as the
   zone editor's starting proposal.
6. **Static-box suppression** before any further TTA experiment.

## 10. Evaluation gaps to close before trusting any rate

- Human-verified labels for the floor windows, at least.
- More floor events, staged by a helper, not the older adult: sitting
  against the bed, kneeling, lying towards and away from the camera,
  lying half hidden by the bed, and crouching or bending that is *not*
  a fall.
- Long ordinary recordings (whole nights, daytime routines) to count
  false alarms per night.
- Event-level metrics: time from floor onset to first `on_floor`, and
  false alarm events per hour, instead of frame recall.

## 11. Reproducing

Scripts and their outputs are kept outside git with the recordings, in
`../data-ai-agent-dementia/analysis/floor-detection-2026-09-14/`:

| File | Purpose |
| --- | --- |
| `dump_features.py <out_dir> [tags]` | Runs the real perceive backends over both clips and writes per-frame keypoints, plus raw YOLO detections at 0/90/270 degrees. Setups: `mp-320`, `v8n-320`, `v8s-320`, `v8n-640`, `v8s-640`. |
| `floor_features.py <features_dir> <tag> [--frames] [--tta]` | Height/leg/shin ratio distributions by label; `--frames` prints each frame around the floor windows. |
| `floor_sim.py <features_dir> <tags...>` | Frame-level recall and false positives for the candidate rules. |
| `vlm_floor_check.py <model> <out.jsonl> [frame_key]` | Local Ollama floor question on the stratified sample. |
| `features/`, `vlm-*.jsonl` | Outputs used for the numbers in this document. |

Run them with `.venv-video-eval/bin/python` from the repository root and
`PYTHONPATH` set to that folder (the analysis scripts import each other).
