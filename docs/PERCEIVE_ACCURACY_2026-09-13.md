# Perceive accuracy quick wins: handoff (2026-09-13)

Two sessions on the same day. Sections 1 to 6 are the first session as it
was handed over; sections 7 onward continue it and supersede section 5.

State of an in-flight investigation into why the live pose pipeline missed
bed and floor events in the two recorded bedroom clips
(`PIPELINE_TESTING.md`, untracked in the main checkout). Branch
`perceive-accuracy-quick-wins`, worktree
`.claude/worktrees/perceive-accuracy`, branched from
`issue-52-embodiment-bridge-aspect-ratio` at `fbb83bc`. Everything below is
committed on that branch. Nothing is merged and the live bridge is unchanged.

## 1. What the document got wrong

`PIPELINE_TESTING.md` concluded that MediaPipe and YOLO share a blind spot
on bed and floor states. Re-reading the prediction files showed three
pipeline causes that are not model limits:

1. Neither clip had a `zones.yaml`, so every frame scored zone `other`.
   `in_bed` requires the bed zone and the blanket hold only engages once
   `in_bed` is reached, so the eval could never emit `in_bed`.
2. YOLO boxed many of the "missed" in-bed frames at confidence 0.27 to
   0.49; the 0.5 floor turned them into `absent` (97 of 344 detections in
   clip 1).
3. ultralytics zeroes the coordinates of any keypoint under 0.5
   confidence. Those (0, 0) points made a lying person's hidden hips read
   as an upright torso and stretched every body extent to the frame edge,
   so lying people classified as `sitting_up` and floor frames never met
   the lying test.

Resolution matters for the above-blanket case: YOLOv8n on 640-wide frames
boxed 27 of 27 in-bed frames in clip 2 where the 320 bridge boxed 6. The
under-blanket case (clip 1, 11 to 35 s) is invisible to every detector at
every size; only the zone hold or a vision model covers it.

The local VLM reference has a known error: it labels clip 1 at 36.0 to
45.5 s as `absent` while the script has the person in bed until 46 s. Any
run that reaches `in_bed` scores 0/20 on `absent` for that reason and is
right, not wrong.

## 2. Code changes on the branch

`services/perceive`:

- `classify.py`: new `ClassifyThresholds.presence_confidence` (0.25,
  `PERCEIVE_PRESENCE_CONFIDENCE`). A detection between it and
  `min_confidence` in the bed zone is `in_bed`; anywhere else it holds the
  tracker's current state instead of publishing `absent`. Body extents now
  come from `bbox` (identical for MediaPipe, correct for YOLO).
  `centroid_of` uses the landmark mean only when all nine landmarks are
  present, else the box centre. A frame with no detection no longer resets
  a pending confirmation (it still publishes `absent` immediately).
- `backends.py`: YOLO drops keypoints ultralytics zeroed; `PoseResult`
  contract now allows missing keys. `MediaPipeBackend(static_image_mode=)`
  and `build_backend(..., static_image_mode=)`.
- `main.py`: config keys `PERCEIVE_YOLO_MODEL`, `PERCEIVE_MEDIAPIPE_VIDEO_MODE`,
  `PERCEIVE_PRESENCE_CONFIDENCE`, wired through `build_tracker` and `run`.
- Tests: `tests/test_presence.py`, `tests/test_partial_landmarks.py`.
  95 pass, ruff clean.

`tools/video_eval`:

- New bridge variant `letterbox640` (640x480) via `paths.BRIDGE_VARIANTS`;
  `prepare`, `predict`, `zones`, `replay` all read variants from it.
- `predict` gained `--yolo-model`, `--presence-confidence`,
  `--mediapipe-video-mode`; the tag carries the model stem or `_video`.
- `scripts/vlm_agreement.py`: stop-gap scorer against `labels/local.jsonl`
  and the `clip.yaml` script until `reference.yaml` exists. 32 tests pass.

`.env.example` documents the three new keys.

## 3. Data changes (outside git, `../data-ai-agent-dementia`)

- `clips/*/zones.yaml`: a `bed` polygon per clip, derived from YOLO boxes
  on VLM `in_bed` frames with zero margin, in letterbox coordinates. In
  clip 1 the floor events lie immediately right of the bed edge (x 0.64
  to 0.69 against a bed edge at 0.63), so any margin swallows them. These
  are stand-ins; a human should draw them in the dashboard Zones editor.
  Squash-variant runs use the wrong coordinate frame with these zones.
- `clips/*/bridge-letterbox-640/` and `bridge_letterbox_640_path` in
  `frames.jsonl`.
- 16 new prediction files per the matrix below. Tags with `p50` were run
  with `VIDEO_EVAL_GIT_SHA=p50-<sha>` and presence disabled (0.5); the meta
  files carry the real parameters.
- `yolov8s-pose.pt` (22 MB) downloaded into the main checkout root next to
  the untracked `yolov8n-pose.pt`. Neither belongs in git.

## 4. Results (per-frame agreement with VLM posture, gated runs)

Clip 1, 244 s, includes under-blanket sleep and two floor lies:

| run | agree | in_bed | upright | events matched |
|---|---|---|---|---|
| mediapipe squash, no zones (document baseline) | 0.30 | 0.00 | 0.49 | 2/9 |
| mediapipe letterbox + zones | 0.48 | 0.74 | 0.53 | 7/9 |
| mediapipe letterbox, video mode | 0.56 | 0.76 | 0.66 | 7/9 |
| yolo n letterbox 320 | 0.56 | 0.49 | 0.81 | 7/9 |
| yolo n letterbox 320, presence off | 0.43 | 0.00 | 0.71 | 4/9 |
| yolo n letterbox 640 | 0.52 | 0.22 | 0.81 | 5/9 |
| yolo s letterbox 320 | 0.57 | 0.49 | 0.82 | 6/9 |
| yolo s letterbox 640 | 0.52 | 0.00 | 0.86 | 4/9 |

Clip 2, 114 s, above-blanket bed, sitting on floor:

| run | agree | in_bed | sitting_up | events matched |
|---|---|---|---|---|
| mediapipe squash, no zones (document baseline) | 0.50 | 0.00 | 0.61 | 9/11 |
| mediapipe letterbox + zones | 0.59 | 0.00 | 0.75 | 10/11 |
| yolo n letterbox 320 | 0.65 | 0.23 | 0.55 | 10/11 |
| yolo n letterbox 640 | 0.73 | 0.63 | 0.84 | 10/11 |
| yolo n letterbox 640, presence off | 0.65 | 0.43 | 0.70 | 10/11 |
| yolo s letterbox 640 | 0.72 | 0.57 | 0.98 | 10/11 |

Reading:

- Zones plus the presence rule plus the keypoint fix lift every backend on
  both clips. YOLO beats MediaPipe on both.
- 640 input is a clear win on clip 2 and a loss on clip 1's `in_bed`
  (0.22 versus 0.49 at 320). Not yet explained; the bridge constant in
  `services/embodiment/embodiment/static/script.js` was left at 320x240
  because the numbers conflict. HANDOFF.md section 12 says not to change
  it without numbers.
- `on_floor` is 0 of 35 frames in every run on clip 1 even though YOLO
  boxes the lying person at 0.55 to 0.93 confidence at 120 to 127 s and
  140 to 145 s with box centroid around (0.66, 0.85) in review
  coordinates. The lying rule or the zone must be failing there; unresolved.
- Clip 2's "sitting on floor" is not lying, so the classifier cannot
  express it. Expected, not a bug.
- MediaPipe video mode helps clip 1 and slightly hurts clip 2; left off.

## 5. Next steps, in order (first session; status in section 10)

1. Explain clip 1 `on_floor` = 0. Run `predict` at 320 and 640 with
   `--no-gate`, dump per-frame `bbox`, zone and state for 118 to 146 s, and
   check the lying test (`vertical <= 0.5 * horizontal`), `centroid_y >= 0.6`
   in letterbox coordinates, and whether the box centre falls inside the
   derived bed polygon. Suspect the polygon's right edge or the extent ratio.
2. Explain why 640 loses `in_bed` on clip 1 (confusion shows in_bed going
   to `sitting_up` and `absent`). Likely the covered person at 640 gets
   confident partial detections classified upright by `torso_vertical_ratio`.
3. Decide the bridge resolution with both clips in hand, then change the
   page constants and the embodiment tests that assert 320x240.
4. Have the owner draw real zones in the dashboard and rerun.
5. Fix the labeller `person_visible` bug (HANDOFF.md section 13) and
   relabel so the reference stops calling a covered sleeper `absent`.
6. Consider the "lost in room" rule (absent only after passing the door
   zone) and the async VLM second opinion; both are designed in the
   conversation that produced this file but not built.

## 6. Commands

Run from the worktree. The venv's editable installs point at the main
checkout, so the worktree must lead `PYTHONPATH`.

```sh
WT=$PWD
PY=/Users/mathiasserver/Documents/ai-agent-dementia/.venv-video-eval/bin/python
export PYTHONPATH=$WT/services/perceive:$WT/services/capture:$WT/tools/video_eval:$WT/shared
DATA=/Users/mathiasserver/Documents/data-ai-agent-dementia

$PY -m pytest -q services/perceive/tests tools/video_eval/tests
$PY -m ruff check services/perceive tools/video_eval

$PY -m video_eval --data-root $DATA predict --force --clip 2026-09-13_bedroom-sample-01 \
  --backend yolo --variant letterbox640 --yolo-model /Users/mathiasserver/Documents/ai-agent-dementia/yolov8n-pose.pt
$PY tools/video_eval/scripts/vlm_agreement.py --data-root $DATA --clip 2026-09-13_bedroom-sample-01 --confusion
```

The complete matrix script used for the tables lives only in the job's
temporary directory and is reproduced by the eight `predict` invocations per
clip implied by section 4.

## 7. Second session: method

Re-running `predict` for every rule tweak re-runs the model, so the second
session split inference from classification.

- `tools/video_eval/scripts/detection_cache.py` runs a model once over every
  prepared bridge frame and stores every candidate box above 0.1 confidence,
  all 17 keypoints with their confidences, CPU latency, and the motion-gate
  decision. Output: `clips/<clip>/cache/<model>-<variant>.jsonl` and
  `gate-<variant>.json`, in the private data root with the rest of the data.
- `tools/video_eval/scripts/rule_replay.py` rebuilds the `PoseResult` the
  live backend would produce (0.25 model floor, zeroed keypoints dropped),
  drives the real `StateTracker`, and scores with `vlm_agreement.py`. It adds
  a false floor episode count: a run of `on_floor` frames with no VLM
  `on_floor` frame in it. The agent pages on the first `on_floor` frame
  (`AGENT_FLOOR_LIMIT_SECONDS=0`), so episodes are what a caregiver feels.
- Parity: the replay reproduces the section 4 numbers for YOLOv8n and v8s at
  320 exactly (0.56 / 0.49 / 7 of 9 and 0.57 / 0.49 / 6 of 9), and a real
  `predict` run with the final rule set matches the replay exactly on both
  clips (section 9).
- Six YOLO pose models (v8n, v8s, 11n, 11s, 26n, 26s; the new weights live in
  `../data-ai-agent-dementia/models/`) and MediaPipe in image and video mode,
  each at letterbox 320 and 640, on both clips. The scoring reference is
  still the first-session VLM labels, so the tables stay comparable with
  section 4; section 11 covers the relabel.
- Rules and thresholds were chosen on clip 1 only. Clip 2 is the hold-out.

## 8. Findings

### 8.1 Clip 1 `on_floor` = 0 (first-session step 1)

Not the zone and not the centroid height. Every YOLO box on a floor frame is
taller than wide (height/width 1.2 to 2.4 against the 0.5 lying test), with
its bottom on the letterbox edge (y 0.87). The VLM notes say why: "sitting on
floor near bed" (120 to 127 s), "bent over near bed" (135 to 139 s), "on floor
near dresser and bed" (140 to 144 s). Clip 2's floor event is also sitting.
The label prompt defines `on_floor` as lying, kneeling or crawling, and the
scorer already maps the scripted `sitting_on_floor` to `on_floor`, but the
classifier only ever recognised a body stretched across the frame.

What separates them is where the box starts. Outside the bed zone on clip 1
(YOLOv8n, 320), the box top sits at a median 0.27 (95th percentile 0.51) for
`upright` frames and a median 0.59 (5th percentile 0.46) for `on_floor`.

### 8.2 640 losing `in_bed` on clip 1 (first-session step 2)

Not upright misreads, as guessed. At 640 YOLOv8n puts fewer 0.25+ boxes on
the covered sleeper: 90 `in_bed` frames go to `absent` against 58 at 320.
The bed-vanish rule below removes the difference (0.89 at 640, 0.91 at 320).

### 8.3 The dominant error was `absent`, not posture

On clip 1 at 320 with YOLOv8n, 131 of 461 labelled frames were `absent`
while a person was plainly there: one- to four-frame detector dropouts
published `absent` immediately, and every recovery then waited three
confirming frames. Worse, getting under the blanket is itself the moment the
detector loses the person (11.5 to 22 s, 90 to 98 s), so the tracker never
saw a lying pose, never confirmed `in_bed`, and the bed hold never engaged.

### 8.4 Static false positives: not worth a filter

A fixed box at (0.37, 0.13, 0.55, 0.30), confidence 0.26 to 0.40, appears on
clip 1 (14 frames for YOLOv8n at 320) and is picked over the real person in
15 of 310 detected frames. A "same box for five frames" filter would also
drop the real person sitting still on clip 2, so none was built.

### 8.5 MediaPipe and the box-top rule do not mix

MediaPipe's box is the envelope of its nine landmarks, which jumps when a
limb is misplaced. Even before any new rule it produced 5 false floor
episodes of 6 on clip 2 at 320; with the rules, 2 to 3 per run.

## 9. Rules added and results

Four thresholds in `ClassifyThresholds`, each with an env key. All four are
off in `ClassifyThresholds` itself so the existing unit tests keep their
meaning; `PerceiveConfig` turns two of them on:

| key | rule | `PerceiveConfig` default |
|---|---|---|
| `PERCEIVE_FLOOR_TOP_Y` | outside the bed zone, a 0.5+ detection whose box top is at or below this is `on_floor` | off (1.01) |
| `PERCEIVE_ABSENT_CONFIRM_SECONDS` | `absent` only after this long with no detection; the last state is held meanwhile | **3** |
| `PERCEIVE_BED_VANISH_HOLD` | a person lost while last seen in the bed zone confirms `in_bed` (then the bed hold applies) | off |
| `PERCEIVE_HOLD_FLOOR` | `on_floor` is held while nobody is detected | **on** |

Tests: `services/perceive/tests/test_floor_and_dropout_rules.py`.

"Rules" below means all four: floor top 0.5, absent 3 s, bed vanish, floor
hold. Agreement is per-frame agreement with the VLM; "ev" is scripted events
matched; "fl" is `on_floor` recall; "FE" is false floor episodes of all floor
episodes.

Letterbox 320:

| model | c1 base | c1 rules | c1 in_bed | c1 fl | c1 ev | c1 FE | c2 base | c2 rules | c2 in_bed | c2 fl | c2 ev | c2 FE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n | 0.56 | **0.76** | 0.91 | 0.51 | 9/9 | 0/2 | 0.65 | **0.82** | 0.73 | 0.95 | 11/11 | 0/1 |
| yolov8s | 0.57 | 0.75 | 0.89 | 0.54 | 9/9 | 1/3 | 0.64 | 0.78 | 0.13 | 0.95 | 11/11 | 0/1 |
| yolo11n | 0.51 | 0.76 | 0.91 | 0.54 | 9/9 | 0/2 | 0.73 | 0.90 | 0.93 | 0.95 | 11/11 | 0/1 |
| yolo11s | 0.66 | 0.76 | 0.91 | 0.54 | 9/9 | 0/2 | 0.73 | 0.90 | 0.80 | 0.95 | 11/11 | 0/1 |
| yolo26n | 0.46 | 0.74 | 0.90 | 0.54 | 9/9 | 0/2 | 0.63 | 0.91 | 0.83 | 0.90 | 11/11 | 0/1 |
| yolo26s | 0.48 | 0.76 | 0.89 | 0.54 | 9/9 | 0/2 | 0.59 | 0.81 | 0.70 | 0.95 | 11/11 | 0/1 |
| mediapipe | 0.48 | 0.68 | 0.91 | 1.00 | 9/9 | 0/2 | 0.59 | 0.68 | 0.00 | 1.00 | 9/11 | 3/4 |
| mediapipe video | 0.56 | 0.65 | 0.92 | 0.34 | 9/9 | 0/3 | 0.56 | 0.69 | 0.00 | 1.00 | 9/11 | 2/3 |

Letterbox 640:

| model | c1 base | c1 rules | c1 in_bed | c1 fl | c1 ev | c1 FE | c2 base | c2 rules | c2 in_bed | c2 fl | c2 ev | c2 FE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n | 0.52 | 0.76 | 0.89 | 0.54 | 9/9 | 0/2 | 0.73 | 0.85 | 0.73 | 0.95 | 11/11 | 0/1 |
| yolov8s | 0.52 | **0.80** | 0.89 | 0.80 | 9/9 | 0/3 | 0.72 | 0.84 | 0.57 | 0.95 | 11/11 | 0/1 |
| yolo11n | 0.66 | 0.74 | 0.90 | 0.54 | 9/9 | 0/2 | 0.66 | 0.77 | 0.00 | 0.95 | 10/11 | 0/1 |
| yolo11s | 0.63 | 0.79 | 0.90 | 0.74 | 9/9 | 0/2 | 0.73 | 0.84 | 0.70 | 0.95 | 11/11 | 0/1 |
| yolo26n | 0.54 | 0.75 | 0.88 | 0.54 | 9/9 | 0/2 | 0.66 | **0.88** | 0.73 | 0.95 | 11/11 | 0/1 |
| yolo26s | 0.52 | 0.78 | 0.88 | 0.63 | 9/9 | 0/2 | 0.73 | 0.84 | 0.67 | 0.95 | 11/11 | 0/1 |
| mediapipe | 0.45 | 0.69 | 0.91 | 1.00 | 9/9 | 0/2 | 0.62 | 0.73 | 0.30 | 1.00 | 11/11 | 3/4 |
| mediapipe video | 0.53 | 0.71 | 0.92 | 0.77 | 9/9 | 0/3 | 0.61 | 0.79 | 0.67 | 1.00 | 11/11 | 1/2 |

Single rules on clip 1, YOLOv8n at 320 (agree / events): baseline 0.56 / 7;
absent 3 s 0.67 / 7; bed vanish 0.71 / 7; floor top 0.5 0.58 / 9 (0 false
episodes of 3). On the hold-out clip the same single rules take YOLOv8n from
0.65 to 0.67, 0.72 and 0.73, and to 0.82 with all four.

Sensitivity on clip 1 with the other rules on (five models): floor top 0.45
to 0.55 is flat; 0.6 drops floor recall to 0.11 to 0.43. Absent confirmation
of 1 to 8 s is flat within 0.02. Presence 0.25 is best (0.35 costs `in_bed`
0.2 to 0.3). Min confidence 0.5 is best; 0.4 gives YOLO11s a false floor
episode.

End-to-end check through `predict` and `PerceiveConfig.from_env`, YOLOv8n at
320 with the four rule keys set: clip 1 0.76, `in_bed` 0.91, `on_floor`
0.51, 9 of 9 events, median latency 1.0 s; clip 2 0.82, 11 of 11, median
0.0 s. Identical to the replay.

### 9.1 Latency (single frame, CPU, Mac mini M4, median)

| backend | 320 | 640 |
|---|---|---|
| yolov8n / 11n / 26n | 30.8 / 34.0 / 34.4 ms | 29.3 / 31.4 / 32.9 ms |
| yolov8s / 11s / 26s | 59.1 / 58.5 / 62.1 ms | 57.4 / 56.5 / 58.3 ms |
| mediapipe / video mode | 18.6 / 12.0 ms | 19.2 / 12.4 ms |

Ultralytics resizes to 640 either way, so bridge resolution does not change
inference cost. Every backend fits a 500 ms, 2 fps budget many times over.
