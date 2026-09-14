# Perceive accuracy quick wins: handoff (2026-09-13)

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

## 5. Next steps, in order

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
