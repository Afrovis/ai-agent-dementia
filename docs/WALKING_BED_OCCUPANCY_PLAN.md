# Walking detection and sticky bed occupancy: plan (2026-09-16)

Owner request: improve standing/walking, think about zones and bed zones, and
make bed occupancy sticky ("once in bed, in bed until they come out"). Decide
later whether the product needs `walking` as a separate state.

## 1. Where we are

- `StateTracker._is_walking` (`services/perceive/perceive/classify.py`) calls a
  `standing` person `walking` when `max(x) - min(x)` of the centroid over the
  last 5 *frames* reaches 0.15 of the frame width.
  - Horizontal only: walking towards or away from the camera (the bed-to-door
    path in most rooms) never moves x far enough.
  - Frame-width units: the same step is a smaller fraction far from the camera.
  - The centroid switches between the landmark mean (all 9 keypoints) and the
    bbox centre, so a keypoint dropping in or out is a fake jump.
  - Any non-`standing` frame clears the history, and a mid-stride frame with a
    hidden ankle is `sitting_up`, so a walk rarely builds 2+ samples.
  - Window counts frames, but the motion gate drops frames: 5 frames can be
    2.5 s or 20 s.
- Newest report (clip 04, yolo11s letterbox640, gated): walking recall 28.6%,
  45 of 84 walking frames read as `standing`. Confirmed references exist for
  clips 01-04: 11 walking spans (108 s), 13 standing spans (201 s).
- `rule_replay.py` scores against the local VLM labels, which cannot tell
  walking from standing in one frame. It cannot measure this problem.
- The agent's IDLE -> OBSERVING trigger (`services/agent/agent/session.py:503`)
  fires on `sitting_up`/`standing` only. A better walking detector could turn a
  bed exit straight into `walking` and skip it.
- Bed hold today: `_holds_bed` holds `in_bed` only while *nothing* is detected.
  `bed_vanish_hold` (off by default) covers "lost in the bed zone before
  `in_bed` was confirmed". docs/BED_OCCUPANCY_2026-09-15.md: calibrated bed zone
  + vanish hold + thigh ratio gave 93% bed recall and 0 false exits.

## 2. Does walking help bed occupancy?

Yes, as *exit evidence*, not as a posture. A person who is in bed leaves it by
moving: standing up and then walking out of the bed zone. A stationary
`standing` reading inside the bed zone (bed-edge sitting misread, blanket
shapes, the zone reaching the wall behind the bed) is the main false exit.
So the sticky bed rule is:

- **Enter** occupancy: `in_bed` confirmed (existing), or vanish in the bed zone
  when not walking (existing `bed_vanish_hold`).
- **Stay** while: undetected (existing), low-confidence (existing), seated or
  lying in the bed zone, or upright but *stationary* with the feet in the bed
  zone.
- **Leave** only on: upright with the ground point (feet) outside the bed zone,
  or measurable walking, or a door/bathroom zone, or `on_floor` (never
  suppressed: safety first).

While occupied, an upright-but-stationary-in-bed frame reports `sitting_up`,
not `in_bed`: the agent's wake-up path (IDLE -> OBSERVING) keys on
`sitting_up`, so waking someone who sits up is unchanged. What changes is that
it no longer becomes `standing`/`walking` (= "out of bed") without evidence.

Zones matter here: the rule uses the *ground point* (ankle midpoint, else bbox
bottom-centre) against the bed polygon, because a standing person's centroid
next to the bed often falls inside a zone drawn over the wall behind it. That
works best with a `perceive.calibrate_bed` zone (bed surface, grown 15% up),
not a hand-drawn rectangle.

## 3. Work, in priority order

| # | Item | Why first | Done when |
|---|------|-----------|-----------|
| P0 (done) | `tools/video_eval/scripts/state_sweep.py`: replay caches, score against confirmed `reference.yaml` (exact per-state P/R, standing/walking F1, upright-collapsed, bed-occupancy P/R, bed-exit latency, false bed exits, false transitions), grid-search `--grid key=v1,v2`, leave-one-clip-out | Can't tune what we can't measure; existing replay can't see walking | Baseline numbers for clips 01-04 recorded here |
| P1 (done, opt-in) | New motion feature: time-window (seconds), ground-point track, 2D displacement + bbox-height change, normalised by person height, short gap tolerance instead of clearing history on one non-standing frame | Root causes in section 1 | Walking F1 and standing F1 both up on held-out clips; no floor/bed regression |
| P2 (done) | Agent: IDLE -> OBSERVING also on `walking` | One-line safety fix enabled by P1 | Unit test |
| P3 (done, opt-in) | Sticky bed occupancy (`bed_latch`) per section 2, plus `upright_zone_from_feet` | Owner request; depends on P1 motion | Bed recall up, false bed exits not up, every scripted bed exit still caught |
| P4 (recommendation, section 5) | Pick defaults from the sweep; document in `ClassifyThresholds`, `.env.example`, this file | | |
| later | Decide if `walking` stays a separate state (evidence from P0/P1); IR night clips; caches for clip 05; caregiver-drawn door zone | Needs owner / new data | |

Guard rails: the old walk rule stays reachable (`walk_mode=legacy`) until the
new one wins on held-out clips; floor rules and the 2 s `on_floor` latency must
not regress; everything is replay-tested on cached detections (no model runs).

Data limits: 4 short, lamp-lit RGB clips of one person, one camera. Thresholds
chosen here are a starting point, not a validated deployment setting.

## 4. Results (2026-09-16)

All numbers: `yolo11s-pose-letterbox640` cached detections, motion-gated,
clips 01-04 (1,357 scored frames, 8 scripted bed exits), scored by
`tools/video_eval/scripts/state_sweep.py` against the confirmed
`reference.yaml`. "Recommended" = the 2026-09-15 bed settings
(`bed_vanish_hold`, `sitting_thigh_ratio=0.55`, `absent_confirm_seconds=3`,
`hold_floor`). "Segmented zones" = `perceive.calibrate_bed` on every 10th
640 letterbox frame, default 15% grow-up; the clips' own `zones.yaml` are
the old hand-drawn rectangles (clip 01's covers 63% of the width).

| Config | acc | upright acc | standing F1 | walking F1 | bed R / P | floor R | bed exits | exit delay | false exits | false floor | upright frames in bed zone |
|---|---|---|---|---|---|---|---|---|---|---|---|
| production defaults, clip zones | 0.54 | 0.67 | 0.60 | 0.18 | 0.71 / 0.91 | 0.55 | 6/8 | 0.9 s | 6 | 1 | 219 |
| recommended, clip zones | 0.55 | 0.70 | 0.57 | 0.18 | 0.87 / 0.83 | 0.56 | 6/8 | 0.9 s | 10 | 1 | 232 |
| + walk 0.08, confirm 2 | 0.56 | 0.70 | 0.54 | 0.36 | 0.88 / 0.81 | 0.54 | 7/8 | 0.2 s | 12 | 1 | 226 |
| recommended, **segmented zones** | 0.59 | 0.74 | 0.56 | 0.19 | 0.85 / 0.95 | 0.87 | 6/8 | 0.9 s | 3 | 0 | 89 |
| + walk 0.08, confirm 2 | 0.59 | 0.74 | 0.54 | 0.36 | 0.86 / 0.95 | 0.83 | 7/8 | 0.2 s | 4 | 0 | 86 |
| + `upright_zone_from_feet` | 0.59 | 0.74 | 0.54 | 0.36 | 0.86 / 0.95 | 0.83 | 7/8 | 0.2 s | 5 | 0 | **20** |
| + `bed_latch` | 0.59 | 0.73 | 0.54 | 0.36 | 0.86 / 0.95 | 0.83 | 7/8 | 0.2 s | 5 | 0 | 19 |

Leave-one-clip-out over `walk_displacement_threshold` x `confirm_frames`
(segmented zones, feet zone on) picked 0.08 / 2 on every fold, so the pooled
held-out row equals the last-but-one row above.

Findings:

1. **The bed zone is the biggest lever.** Segmented zones cut false bed exits
   12 -> 4, false floor episodes 1 -> 0, raised floor recall 0.54 -> 0.83
   (floor rules skip the bed zone, and the rectangles covered floor) and bed
   precision 0.81 -> 0.95. Nothing else in this work comes close.
2. **The walking labels cannot separate walking from standing.** Median
   speed of labelled `standing` spans is 0.02-0.19 body heights/s, of
   `walking` spans 0.09-0.31; clip 04 has a walking span and a standing span
   both at 0.11. Clip 01's 54 s `standing` span is making the bed and getting
   dressed. No motion feature tried (centroid, ground point, box size; 1-3 s windows)
   scores above AUC 0.61 on these labels.
   The tuned legacy rule (0.08, confirm 2) doubles walking F1 to 0.36 at a
   small standing cost; the new `walk_mode=ground` scored 0.30 pooled and
   0.15 held-out, so it stays opt-in. Better labels are needed before any
   walking rule can be judged (see section 5).
3. **Walking as exit evidence works through zones, not labels.** With the
   feet-based zone, upright frames reported in the bed zone while the person
   was off it fell 86 -> 20, which is what feeds `bed_vanish_hold` false
   `in_bed` holds and the agent's "standing at the bed counts as progress".
4. **`bed_latch` has no measurable effect once the zone is segmented**: an
   upright person's feet are almost never inside a tight bed polygon. It is
   kept, opt-in, for rooms where the bed zone cannot be drawn tightly.
5. **The one "missed" bed exit is a label error.** Clip 01's reference has
   `in_bed` until 190 s, but review frame 340 (170 s) shows sitting up and
   frame 360 (180 s) shows walking in front of the bed; perceive reports
   `standing` from 177.5 s, which also counts as one of the false exits.
6. Clip 04 is still the weakest (upright acc 0.59, standing F1 0.27).
   Pooled over all clips, the best config still reads 27 walking frames as
   `absent` and 51 standing frames as `sitting_up`.

### Visual check

Side-by-side review videos for clips 01-04 live in
`data-ai-agent-dementia/analysis/walking-bed-2026-09-16/`
(`<clip>__pipeline__baseline.mp4`, `<clip>__pipeline__improved.mp4`). The
segmented zones they use are in its `zones/` folder. "Baseline" is
`perceive.main`'s production defaults with the clips' own zones; "improved" is
the last-but-one config in the table above. Each frame shows the reference
state next to the pipeline state (match / both upright / differs), the feet
point as a cross (magenta when on the bed), and a two-row timeline (pipeline
above, reference below). To regenerate:

```bash
python tools/video_eval/scripts/state_sweep.py --data-root ../data-ai-agent-dementia \
    --zones-dir ../data-ai-agent-dementia/analysis/walking-bed-2026-09-16/zones \
    --set bed_vanish_hold=true --set sitting_thigh_ratio=0.55 \
    --set absent_confirm_seconds=3 --set hold_floor=true \
    --set walk_displacement_threshold=0.08 --set confirm_frames=2 \
    --set upright_zone_from_feet=true \
    --predictions-tag replay_yolo11s-pose-letterbox640-improved
python -m video_eval --data-root ../data-ai-agent-dementia visualize \
    --clip <clip> --mode pipeline \
    --pipeline-tag replay_yolo11s-pose-letterbox640-improved \
    --zones ../data-ai-agent-dementia/analysis/walking-bed-2026-09-16/zones/<clip>.yaml \
    --label improved --output-dir ../data-ai-agent-dementia/analysis/walking-bed-2026-09-16
```

Rendering these exposed two renderer bugs, fixed here: the review frame was
never scaled up to its panel, so every overlay was misplaced, and the overlay
mapping assumed a 16:9 camera (clip 04 is 3:2).

## 5. Recommendations

1. Run `perceive.calibrate_bed` in the real room and turn on
   `PERCEIVE_BED_VANISH_HOLD=true`, `PERCEIVE_SITTING_THIGH_RATIO=0.55`,
   `PERCEIVE_UPRIGHT_ZONE_FROM_FEET=true`, `PERCEIVE_CONFIRM_FRAMES=2`,
   `PERCEIVE_WALK_THRESHOLD=0.08`. Check a live night before making any of
   them the code default.
2. Fix clip 01's bed exit time (about 176 s, not 190 s) in `reference.yaml`.
3. Walking vs standing as separate states: the agent never needs the
   difference (goals key on zones, sessions on "upright"), and neither the
   labels nor the detector separate them. Recommend treating them as one
   "upright" state for decisions, and only relabelling (short spans, "moving"
   vs "still") if a feature needs it.
4. Save the segmented zones next to each clip (for example
   `zones.segmented.yaml`) so replays use the same zones as the room.
5. Still open: IR night clips, caches for clip 05, a door zone.
