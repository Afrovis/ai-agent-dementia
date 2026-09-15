# In bed or out of bed: accuracy on clips 1-4 (2026-09-15)

`perceive` often got the one reading the agent cares most about at night
wrong: is the person in bed. This note measures that on the four recorded
bedroom clips, explains where the errors come from, answers whether the bed
should be segmented before the real-time algorithm runs, and records what
changed as a result.

## Setup

- Clips `2026-09-13_bedroom-sample-01` to `-03` and `2026-09-14_bedroom-sample-04`
  (videos 1-4), 2 fps bridge frames, 1,429 scored frames in total.
- Detector: YOLO11s-pose at letterbox 640, the production default. Its
  per-frame boxes and keypoints were already cached in each clip's
  `predictions/*.jsonl`, so every rule change below was replayed through the
  real `StateTracker` offline without re-running YOLO. With no changes the
  replay reproduces the stored predictions exactly.
- Truth: the manual `labels/reference.yaml` timelines. The local VLM labels
  were used only to adjudicate disagreements.
- **Bed occupancy** is binary. The reference says "in bed" when its zone is
  `bed` (lying, or sitting up on the bed). The prediction says "in bed" for
  `in_bed`, or `sitting_up` in the bed zone. Precision matters most: a false
  "in bed" while the person is up is a missed bed exit.
- Bed exits and entries are scored as events: caught if the prediction
  follows within 10 s. False exits and entries are prediction flips while
  the reference stays put for 2 s either side.

### Clip 1's reference

Clip 1's timeline was transcribed from the recording script, not annotated
against the video. In two stretches the detector (confidence 0.9, upright
torso, full-height box walking across the frame) and the local VLM
(`upright`) both see a person on their feet where the script says `in_bed`:

| Script says | Detector and VLM see | Used in "corrected" |
| --- | --- | --- |
| 0-46 s `in_bed` | walking 4-11 s, into bed at 11 s | 4-11 s `walking` |
| 152-190 s `in_bed` | up and moving from 176.5 s | 176.5-190 s `standing` |

Numbers below are against this corrected reference unless marked
"original". Both versions are given for the headline result. **Please check
these two stretches against the video**; if the script was right, use the
"original" rows.

## Baseline

Production defaults (`PERCEIVE_BED_VANISH_HOLD=false`), existing zones:

| Clip | Bed accuracy | Recall | Precision | False bed frames | Missed bed frames |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.962 | 0.914 | 0.988 | 2 | 16 |
| 2 | 0.991 | 0.977 | 0.977 | 1 | 1 |
| 3 | 0.942 | 0.916 | 0.962 | 6 | 14 |
| 4 | 0.741 | 0.020 | 0.040 | 24 | 49 |
| **All** | **0.915** | **0.820** | **0.917** | **33** | **80** |

Against the original clip 1 reference the baseline is 0.889 overall
(clip 1: 0.891). Clip 4 is effectively broken: it finds 1 of 50 bed frames
and calls 24 out-of-bed frames "in bed".

## Where the errors come from

1. **The bed zone is in the wrong place.** Clips 3 and 4 had provisional
   rectangles derived from YOLO boxes, and clips 1 and 2 hand-drawn ones.
   Laid over the frame, all four take in wall above the bed, clip 4's also
   covers the floor where people stand, and clips 1 and 4 miss the foot of
   the mattress. With these zones, 51% of confident standing/walking frames
   off the bed land in the bed zone.
2. **Sitting on the bed edge reads as `standing`.** The torso is upright
   and the ankles hang below the hips, which is the whole standing test.
   Knees were never looked at. (Clip 4, 31-33 s and 45-52 s.)
3. **Getting under the blanket loses the detection**, and with
   `PERCEIVE_BED_VANISH_HOLD=false` the tracker holds its last state
   (`standing`) and then reports `absent`. (Clip 4, 34-45 s.)
4. **Walking near or across the bed** in the image, with hips inside an
   oversized zone. (Clip 4, 26-30 s, 98-106 s, 123-132 s.)
5. **A chair beside the bed.** A person seated on it projects onto the bed
   in 2D (clip 2, 23-30 s). No single-camera geometry fully separates this.
6. **Reference errors**, as above (clip 1).

## Should the bed be segmented first?

**Yes, once, as calibration, not per frame.**

A COCO instance-segmentation model (`yolo11m-seg`, class `bed`) found the
bed in 137 of 137 sampled frames across the four clips (median confidence
0.65-0.89), with people and blankets in shot. Its outline follows the
actual mattress in all four rooms, and overlaps the existing zones poorly
(IoU 0.23-0.39).

It should not run per frame:

- The bed does not move. A per-frame mask adds latency and a second model to
  the 2 fps loop, and the answer is identical every frame.
- A per-frame mask is worse exactly when it matters. The person and the
  blanket hide the bed, so the mask shrinks or disappears while someone is
  in it.
- Occupancy is not "person pixels overlap bed pixels". Someone standing in
  front of the bed overlaps it in 2D too. The zone only says where the bed
  is; posture still has to say whether the person is on it.

What works is segmenting across a few dozen frames at setup, voting the
masks so no single occlusion matters, and writing the result as the `bed`
polygon. That is `perceive.calibrate_bed`.

One adjustment was needed. The mask is the mattress surface, while a body
lying or sitting on it rises above that surface in the image, so its
centroid and box centre fall above the mask. Stretching the outline 15%
upward fixed that. 30% started taking in the chair beside the bed in clip 2.

## What was tried

All rows use the corrected reference. "Knee rule" is
`PERCEIVE_SITTING_THIGH_RATIO=0.55`. "Vanish hold" is
`PERCEIVE_BED_VANISH_HOLD=true`.

| Zones | Zone test | Knee rule | Vanish hold | Bed acc. | Recall | Precision | False bed | Missed | False entries | False exits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| existing | centroid + box | - | - | 0.915 | 0.820 | 0.917 | 33 | 80 | 8 | 0 |
| existing | centroid + box | - | on | 0.930 | 0.884 | 0.906 | 41 | 52 | - | - |
| existing | centroid + box | on | on | 0.924 | 0.940 | 0.851 | 74 | 27 | 11 | 0 |
| segmented | centroid + box | - | on | 0.931 | 0.813 | 0.976 | 9 | 83 | 2 | 3 |
| segmented | centroid + box | on | on | 0.939 | 0.854 | 0.960 | 16 | 65 | 3 | 3 |
| segmented | hips | on | on | 0.942 | 0.933 | 0.898 | 47 | 30 | 6 | 0 |
| segmented +15% | centroid + box | - | on | 0.947 | 0.874 | 0.965 | 14 | 56 | 4 | 1 |
| **segmented +15%** | **centroid + box** | **on** | **on** | **0.961** | **0.930** | **0.952** | **21** | **31** | **5** | **0** |
| segmented +30% | centroid + box | on | on | 0.947 | 0.928 | 0.916 | 38 | 32 | 10 | 0 |
| segmented +15% | hips | on | on | 0.941 | 0.933 | 0.896 | 48 | 30 | 6 | 0 |

Every configuration caught all 9 bed exits, so the differences are in
false bed frames, missed bed frames and spurious entries.

What each change is worth:

- **Segmented zones** are the foundation. On their own they cut false bed
  frames from 41 to 9.
- **The knee rule** only helps on a good zone. On segmented +15% zones it
  gains 25 bed frames for 7 false ones. On the existing zones it gains 25
  for 33 false ones, because seated people in an oversized zone become bed
  occupants. That is why it ships off by default.
- **Vanish hold** is what recovers the person under the blanket (clip 4
  strict `in_bed` recall 0.04 to 0.71).
- **Testing bed membership at the hips** instead of the centroid-and-box-centre
  test raised recall but let in the chair and people standing in front of
  the bed. It was reverted.
- **Merging `in_bed`/`sitting_up` frames** into one confirmation count was
  also tried. It changed nothing once vanish hold was on, and was removed.

## The shipped tool, end to end

The table above used a prototype outline (OpenCV contour of the voted
mask). The shipped `perceive.calibrate_bed` was then run as a user would:
on every 10th frame of each clip, `yolo11m-seg.pt`, default settings. Its
zones were replayed with vanish hold and the knee rule on:

| Clip | Bed accuracy | Recall | Precision | False bed | Missed | Baseline accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.979 | 0.951 | 0.994 | 1 | 9 | 0.962 |
| 2 | **0.895** | 0.977 | **0.662** | **22** | 1 | 0.991 |
| 3 | 0.945 | 0.922 | 0.962 | 6 | 13 | 0.942 |
| 4 | 0.969 | 0.840 | 0.977 | 1 | 8 | 0.741 |
| **All** | **0.954** | **0.930** | **0.932** | **30** | **31** | 0.915 |
| All, original clip 1 reference | 0.927 | 0.861 | 0.932 | 30 | 67 | 0.889 |

Strict `in_bed` recall rises from 0.761 to 0.838, all 9 bed exits are still
caught, and there are no false exits.

**Clip 2 gets worse.** Its old hand-drawn rectangle was small and happened
to exclude the chair next to the bed. The segmented outline covers the whole
bed, and a person on that chair has their hips over it in the image. The
knee rule then correctly reads them as seated, so 22 chair frames count as
"in bed" (7 spurious entries). Clips 1, 3 and 4 match the prototype exactly;
clip 2's tool outline is slightly larger than the prototype's, which had 13
such frames. If a chair or bench sits directly beside or in front of the bed
in the camera's view, check the polygon in the dashboard Zones editor and
trim it there, or leave the knee rule off.

## What changed

- `perceive.calibrate_bed` (new): segments the bed across frames from Redis
  or a directory, votes the masks, simplifies the outline, stretches it 15%
  upward and writes only `bed` into `zones.yaml`. See CLAUDE.md Gotchas.
- `PERCEIVE_SITTING_THIGH_RATIO` (new, default `0`, off): an upright person
  whose visible thigh drops less than this fraction of their torso length is
  `sitting_up`, not `standing`. Seated frames measured median 0.11 (p90
  0.42), standing and walking p10 0.67.
- No default changed. The recommended bed setup is to run
  `calibrate_bed`, then set `PERCEIVE_BED_VANISH_HOLD=true` and
  `PERCEIVE_SITTING_THIGH_RATIO=0.55`, then restart `perceive`.

## Caveats

- **No hold-out.** The 15% stretch and the 0.55 thigh threshold were chosen
  on the same four clips they are scored on. The thigh threshold sits in a
  wide gap between seated and standing frames, so it is unlikely to be
  fragile. The stretch is a guess that happened to work in four rooms. Treat
  the headline as optimistic until a fifth clip confirms it.
- **Clip 1's corrected reference** needs a human look (see above).
- **The chair beside the bed** (clip 2) and **people standing close to the
  camera with their legs out of frame** remain the main false-bed sources.
  A second camera angle, depth, or the bed pressure sensor PLAN.md defers to
  v2 would resolve them. More 2D geometry will not.
- **Phantom filtering in the replay is approximate.** A phantom box was
  dropped rather than replaced by the next candidate, which `perceive` does
  live. It only affects clips 3 and 4's single low-confidence phantom.
- **Only the cached YOLO11s-pose letterbox-640 run was replayed.** Other
  backends were not re-measured.
