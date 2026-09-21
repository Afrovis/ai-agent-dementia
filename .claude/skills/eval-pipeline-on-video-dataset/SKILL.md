---
name: eval-pipeline-on-video-dataset
description: Evaluate the real-time perceive pipeline (fall / on_floor / posture detection) against an annotated third-party video dataset such as GMDCSA24, or any new set of clips. Use whenever the user asks to test, score or benchmark the pipeline on new videos or a new dataset. Zoning always comes first.
---

# Evaluate the pipeline on an annotated video dataset

**Rule: zone first, then evaluate.** Never report a number from an unzoned run
as the pipeline's accuracy. An unzoned run (GMDCSA24: 11% fall recall) is only
the baseline. Zoned: 32% recall. The pipeline assumes a calibrated fixed camera.

Raw videos stay where they are. Links, annotations, zones, predictions and
reports go under `../data-ai-agent-dementia/datasets/<name>/`.

## Steps

1. **Ingest.** Copy annotations and symlink videos into the data dir, and build
   a `clips.jsonl` index (`tools/gmdcsa24_eval/run_eval.py` is the template).
2. **Zone (mandatory, before any scoring).** Reference: `tools/gmdcsa24_eval/zone_scenes.py`.
   - Cluster clips into fixed-camera scenes (datasets often move the camera
     between clips; per-person zoning is wrong). Look at a first-frame contact
     sheet to sanity-check the clustering.
   - Per scene: `perceive.calibrate_bed` for the bed polygon (offline, on
     frames; writes only `bed` into a zones yaml), and a Theil-Sen ground line
     (`perceive.classify.fit_ground_line`) fitted from the scene's
     **non-fall** clips only, pinned as `PerceiveConfig.ground_line`. A tracker
     resets per clip and short clips never reach the 20 samples the online fit
     needs, so without a pinned line the height-ratio floor rule never fires.
   - With a calibrated bed: `bed_vanish_hold=True`, `sitting_thigh_ratio=0.55`
     (see CLAUDE.md). Phantoms need an empty room; skip and say so if none.
   - Never fit calibration on the clips being scored for falls.
3. **Evaluate.** Feed the real tracker + capture motion gate frame by frame at
   2 fps, 320x240 squash (same as the browser bridge), using each clip's zones
   and settings. `run_eval.py` refuses to run without `scenes.json` unless
   `--no-zones` (baseline).
4. **Score.** Fall recall and latency from annotated fall onset, and false
   `on_floor` alarms on non-fall clips, per person. Report per-subject numbers
   and list misses and false alarms. State caveats (e.g. floor-sitting labelled
   "Sitting" counts as a false alarm).

## Run (from the repo root)

```sh
export PERCEIVE_YOLO_MODEL=<abs path>/yolo11s-pose.pt
cd tools/gmdcsa24_eval
../../.venv-video-eval/bin/python zone_scenes.py --yolo-model $PERCEIVE_YOLO_MODEL
../../.venv-video-eval/bin/python run_eval.py --backend yolo --yolo-model $PERCEIVE_YOLO_MODEL
```
