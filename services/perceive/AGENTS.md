# perceive

Person detection, pose classification and the gaze point for the eyes.

`perceive` reads `config/zones.yaml` and the phantom file only at startup:
restart it after the dashboard Zones editor or either calibration below writes
them. The exception is the embodiment debug "Detect bed zone" button, which
reloads it live.

`Gaze` (face point, bed centroid, or none) goes on the capped `gaze` stream,
which `store` does not persist.

## Phantom boxes

The YOLO backend can lock onto a fixed non-person object (a lamp, a headboard
corner) that scores as "person" with an unmoving box, even in an empty room.
`PERCEIVE_PHANTOMS_FILE` points at a short calibrated list of such boxes to
exclude below `PERCEIVE_PHANTOM_MAX_CONFIDENCE` (default 0.7); unset, the filter
is off. Calibrate with the room empty, and again whenever the camera or the
static furniture changes:

```sh
docker compose exec perceive python -m perceive.calibrate_phantoms \
  --out /app/config/phantoms.yaml --redis redis://bus:6379 --count 60
```

Offline, use `--frames-dir DIR` instead of `--redis`/`--count` (as
`tools/video_eval` does).

## Bed zone

The bed zone decides most in-bed versus out-of-bed readings, and a hand-drawn
rectangle is usually wrong both ways: it takes in the wall above the headboard
and the floor in front of the bed, and misses the foot of the mattress.
`perceive.calibrate_bed` traces it from the image with an ultralytics `-seg`
model (COCO `bed`, `yolo11m-seg.pt` by default, downloaded on first use), votes
the masks across frames, stretches the outline 15% upward so a lying or seated
body still counts, and writes only `bed` into `zones.yaml`. Run it once at
setup, and again when the camera or the bed moves:

```sh
docker compose exec perceive python -m perceive.calibrate_bed \
  --zones /app/config/zones.yaml --redis redis://bus:6379 --count 40
```

or `--frames-dir DIR` offline. With a calibrated bed zone, set
`PERCEIVE_BED_VANISH_HOLD=true` and `PERCEIVE_SITTING_THIGH_RATIO=0.55`; the
evidence is in `docs/BED_OCCUPANCY_2026-09-15.md`.
