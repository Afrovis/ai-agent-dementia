# Perception bench (issue #11)

Measures `perceive`'s pose pipeline (`services/perceive/perceive/{backends,
classify,zones}.py`) against three tiers of fixtures, from weakest to
strongest evidence. See PLAN.md section 12 and HANDOFF.md section 8 for why
this exists in three tiers rather than one: no infrared recordings of
consenting volunteers exist yet.

| Tier | Source | Always available? | Measures |
|---|---|---|---|
| 1 | Synthetic scripted clips, generated in-process | Yes, no downloads | Per-state accuracy, confusion matrix, **latency** |
| 2 | IndoorActionDataset (real daylight RGB, opt-in download) | No, needs `fetch_daylight.sh` | Per-state accuracy for 4/6 states, no latency, no `in_bed` |
| 3 | Infrared clips from the actual room, via `tools/video_eval` | No -- needs `tools/video_eval` installed and a confirmed manifest | Everything: per-state recall, latency, gate verdicts, pooled by tag family |

**No fabricated numbers, anywhere in this bench's output or docs.** If a
tier has no data, the bench says so and exits 0, not silently. Example
output blocks in this file are explicitly labelled illustrative.

## Running it

```sh
python3.12 -m venv .venv-bench
source .venv-bench/bin/activate
pip install -e services/perceive        # perceive's own deps (pillow, numpy, pyyaml)
pip install -e tests/perception_bench    # this bench's deps, same set
python -m perception_bench
```

`perception_bench` is not a `perceive` dependency in the packaging sense --
it adds `services/perceive` to `sys.path` at import time (see
`perception_bench/__init__.py`) instead, so tier 1 and this bench's own
unit tests need nothing beyond Pillow/numpy/PyYAML, per this issue's hard
constraint that tier 1 and every unit test run with no downloads, no
camera, no model weights, no Redis, no network, and with neither MediaPipe
nor YOLOv8 installed.

Flags:

- `--json` -- machine-readable report instead of the stdout one.
- `--daylight-data-dir PATH` -- where `fetch_daylight.sh` put the dataset
  (default `data/perception_bench/daylight`).
- `--daylight-pose-backend {mediapipe,yolo}` -- which real backend scores
  tier 2 (needs the matching extra installed: `pip install -e
  tests/perception_bench[mediapipe]` or `[yolo]`).
- `--night-degrade` -- run tier 2 through
  `perception_bench.degrade.apply_night_degradation` first (see that
  module's docstring: de-risking, not IR evidence).
- `--ir-manifest PATH` -- tier 3 manifest (default
  `tests/perception_bench/fixtures/ir_manifest.yaml`). See
  `perception_bench/infrared.py`'s module docstring for its format --
  `video_eval reconcile --confirm` keeps it in sync automatically.
- `--ir-tag PATTERN` -- `fnmatch` glob restricting which tier 3 prediction
  tags get scored and reported (default: every tag with an existing
  `predictions/*.jsonl` file).
- `--ir-backend NAME` / `--ir-variant NAME` -- run `video_eval predict`
  with this backend (and bridge variant, default `squash`) for every
  confirmed tier 3 clip before scoring it, instead of only scoring
  predictions that already exist.
- `--ir-rescore` -- force tier 3 to re-run predict/score instead of
  reusing existing up-to-date reports.
- `--skip-daylight` -- skip even checking for tier 2 data.

Exit code is non-zero only if a *measured, gating* target was missed. A
tier or state with no data is reported "not measured" and never makes
the exit code non-zero by itself.

Tier 1 verdicts are **advisory**: they run against synthetic,
generated-in-process clips as a smoke test, not the real recorded
footage the >95%/<2s targets in PLAN.md section 12 are defined against,
so tier 1 is printed with its real measured numbers -- including a real
FAIL when it misses -- but never gates the exit code. Tier 3 is advisory
for the same reason, only more so: a handful of confirmed infrared clips
is real evidence but not yet the statistical evidence those targets
assume, so a real FAIL is printed per tag family but never gates the exit
code either. Only tier 2 gates: a *measured* miss there exits non-zero,
"not measured" still exits 0. The final "Result:" line says which of the
three happened.

## Getting tier 2 data

```sh
tests/perception_bench/fetch_daylight.sh
```

Downloads ~977 MB into `data/perception_bench/daylight/` (gitignored).
Without this, `python -m perception_bench` reports tier 2 as skipped and
exits 0.

## Getting tier 3 data

There is no shortcut for this one: someone has to record the actual room.
See `RECORDING.md` for the protocol, and `perception_bench/infrared.py`
for the manifest format tier 3 reads.

## Illustrative example output (placeholder values, not a real run)

```
Tier 1: synthetic scripted clips (advisory / smoke-test, does not gate exit code)
------------------------------------------------------------
overall frame accuracy: 91.3%          # <- ILLUSTRATIVE, not measured
  [PASS] tier1 standing recall >= 95% (advisory, smoke-test, does not gate exit code): 100.0%
  [PASS] tier1 on_floor recall >= 95% (advisory, smoke-test, does not gate exit code): 100.0%
  [PASS] tier1 latency <= 2s (advisory, smoke-test, does not gate exit code): max 1.00s, mean 0.40s
```

For real numbers, run the bench yourself; see the verification output in
this issue's PR description for what an actual run against this codebase
produced on the day it was written.

## What this bench cannot tell you yet

- Whether the pipeline works on infrared at all (tier 3 is empty).
- Whether it works on `in_bed` at all outside tier 1's synthetic geometry
  (tier 2 has no bed; tier 3 is empty).
- Whether MediaPipe or YOLOv8-pose (HANDOFF.md section 12's open question)
  is the better choice -- that comparison needs tier 2 or tier 3 data
  actually run through both backends with `--daylight-pose-backend`, which
  nobody has done yet in this environment (no network, no model weights
  available here).

Milestone 1 (HANDOFF.md section 8) is not done until tier 3 is not empty.

## Open question for tier 3: does the debounce make <2s latency reachable?

On the synthetic tier 1 clips, the measured latency is `max 2.50s, mean
0.92s` against `perceive`'s default `PERCEIVE_CONFIRM_FRAMES=3`
debounce -- the tracker waits for three consecutive frames of a new pose
before confirming a state change, trading latency for stability against
one-off misclassifications. On those synthetic clips that debounce is
enough to push the worst-case transition past the <2s target.

This is an observation from synthetic clips, not a proven fact about the
real system: tier 1's frame timing and pose noise are not the real
room's. Whether the default debounce actually makes the <2s target
unreachable in practice -- and whether `PERCEIVE_CONFIRM_FRAMES` should
be lowered -- is an open question tier 3's real infrared measurement
will settle, not this one.
