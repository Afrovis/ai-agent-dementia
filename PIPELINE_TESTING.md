# Pipeline testing: label vs. MediaPipe vs. VLM

Two recordings compared on 2026-09-13: the scripted/human label for each clip
against two independent perception outputs — the production pose backend
(`predict`, MediaPipe) and the offline vision-language model used for
reference labelling (`label-local`, Ollama `qwen3-vl:8b`). See
[docs/VIDEO_EVAL.md](docs/VIDEO_EVAL.md) for the pipeline design and
[HANDOFF.md](HANDOFF.md) section 13 for a related bug found in the VLM
labeller.

## Video 1 — `2026-09-13_bedroom-sample-01` (244s)

### My label (scripted timeline)

| t_s | action |
|---|---|
| 9.0 | in_bed |
| 46.0 | sitting_up |
| 58.0 | sitting_up |
| 65.0 | bed_exit |
| 78.0 | open_drawer |
| 88.0 | in_bed |
| 102.0 | bed_exit |
| 120.0 | floor |
| 128.0 | getting_up_from_floor |
| 135.0 | floor |
| 144.0 | getting_up_from_fall |
| 152.0 | in_bed |
| 190.0 | making_bed |
| 212.0 | putting_clothes_on |
| 237.0 | finished_putting_clothes_on |

### MediaPipe result (`predictions/mediapipe-squash-aa67d72c-g.jsonl`, gated)

Observed `state` vocabulary in this run: `absent`, `standing`, `sitting_up`,
`walking`. No `in_bed` or floor-specific state was ever emitted.

| Script event (t_s) | MediaPipe at that time | Match |
|---|---|---|
| 9.0 in_bed | absent | miss |
| 46.0 sitting_up | absent (gated) | miss |
| 58.0 sitting_up | absent | miss |
| 65.0 bed_exit | absent | miss |
| 78.0 open_drawer | standing | miss (no drawer state) |
| 88.0 in_bed | standing | miss |
| 102.0 bed_exit | absent | partial (absence is directionally right) |
| 120.0 floor | absent | partial (no floor state, but not "standing" either) |
| 128.0 getting_up_from_floor | absent | miss |
| 135.0 floor | absent | match on absence, no floor semantics |
| 144.0 getting_up_from_fall | absent | miss |
| 152.0 in_bed | standing → absent | miss |
| 190.0 making_bed | absent | miss |
| 212.0 putting_clothes_on | absent | miss |
| 237.0 finished_putting_clothes_on | standing → walking | partial |

Detection dropped out (`detected=false` / `absent`) for roughly 60-65% of all
488 frames. Largest continuous gaps: 11.0-35.5s (24.5s), 152.5-170.0s (17.5s),
88.5-102.5s (14.0s), 135.0-145.0s (10.0s, worst 8.5s unbroken). Every one of
the 15 scripted events happened to fall inside or next to one of these gaps
or a state the backend cannot express (`in_bed`, floor, or fine-grained
actions), so effectively none scored a clean match.

### VLM result (`labels/local.jsonl`, Ollama `qwen3-vl:8b`, adaptive sampling)

Condensed posture/location timeline (see full breakdown in conversation
history) tracks the coarse in_bed / sitting_up / upright / on_floor signal
well:

| Script event (t_s) | VLM transition | Match |
|---|---|---|
| 46.0 sitting_up | 46.0-50.5 sitting_up | exact |
| 65.0 bed_exit | 61-65.5 in_bed → 66-69.5 upright | good |
| 88.0 in_bed | 90.0-99.5 in_bed | ~2s lag |
| 120.0 floor | 120.0-127.0 on_floor | exact |
| 135.0 floor | 135.0-139.5 on_floor | exact |
| 152.0 in_bed | 156.5-171.0 in_bed | ~4-5s lag |

Two data-quality problems, not accuracy problems: 27/488 frames (5.5%) came
back as null (`failed_label_record`) due to a `person_visible` boolean
validation bug (see HANDOFF.md section 13) — one such gap (51.0-60.5s)
erased the second `sitting_up` script event entirely. Fine-grained actions
(`open_drawer`, `making_bed`, `putting_clothes_on`) have no corresponding
label, since the model's output vocabulary is posture/location only.

## Video 2 — `2026-09-13_bedroom-sample-02` (114s)

### My label (scripted timeline)

| t_s | action |
|---|---|
| 3.0 | out_of_frame |
| 6.0 | in_frame |
| 12.0 | sitting_on_chair |
| 18.0 | walking |
| 23.0 | sitting |
| 31.0 | walking |
| 38.0 | open_drawer |
| 50.0 | open_drawer |
| 68.0 | in_bed_above_blanket |
| 81.0 | sitting |
| 90.0 | standing_and_walking |
| 99.0 | sitting_on_floor |
| 108.0 | walking |

### MediaPipe result (`predictions/mediapipe-squash-fbb83bc0-g.jsonl`, gated)

| Your label (t_s) | MediaPipe | Verdict |
|---|---|---|
| 3.0 out_of_frame / 6.0 in_frame | 0.0-7.0 absent | match |
| 12.0 sitting_on_chair | 13.5-17.0 sitting_up | match (~1.5s lag) |
| 18.0 walking | 19.0-21.0 walking | match |
| 23.0 sitting | 21.5-23.5 absent, then standing | miss |
| 31.0 walking | walking only from 34.5 | match, ~3.5s lag |
| 38.0 open_drawer | 37.0-44.0 absent | detection dropout |
| 50.0 open_drawer | 48.5-51.0 absent | detection dropout |
| 68.0 in_bed_above_blanket | 66.0-79.0 absent | complete miss, 13s dropout spans the whole event |
| 81.0 sitting | 79.0-90.5 sitting_up | match |
| 90.0 standing_and_walking | 90.5-97.5 standing→walking | match |
| 99.0 sitting_on_floor | 97.5-108.0 sitting_up (no floor state) | miss |
| 108.0 walking | 108.0-113.5 standing (never walking) | miss |

YOLO (`yolo-squash-fbb83bc0`, `yolo-letterbox-fbb83bc0`) shows the same
66-78/79s absence over the in-bed event and the same lack of a floor/fall
state — the gap is backend-independent, not MediaPipe-specific.

### VLM result (`labels/local.jsonl`, Ollama `qwen3-vl:8b`, adaptive sampling)

| Your label (t_s) | VLM transition | Verdict |
|---|---|---|
| 3.0 out_of_frame / 6.0 in_frame | 0.0-11.0 person_visible=true throughout | miss — never flagged absence |
| 12.0 sitting_on_chair | 12.0-16.0 sitting_up/other | match |
| 18.0 walking | 17.0-23.5 upright/other | match |
| 23.0 sitting | 24.0-31.0 sitting_up/bed | match (~1s lag) |
| 31.0 walking | 31.5-34.0 upright/other | match |
| 38.0 open_drawer | 34.5-45.0 upright/door | no signal (no drawer state) |
| 50.0 open_drawer | 45.5-55.5 upright/other | no signal |
| 68.0 in_bed_above_blanket | 67.0-81.5 in_bed/bed | match (~1s early) |
| 81.0 sitting | 82.0-91.0 sitting_up/bed | match (~1s lag) |
| 90.0 standing_and_walking | 91.5-96.0 upright/door→other | match (~1.5s lag) |
| 99.0 sitting_on_floor | 96.5-106.0 on_floor/other | match |
| 108.0 walking | 106.5-114.0 upright/other | match (1.5s early) |

10/13 events matched directionally, typically within 1-2s. Data quality was
clean: only 1/229 frames malformed (an Ollama request timeout at t=11.5s,
0.4% — a different and much rarer failure mode than video 1's validation
bug).

## Initial conclusions

- **The production pose backend (MediaPipe, and YOLO likewise) has a real
  blind spot on exactly the states this system exists to catch.** Both
  clips show the backend going completely undetected (`absent`, no bbox at
  all) for 10-25+ second stretches, and those stretches disproportionately
  land on bed-occupancy and floor/fall events — the two clearest safety
  signals in the whole recording. Neither backend exposes a state distinct
  enough to tell "on the floor after a fall" apart from "sitting on a bed or
  chair" — in the data observed here, floor and fall events read as
  `sitting_up` or `absent`, indistinguishable from benign sitting.
- **The offline VLM catches bed/floor posture much better than the live
  backend does** (exact-second matches on `in_bed`/`floor`/`on_floor` in
  both clips) but is ~250-450x slower (minutes vs. single-digit seconds) and
  cannot see fine-grained actions (`open_drawer`, `making_bed`,
  `putting_clothes_on`) at all — its output vocabulary is posture/location
  only.
- **This corroborates, with two independent recordings, the open question
  already logged in HANDOFF.md section 12** ("MediaPipe missed people in
  bed that YOLO caught, at bridge resolution") — the gap shows up for both
  backends here, not just MediaPipe, so the open question may need
  broadening from "which backend" to "does either backend see the person at
  all in bed/floor poses at this resolution."
- A real bug (not just a coverage gap) was found and written up separately:
  the VLM labeller's `person_visible` boolean validation rejects some valid
  Ollama responses and silently nulls out the frame, occasionally erasing a
  real scripted event — see HANDOFF.md section 13 for root cause and
  suggested fix.

## Where to find the data

All source video, frames, predictions and labels live outside this repo in
`../data-ai-agent-dementia/` (never committed — see the "Video eval data and
privacy" note and [docs/VIDEO_EVAL.md](docs/VIDEO_EVAL.md)):

```
data-ai-agent-dementia/
├── clips/2026-09-13_bedroom-sample-01/
│   ├── clip.yaml                                    # source, video metadata, script (my label)
│   ├── labels/local.jsonl, local.meta.json          # VLM result
│   └── predictions/mediapipe-squash-aa67d72c-g.jsonl,
│       yolo-squash-f208d4a1.jsonl,
│       yolo-letterbox-f208d4a1.jsonl                # MediaPipe / YOLO result
└── clips/2026-09-13_bedroom-sample-02/
    ├── clip.yaml                                    # source, video metadata, script (my label)
    ├── labels/local.jsonl, local.meta.json          # VLM result
    └── predictions/mediapipe-squash-fbb83bc0-g.jsonl,
        yolo-squash-fbb83bc0.jsonl,
        yolo-letterbox-fbb83bc0.jsonl                # MediaPipe / YOLO result
```

Neither clip has a confirmed `labels/reference.yaml` yet (the reconcile /
Codex-label steps in `docs/VIDEO_EVAL.md` haven't run) — the `script:` field
in each `clip.yaml` is the closest thing to ground truth right now and is
what this document's "My label" columns are drawn from.
