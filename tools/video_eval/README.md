# Recorded-video evaluation CLI

This package implements the private, offline pipeline in
[`docs/VIDEO_EVAL.md`](../../docs/VIDEO_EVAL.md). Raw video, extracted frames,
labels, and reports belong under `VIDEO_EVAL_DATA` (default
`../data-ai-agent-dementia`) and must never be added to this repository.

Install the tool with the real capture and perceive packages:

```sh
python -m venv .venv-video-eval
.venv-video-eval/bin/pip install \
  -e shared -e services/capture -e 'services/perceive[mediapipe,yolo]' \
  -e tests/perception_bench -e 'tools/video_eval[dev]'
```

Prepare the squash and letterbox variants, then run offline prediction:

```sh
.venv-video-eval/bin/python -m video_eval prepare \
  --video ../data-ai-agent-dementia/incoming/sample.mov --clip 2026-09-13_sample
.venv-video-eval/bin/python -m video_eval prepare \
  --video ../data-ai-agent-dementia/incoming/sample.mov --clip 2026-09-13_sample \
  --variant letterbox
.venv-video-eval/bin/python -m video_eval zones --clip 2026-09-13_sample
.venv-video-eval/bin/python -m video_eval predict \
  --clip 2026-09-13_sample --backend mediapipe
```

Build private labels after prediction. Run the local labeller before the final
blur pass so a YOLO miss can fail closed using its `person_visible` result:

```sh
.venv-video-eval/bin/python -m video_eval label-local --clip 2026-09-13_sample
.venv-video-eval/bin/python -m video_eval blur --clip 2026-09-13_sample
.venv-video-eval/bin/python -m video_eval sheets --clip 2026-09-13_sample
```

Open and inspect at least three files under the clip's `sheets/`
directory. Only after confirming that no face is visible, record that review
and allow the Codex labeller to run:

```sh
.venv-video-eval/bin/python -m video_eval sheets \
  --clip 2026-09-13_sample --confirm-reviewed
.venv-video-eval/bin/python -m video_eval label-codex --clip 2026-09-13_sample
```

Reconcile both labellers with the scenario card, inspect the draft and every
local path in the disagreement report, then explicitly confirm it. Scoring
refuses to use an unconfirmed draft:

```sh
.venv-video-eval/bin/python -m video_eval reconcile --clip 2026-09-13_sample
.venv-video-eval/bin/python -m video_eval reconcile \
  --clip 2026-09-13_sample --confirm --by "Recorder name"
.venv-video-eval/bin/python -m video_eval score --clip 2026-09-13_sample
```

After choosing one offline prediction tag, run the same bridge frames through
a freshly rebuilt live stack. This forces an always-active night window,
disables scene-note vision, records the retained output streams, removes raw
frame streams, and stops the stack even if replay fails:

```sh
.venv-video-eval/bin/python -m video_eval replay \
  --clip 2026-09-13_sample --tag mediapipe-squash-0123abcd-g
```

`label-local` defaults to adaptive sampling with `qwen3-vl:8b`; use `--fast`
for `gemma4:e4b-mlx` or `--all-frames` to disable still-frame propagation.
Invalid local JSON is retried once, then retained as a null label for human
reconciliation. `label-codex` can read only files listed beneath the clip's
`sheets/` directory and refuses to start without the human-review marker.

`zones` prints the path of a prepared bridge frame. Use that frame as the
reference for manually writing the clip's normalised `zones.yaml`; see
[`config/zones.example.yaml`](../../config/zones.example.yaml) for the format.

`prepare` requires `ffmpeg` and `ffprobe`. Commands skip matching completed
outputs unless `--force` is supplied. Every output records the current git SHA,
parameters, package versions, and elapsed time in a neighboring metadata file.
Event scoring accepts a predicted state already active at the labelled onset as
zero-delay when its run began at most 3 seconds early; the signed onset offset
remains in reports.

## Rendering review videos

`visualize` renders silent, synchronized review videos. Each video includes the
full camera view, the exact 320 by 240 letterboxed runtime input,
source-specific annotations, and a state timeline. Audio is never copied. There
is one video per clip per mode:

| Mode | Shows | Reads from `clips/<clip>/` | Fails when |
| --- | --- | --- | --- |
| `manual` | Recorder markers and the coarse state each implies | `clip.yaml` `script` | never; an empty script shows "not yet marked" throughout |
| `pipeline` | Gated real-time pose pipeline state, pose, zone | `predictions/<tag>.jsonl` and its `.meta.json` | the prediction tag is missing |
| `vision` | Local VLM posture, location and note | `labels/local.jsonl` | `local.jsonl` does not cover every frame in `frames.jsonl` |

Outputs go to `../data-ai-agent-dementia/analysis/<clip>__<mode>.mp4` with a
neighboring `.meta.json` recording the git SHA and pipeline tag. The pipeline
visualization is a standard output for every analyzed clip: render it
immediately after selecting the prediction tag. It does not require manual
annotations or local-VLM labels; those sources can be added and rendered later.

```sh
C=2026-09-14_bedroom-sample-04
TAG=mediapipe-squash-0c3f17b1-g   # any stem under clips/$C/predictions/
for mode in manual pipeline vision; do
  .venv-video-eval/bin/python -m video_eval visualize \
    --clip "$C" --mode "$mode" --pipeline-tag "$TAG" --force
done
```

Traps to check before rendering:

- **Every mode loads a pipeline prediction.** Without `--pipeline-tag` it looks
  only for `yolo_yolo11s-pose-letterbox-*-g.jsonl`, so `manual` and `vision`
  also fail on a clip that has only other prediction tags. List
  `predictions/` and pass the tag explicitly.
- **Existing videos are skipped silently.** Overlays are baked in at render
  time, and a rerun without `--force` returns `"skipped": true`. After changing
  `clip.yaml`, `local.jsonl` or predictions, re-render every affected mode with
  `--force`, including `pipeline`, which also prints the recorder marker.
- **The filename has no tag.** Rendering a second backend overwrites
  `<clip>__pipeline.mp4`. Keep comparisons apart with
  `--output-dir ../data-ai-agent-dementia/analysis/<tag>`.
- **Vision needs a complete `label-local` run first.** With `qwen3-vl:8b` it
  takes roughly half a minute per VLM call, and adaptive sampling still sends
  a large share of a two-minute clip. Run it in the background and render
  `vision` when it finishes. A run with different parameters starts again from
  frame 0 and replaces the partial file.

### Turning a human annotation into manual markers

People usually give markings as `MMSS action`, for example `0131 leave frame`.
Convert each to seconds (`0131` is 91.0) and write them to the clip's
`clip.yaml`, replacing `script: []`, in snake_case with the person's wording:

```yaml
script:
- t_s: 4.0
  action: leave_room
- t_s: 31.0
  action: sit_on_bed
```

The coarse state for each marker comes from `_manual_state()` in
`video_eval/visualize.py`. **Any action it does not list renders as
`upright`,** so a new name such as `lay_down_on_ground` silently shows the
wrong state. Check the mapping for every new action name; add missing names
there and in `tests/test_visualize.py`, then run the tests.

A marker's state lasts until the next marker. If the person leaves and comes
back without a marker, the video shows `absent` while they are on screen. Ask
for, or clearly flag, re-entry markers such as `enter_frame`.

`visualize` never reads `labels/reference.yaml`, and `score` never reads the
script. When a human annotation feeds both, write the markers to `clip.yaml`
and the interval timeline to `labels/reference.yaml`, and keep them in step.
Record any interval you inferred rather than took from the annotation in
`reference.yaml`'s `source` field.

Rendered videos, like the frames they come from, show the unblurred person.
They stay in `../data-ai-agent-dementia/`. Never commit, upload or share them,
and never pass their frames to a cloud model.

For package parity, build the optional tooling into the perceive image:

```sh
docker compose build --build-arg INSTALL_VIDEO_EVAL=true perceive
docker compose run --rm --no-deps \
  -v "$PWD/../data-ai-agent-dementia:/eval" \
  -e VIDEO_EVAL_DATA=/eval \
  -e VIDEO_EVAL_GIT_SHA="$(git rev-parse HEAD)" perceive \
  python -m video_eval predict --clip 2026-09-13_sample --backend mediapipe
```
