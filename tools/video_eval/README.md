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

For package parity, build the optional tooling into the perceive image:

```sh
docker compose build --build-arg INSTALL_VIDEO_EVAL=true perceive
docker compose run --rm --no-deps \
  -v "$PWD/../data-ai-agent-dementia:/eval" \
  -e VIDEO_EVAL_DATA=/eval \
  -e VIDEO_EVAL_GIT_SHA="$(git rev-parse HEAD)" perceive \
  python -m video_eval predict --clip 2026-09-13_sample --backend mediapipe
```
