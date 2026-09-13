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
  -e 'tools/video_eval[dev]'
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
