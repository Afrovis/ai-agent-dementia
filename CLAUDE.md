# CLAUDE.md

Guidance for agents and contributors working in this repository.

Read [PLAN.md](PLAN.md) for the design and [HANDOFF.md](HANDOFF.md) for the
execution brief before picking up an issue. This file covers the operational
knowledge that is easy to lose: how to run the stack, how to test it without
hardware, and the TLS setup that the browser media bridge depends on.

## Architecture

Every service is a container that talks over Redis streams. Nothing calls
another service directly.

| Service | Role |
| --- | --- |
| `bus` | Redis streams broker. The only shared dependency. |
| `capture` | Motion-gates raw frames and publishes `Frame`. The only producer of `frames`. |
| `perceive` | Person detection and pose classification on frames. |
| `listen` | Voice activity detection and speech to text, publishes `Utterance`. |
| `agent` | Session state machine. Emits `Say`, `Show`, `Notify`, `GoalChanged`, `LightCommand`. |
| `light` | Feature-flagged local Shelly smart-plug control for the restroom path. |
| `embodiment` | Fullscreen HTTPS page: face, big text, photos, media bridge. |
| `notify` | Caregiver alerts. ntfy by default. |
| `store` | SQLite persistence and nightly summaries. |
| `dashboard` | Caregiver UI for live status, history, profile, strategies, media, and zones. |

Event schemas and the bus wrapper live in `shared/nc_shared`. `events.py`
holds the pydantic models and the two-way registry mapping each event class
to its stream name. Add new events there, not in a service.

`ARCHITECTURE.md` is the generated as-built map of services, streams, event
types, consumer groups, persistence, and outside-world connections. Do not
edit it by hand. After changing an event, stream, subscription, or declared
outside connection, regenerate and check it from the repository root:

```sh
python -m nc_shared.archdoc --write
docker run --rm -v "$PWD":/repo -w /repo python:3.12-slim \
  sh -c "pip install -q -e shared[dev] && pytest shared/tests"
```

## Running locally

```sh
cp .env.example .env
docker compose up --build
```

Ports come from `.env`: embodiment on `EMBODIMENT_PORT` (8443), dashboard on
`DASHBOARD_PORT` (8444), Redis published on 6379.

`embodiment` serves HTTPS when `CERT_FILE` and `CERT_KEY` both exist, and
falls back to plain HTTP otherwise. Plain HTTP is fine on `localhost`, which
browsers treat as a trustworthy origin, but the media bridge will not work
from any other device without a valid certificate.

## TLS for the media bridge

The page uses `getUserMedia`, which browsers only grant on a trustworthy
origin. A self-signed certificate is not enough: browsers restrict camera and
microphone access on pages with certificate errors, so the face renders but
the bridge stays dead.

Over Tailscale you can get a real Let's Encrypt certificate. Enable HTTPS
Certificates on the DNS page of the Tailscale admin console first, then:

```sh
tailscale status --json | python3 -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))"
tailscale cert --cert-file ts.crt --key-file ts.key <that-name>
```

On macOS the CLI is not on `PATH`. It lives at
`/Applications/Tailscale.app/Contents/MacOS/Tailscale`.

Two traps here.

The macOS Tailscale client is sandboxed. It ignores the paths you give it and
writes to `~/Library/Containers/io.tailscale.ipn.macos/Data`. Copy the files
out of there into `data/certs/lan.pem` and `data/certs/lan-key.pem`, then
`docker compose restart embodiment`.

The certificate covers the tailnet name only. Once installed, `localhost`,
the LAN address and the `.local` name all fail validation. Use the tailnet
name everywhere, including on the host itself, where MagicDNS resolves it
locally and validates cleanly.

Certificates last 90 days. Re-run `tailscale cert` to renew, and remember the
sandbox path again.

Verify a certificate is genuinely trusted with strict checks, never with
`curl -k` or a browser flag that ignores certificate errors:

```sh
curl -s -o /dev/null -w "%{http_code}\n" https://<name>:8443/
openssl s_client -connect <name>:8443 -servername <name> </dev/null 2>/dev/null | grep 'Verify return code'
```

## Testing without a camera, mic or Ollama

Every service must be testable with no hardware. Record live bus traffic to
JSONL and replay it. Run the tooling inside a container, where `nc_shared` is
already installed, rather than building a host virtualenv:

```sh
docker compose exec store python -m nc_shared.replay record redis://bus:6379 /app/data/rec.jsonl
docker compose exec store python -m nc_shared.replay play redis://bus:6379 /app/data/rec.jsonl --speed 10
```

`./data` is mounted into every container, so a file written to `/app/data`
appears in `data/` on the host.

The 50-scenario dialogue regression suite is separate from replay tooling. Its
unit tests need no Ollama; the actual comparison command calls local Ollama:

```sh
pip install -e services/agent -e tests/dialogue_bench[dev]
pytest tests/dialogue_bench/tests
python -m dialogue_bench --model llama3.1:8b --model qwen2.5:7b
```

Evaluating perception and the agent against recorded bedroom videos is
described in [docs/VIDEO_EVAL.md](docs/VIDEO_EVAL.md): a one-time tooling
build under `tools/video_eval/` and a per-video runbook. Installation,
every subcommand and container-parity commands are in
[tools/video_eval/README.md](tools/video_eval/README.md). Before rendering
review videos, or turning someone's timestamped annotation into markers and a
reference timeline, read its "Rendering review videos" section: every mode
needs an explicit `--pipeline-tag` on most clips, existing videos are skipped
without `--force`, and unmapped marker names silently render as `upright`.
Recordings,
frames and labels live outside the repository in `../data-ai-agent-dementia/`
and never enter git; only face-blurred, verified frames may be sent to Codex.

To check the media bridge, open the page, grant camera and microphone
permission, then watch these climb above zero. They sit at zero when no
browser is attached, which is correct rather than broken:

```sh
docker compose exec bus redis-cli XLEN frames_raw
docker compose exec bus redis-cli XLEN audio_in
```

`listen` drains `audio_in` continuously but only runs VAD/STT while a session
is `OBSERVING` or later. Its first complete utterance downloads `small.en` into
`data/models/faster-whisper`; later container recreations reuse those weights.
For hardware-free checks, the service tests inject both VAD and transcription:

```sh
docker compose build listen
docker compose run --rm --no-deps listen \
  sh -c "pip install -q pytest ruff && pytest -q && ruff check . && ruff format --check ."
```

`embodiment` turns `Say` events into local Piper WAV files and serves them back
to the bedside page under `/speech/<opaque-id>.wav`; speech bytes never enter
Redis. The image downloads `en_US-lessac-medium` when it is built. Startup
pre-renders the configured fixed strategy phrases into the ephemeral
`PIPER_CACHE_DIR`, including all twelve possible hour variants of the greeting;
generated speech is not retained under `data/`. To rebuild and exercise the
real voice path:

```sh
docker compose build embodiment
docker compose run --rm --no-deps embodiment \
  sh -c "pip install -q pytest ruff && pytest -q && ruff check . && ruff format --check ."
```

Barge-in does not wait for Whisper. At the first WebRTC VAD speech frame,
`listen` publishes `SpeechStarted` on `speech_in`; `embodiment` forwards it to
the page, which stops the current audio only when its `Say.interruptible` flag
is true. The page requests browser/OS echo cancellation from `getUserMedia`.
There is intentionally no software AEC in v1.

`frames_raw` is what the browser bridge writes. `capture` reads it, applies
the motion gate, and republishes onto `frames`, so watch that one to see
what `perceive` will actually receive:

```sh
docker compose exec bus redis-cli XLEN frames
```

Both streams are capped at 50 entries, so `XLEN` stops climbing there even
while frames keep flowing. A length that sits below the cap and never moves
is the real sign something is wrong.

`frames` fills more slowly than `frames_raw` by design. The browser sends
2 fps; the gate publishes all of it while the room moves, and drops to
0.5 fps once the room has been still for 30 seconds. Tune that with
`CAPTURE_FPS`, `CAPTURE_IDLE_FPS`, `CAPTURE_STATIC_SECONDS` and
`CAPTURE_MOTION_THRESHOLD` in `.env`.

To exercise notify, publish an event by hand. `Notify` requires a `source`
field, which is easy to miss:

```sh
docker compose exec store python -c "
import redis
from nc_shared.bus import Bus
from nc_shared.events import Notify
Bus(redis.Redis.from_url('redis://bus:6379')).publish(
    Notify(source='manual-test', level='attention', title='test',
           body='hello', repeat_until_ack=False))
"
```

## Gotchas

`dashboard` on port 8444 serves the caregiver pages, including the Zones
editor (issue #10): draw the bed, door, and bathroom-path zones on a live frame and save them to
`config/zones.yaml`. Every route needs HTTP Basic auth against
`DASHBOARD_PASSWORD`; with that unset (the default in a fresh `.env`) every
route answers 503 naming the variable rather than serving anything, camera
frame included. Set `DASHBOARD_PASSWORD` and restart the container to use
it. `perceive` only reads `zones.yaml` at startup, so a save here needs
`docker compose restart perceive` before it takes effect. Profile and strategy
saves atomically replace their YAML under `config/`; restart `agent` after
either change and `embodiment` after a strategy change. Photos and consented
family voice clips are validated and stored under `data/`, never Redis. The
System page remains a placeholder.

`capture` needs OpenCV only for `CAPTURE_SOURCE=usb` or `rtsp`. The browser
MVP source is the default and needs none of it, so the container does not
install it. Set the source and install the `camera` extra together, or the
service fails at startup with a message telling you exactly that.

`notify` falls back to a logging backend when `NTFY_URL` is empty, so alerts
appear in `docker compose logs notify` instead of on a phone. Set the variable
to a hard-to-guess ntfy topic to test real delivery.

Demo photos ship in `services/embodiment/embodiment/demo_photos` and resolve
at `/photos/demo_family` and `/photos/demo_room`. Caregiver uploads under
`PHOTO_DIR` shadow a demo photo of the same id.

`familiar_voice` is disabled by default. Upload a consented PCM WAV on the
dashboard Media page, copy its displayed id into that strategy's `clip_id`,
enable it, and restart `agent`; both `agent` and `embodiment` must use the same
`VOICE_CLIP_DIR` (the compose default is `/app/data/voice-clips`). It never
falls back to Piper or cloned speech when the configured clip is unavailable.

`data/` is gitignored. Certificates, the SQLite database and recordings all
live there and none of them belong in a commit.

`perceive`'s YOLO backend can lock onto a fixed non-person object (a lamp,
a headboard corner) that scores as "person" with an unmoving box, including
in an empty room. `PERCEIVE_PHANTOMS_FILE` points at a short, calibrated
list of such boxes to exclude below `PERCEIVE_PHANTOM_MAX_CONFIDENCE`
(default 0.7); unset (the default), the filter is off. Calibrate it with
the room empty:

```sh
docker compose exec perceive python -m perceive.calibrate_phantoms \
  --out /app/config/phantoms.yaml --redis redis://bus:6379 --count 60
```

or offline against prepared frames (`--frames-dir DIR` instead of
`--redis`/`--count`, used by `tools/video_eval`). Restart `perceive`
afterwards to pick up the file. Do this again, and re-run it, any time the
camera or the room's static furniture changes.

To debug what's actually in a single recorded frame (e.g. is a suspected
phantom box really a static object, or the bed with a person on it) without
sending anything off-machine, use `tools/video_eval/scripts/ask_frame.py`.
It asks a free-form question about one `frame_index` in one clip, optionally
with a detection box drawn on it first, through the local MLX VLM (not
Ollama -- same model family, ~6x faster per call, no request timeouts).
Install with `pip install 'tools/video_eval[mlx]'`. Everything stays local,
so it is safe to point at raw, unblurred `review_path` frames, unlike
`label-codex`.

The bed zone decides most in-bed versus out-of-bed readings, and a
hand-drawn rectangle is usually wrong in both directions: it takes in the
wall above the headboard and the floor in front of the bed, and misses the
foot of the mattress. `perceive.calibrate_bed` traces it from the image
instead with an ultralytics `-seg` model (COCO `bed`, `yolo11m-seg.pt` by
default, downloaded on first use), votes the masks across frames, stretches
the outline 15% upward so a lying or seated body still counts, and writes
only `bed` into `zones.yaml`. It runs once at setup, not per frame; re-run
it when the camera or the bed moves:

```sh
docker compose exec perceive python -m perceive.calibrate_bed \
  --zones /app/config/zones.yaml --redis redis://bus:6379 --count 40
```

or `--frames-dir DIR` offline. Restart `perceive` afterwards. With a
calibrated bed zone, set `PERCEIVE_BED_VANISH_HOLD=true` and
`PERCEIVE_SITTING_THIGH_RATIO=0.55`; the evidence is in
`docs/BED_OCCUPANCY_2026-09-15.md`.

## Conventions

Log one structured JSON line per event to stdout, with a `service` field.

Keep services independently testable. No service may require a camera, a
microphone, Ollama or a live Redis to run its tests.

This is not a medical device and not a substitute for supervision. Changes
that affect what the person sees or hears at night deserve more care than the
code alone suggests.
