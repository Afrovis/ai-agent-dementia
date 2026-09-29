# Night Companion

Night Companion is a local AI agent that keeps company with a person living
with dementia when they wake up at night. When someone with dementia wakes
in the dark, they can become disoriented — unsure of the time, the place, or
why they're awake — which is when wandering and falls happen. Night
Companion notices the wake-up with a camera, greets the person with calm
glowing eyes and a soft voice on a bedside screen, reminds them where and
when they are, and gently guides them back to bed. If a real need comes up
(the bathroom, water, pain) it adapts instead of just repeating itself, and
if nudging doesn't work — or something looks genuinely wrong, like a fall —
it wakes up the family caregiver instead of the person.

> [!IMPORTANT]
> Night Companion is **not a medical device** and **not a substitute for
> supervision**. It is research software under active development, has not
> been clinically validated, and can miss a fall or misread a situation. It
> is an assistive layer for nights when no one else is awake in the room,
> never a replacement for a person who is.

## Who it's for

- **The person with dementia** — one bedroom, one person, active only during
  a configurable night window (default 22:00–07:00).
- **The family caregiver** — writes the phrases the agent uses, picks
  family photos, sets which strategies run in what order, and is the one
  who gets alerted. They stay in control; the agent never improvises safety
  decisions on its own.

## How it works, in plain terms

1. A camera watches the room. Motion wakes the system up.
2. The agent checks whether the person is in bed, sitting up, standing,
   walking, on the floor, or absent.
3. If they've left the bed, a bedside screen shows calm eyes that follow the
   person, large text, and a soft voice — reorienting them ("It's 3am.
   You're at home in your bedroom. Let's go back to bed.") and, if they
   respond, listening and adapting instead of repeating a script.
4. A short list of strategies runs in order: a quiet clock, a gentle
   greeting, orienting text with a familiar photo, listening and
   responding, step-by-step guidance back to bed, a recorded message from a
   family member, lighting the path to the bathroom, and — if nothing works
   or something looks wrong — a push notification to the caregiver's phone
   that repeats until acknowledged.
5. Every session is logged for a caregiver dashboard and a morning summary,
   so the family can see what happened without watching video.

Safety-relevant decisions — state transitions, escalation timers, when to
call the caregiver — live in deterministic code, never in the language
model. The LLM composes language and interprets intent; it doesn't decide
whether something is an emergency.

## Privacy

Everything runs on hardware in the home. Camera frames and audio never
leave the box and are not written to disk unless the caregiver turns on a
documented debug option; only structured events (state changes, what was
said) are stored. An optional cloud fallback for hard reasoning cases is
off by default, and when enabled it only ever sees text, never images or
audio.

The repository contains no recordings. Test fixtures are event logs and
synthetic scenarios; recorded footage used for evaluation lives outside the
repository and is never committed.

## Current status (v0.1, in progress)

**Working today:**
- The full session pipeline: wake detection → screen/voice reorientation →
  strategy escalation → caregiver alert.
- 8 of 10 caregiver strategies (quiet clock, greeting, orient with photo,
  listen-and-respond, guided return, family voice message, path light,
  phone escalation). Family voice plays only a consented recording a
  caregiver uploaded, never a cloned voice, and is off by default.
- Voice in and out (faster-whisper speech-to-text, Piper text-to-speech)
  with barge-in — the agent stops talking the moment the person starts
  speaking.
- The caregiver dashboard: live status, session history, profile and
  strategy editors, photo and voice-clip uploads, a zone-drawing tool,
  morning summaries.
- Test benches that need no person in the room: a 50-scenario dialogue
  suite, a decision benchmark against human-reviewed labels, replayed live
  sessions, simulated nights (`scene_lab`), and an offline video pipeline
  that scores perception against hand-checked reference recordings.

**Not built yet:**
- Music or story playback (strategy 7).
- A softer in-home chime escalation step (strategy 9), which needs extra
  hardware.
- Multiple rooms, multiple people, or languages other than English.
- Bed or door pressure sensors — perception is camera-only for now.
- The two-week dry run with a volunteer
  ([#27](https://github.com/Afrovis/ai-agent-dementia/issues/27)); nothing
  here has been used with a person living with dementia yet.

## Results so far

Perception has been benchmarked offline against recorded bedroom video
(RGB, lamp-lit, hand-labelled reference timelines) rather than trusted on
sight:

- **Per-frame state agreement** (in bed / sitting / standing / walking /
  on floor / absent) with the best model and rule set: **0.78–0.90**
  across the first two test clips, up from 0.56–0.65 with rules off.
- **Scripted events matched:** 9/9 and 11/11, with zero false events.
- **Fall / floor detection:** a "height ratio" rule — comparing a detected
  person's box height against what a calibrated standing person should
  measure at that camera position — catches both scripted floor events
  with **zero false floor alarms**, versus 0% recall from the original
  geometry-only rule.
- **Latency:** perception runs in 12–60 ms per frame depending on model,
  well inside the ~500 ms budget at the 2 fps the camera streams.

The honest caveat: this is a handful of clips, not a validated accuracy
rate. It shows the pipeline and rules move in the right direction; many
more recorded nights are needed before treating these numbers as reliable.
Details are in
[docs/PERCEIVE_ACCURACY_2026-09-13.md](docs/PERCEIVE_ACCURACY_2026-09-13.md),
[docs/FLOOR_DETECTION_HANDOFF.md](docs/FLOOR_DETECTION_HANDOFF.md), and the
dated log in [EXPERIMENTS.md](EXPERIMENTS.md).

## Getting started

### Requirements

- Docker with Compose. The target is an Apple-silicon Mac mini or MacBook
  with 16 GB of memory; other hosts that run Docker should work but are
  less tested.
- [Ollama](https://ollama.com) on the host, with the two local models:

  ```sh
  ollama pull gemma4:e4b-mlx   # the agent's text model and the floor check
  ollama pull moondream        # scene notes
  ```

- A webcam and microphone in a browser on the bedside device. No camera is
  needed to run the tests.

### Run the stack

```sh
cp .env.example .env
# At minimum set TZ to your time zone and DASHBOARD_PASSWORD to something
# private. Every setting is documented inline in .env.example.
docker compose up --build
```

- Bedside page: `http://localhost:8443` (`EMBODIMENT_PORT`)
- Caregiver dashboard: `http://localhost:8444` (`DASHBOARD_PORT`), HTTP
  Basic auth with `DASHBOARD_PASSWORD`. It answers 503 until that is set.

Plain HTTP is fine on `localhost`. To use a tablet or another device as the
bedside screen, the camera and microphone need a trusted certificate; see
[docs/TLS.md](docs/TLS.md).

Then personalise it for the person, all from the dashboard or by copying the
templates in [`config/`](config):

- **Profile** (`person.example.yaml`): name, the caregiver's name, recurring
  night-time beliefs, calming things, things to avoid.
- **Strategies** (`strategies.example.yaml`): which run, in what order, and
  the exact phrases.
- **Zones**: draw the bed, door and bathroom-path zones on a live frame.

Caregiver alerts go to [ntfy](https://ntfy.sh) by default. Set `NTFY_URL` to
a hard-to-guess topic and subscribe to it on the caregiver's phone; with it
empty, alerts appear in `docker compose logs notify` instead. The hallway
light (`LIGHT_ENABLED`) drives a local Shelly smart plug and is off by
default.

### Tests

Every service is testable with no camera, microphone, Ollama or live Redis.
Each service's tests run in its own image, for example:

```sh
docker compose build agent
docker compose run --rm --no-deps agent \
  sh -c "pip install -q pytest ruff && pytest -q && ruff check . && ruff format --check ."
```

Live bus traffic can be recorded and replayed to exercise the pipeline
without hardware. [AGENTS.md](AGENTS.md) covers that, the benches under
`tests/` and `tools/`, and the operational traps.

## Architecture

Every service is a container that talks to every other one over Redis
streams — nothing calls another service directly, so any one part can be
swapped, replayed, or tested without the others running.

| Service | Role |
| --- | --- |
| `bus` | Redis streams broker. The only shared dependency. |
| `capture` | Motion-gates raw frames and publishes them for perception. |
| `perceive` | Detects the person and classifies their pose/state. |
| `listen` | Voice activity detection and speech-to-text. |
| `agent` | The session state machine: goals, strategies, LLM calls. |
| `light` | Feature-flagged smart-plug control for the hallway light. |
| `embodiment` | The bedside screen: eyes, text, voice, camera/mic bridge. |
| `notify` | Caregiver alerts (ntfy by default). |
| `store` | SQLite persistence and nightly summaries. |
| `dashboard` | Caregiver web UI. |

`volunteer/` is a separate recording site that collects consented clips of
volunteers acting out night-time scenarios for the perception benchmarks.
Clips are encrypted in the browser and stored outside the repository. It is
not part of the bedside stack.

### Technology stack

- **Runtime:** Docker Compose, with Ollama running on the host.
- **Language:** Python 3.12 across every service.
- **Perception:** YOLO11 pose (Ultralytics) or MediaPipe, plus a local
  Ollama vision model for harder cases.
- **Voice:** faster-whisper for speech-to-text, Piper for text-to-speech.
- **LLM:** a local Ollama model by default (`gemma4:e4b-mlx`), or an MLX
  model served on the host; an optional text-only Claude fallback for hard
  reasoning, off by default.
- **Storage:** SQLite via SQLModel.
- **Web:** FastAPI for services, HTMX for the dashboard, plain HTML/JS for
  the bedside page.
- **Messaging:** Redis streams, at-least-once delivery, Pydantic-typed
  events.

## Where to read more

- [PLAN.md](PLAN.md) — full design, guiding principles and the ethics of
  putting a camera in someone's bedroom.
- [HANDOFF.md](HANDOFF.md) — fixed decisions, event contracts, safety rules
  and the definition of done.
- [ARCHITECTURE.md](ARCHITECTURE.md) — generated, as-built map of
  services, streams, and events.
- [AGENTS.md](AGENTS.md) — how to run, test and debug the stack; shared by
  human contributors and coding agents (`CLAUDE.md` imports it).
- [EXPERIMENTS.md](EXPERIMENTS.md) — dated log of measured results.
- [docs/](docs) — accuracy investigations, evaluation runbooks and plans.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
Changes that affect what the person sees or hears at night deserve more care
than the code alone suggests. To report a security or privacy problem, see
[SECURITY.md](SECURITY.md) rather than opening a public issue.

## License

[GNU Affero General Public License v3.0](LICENSE). Night Companion builds on
Ultralytics YOLO (AGPL-3.0) and Piper (GPL-3.0); the model weights it
downloads carry their own licenses.
