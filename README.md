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
>
> It is also a **prototype that is not HIPAA compliant**. Do not use it with
> real patient data or protected health information.

## See it work

Three short silent videos. Every label, quoted line and number in them comes
from a real run of the code; where the person is acting, scripted or
simulated, the video says so.

### What the camera sees

PERCEPTION_VIDEO_URL

35 s, silent. Also in the repository as an
[animated image](docs/media/perception-reel.webp) and an
[MP4](docs/media/perception-reel.mp4).

A recorded bedroom test, lamp-lit, with the project owner acting out three
moments of a night: crossing the room, getting into bed, and sitting down on
the floor and getting up again.

- **The skeleton** is drawn by a large pose model (YOLO11x-pose) on the full
  4K recording, only so the body is easy to follow in the video.
- **The label at the bottom left** (*Walking*, *Standing*, *Sitting up*,
  *In bed*, *On the floor*), with its confidence and zone, is what the real
  perception pipeline concluded from the 2 frames per second the bedside
  camera actually sends. It is shown as it happened, so it trails the
  movement by a beat; it still says *On the floor* for a moment after the
  person stands up.
- **The bed outline** fills in when the pipeline places the person in the
  calibrated bed zone.
- **The two lanes at the bottom right** compare the pipeline (SEEN) with a
  hand-labelled reference timeline (TRUTH). Agreement between them is what
  the perception benchmark scores; see [Results so far](#results-so-far).
- **The "On the floor" chip** counts how long the reading has held. The
  agent's floor rule works from that state, as the next video shows.

Only that structured reading, such as `on_floor, 0.93, zone room`, leaves
perception as a `PersonState` event. In normal use the frames themselves stay
on the device and are not stored.

### One night moment, as events

https://github.com/user-attachments/assets/ae1f7ab7-4799-48c4-8d9b-53c7c4f6002e

62 s, silent. Also in the repository as an
[animated image](docs/media/events-explainer.webp) and an [MP4](docs/media/events-explainer.mp4).

A person gets up at night, asks for her late husband, asks to go home and
ends up on the floor. Her words and movements are scripted; everything the
system does is a real run of the agent code with its local model:

- **"Where is Henri?"** The model reads the intent (looking for someone,
  mild distress). The agent, not the model, picks the answer: a fixed,
  pre-approved line, *"You're worried about them, Jean; that's a caring
  thing to feel."*
- **"I want to go home now."** The model drafts a reply, but the draft
  contains "but", which the agent's check refuses because it can cancel the
  acknowledgement before it. The caregiver's own fallback phrase is spoken
  instead.
- **On the floor for 10 seconds.** Rule 5 escalates without calling the
  model at all: a critical alert on the caregiver's phone that repeats until
  it is seen, and *"Someone is coming to help."* at the bedside.

Every arrow in the video is one typed event on a Redis stream
(`PersonState`, `Utterance`, `Say`, `Notify`, ...). Services never call
each other, which is why each part can be replayed and tested on its own.

### Simulated nights: Claude finds the bugs, a person decides

https://github.com/user-attachments/assets/340c0c3c-2ca6-4e78-b4e2-a314e459c27f

74 s, silent. Also in the repository as an
[animated image](docs/media/scene-lab-loop.webp) and an [MP4](docs/media/scene-lab-loop.mp4).

No one can test a night companion at 3 a.m. with a real person in the
room. [`scene_lab`](tests/scene_lab) plays simulated nights against the
real stack instead, and closes the loop from bug to fix:

1. **Claude Opus directs.** It writes each scene (a fall, a toilet trip, a
   person who is hard of hearing, a question asked while the agent is
   speaking) and, when
   a scene breaks something, writes a smaller variant to isolate the cause.
2. **Claude Sonnet plays the person**, through real synthesised speech. The
   real `listen`, `agent` (with its local model) and `embodiment` services
   answer, exactly as they would at the bedside.
3. **Fifteen checks read every trace**: did each question get a reply
   within 5 seconds, did the agent talk over the person, did it fall silent
   when it shouldn't. They need no hand-written answer key.
4. **Opus triages the night.** It traces each cluster of failures to the
   code, tries the fix in a throwaway copy of the repository, replays the
   failing moment and writes a ranked fix list. It has no network, Docker or
   commit access.
5. **A person decides.** Nothing is applied until the owner asks for it,
   and the fix list puts open policy questions to them ("What should raise a
   second, critical alert while escalated?"). Later nights confirm the fix.

The video follows one real bug. In a simulated fall, a woman lying on the
floor said "Just want my bed" and was told *"That's right, Jean, take your
time getting back to bed."*, twice. Triage proposed a new veto, and a replay
of the same moment sent that sentence 0 times instead of 2. It landed in
`862537b`. Before the fix, the prompt went out 3 times in 8 floor scenes.
In the four director runs since, it has gone out 0 times in 18 floor
scenes.

`scene_lab` is a development tool: none of its Claude calls run at the
bedside, where the agent's model is local.

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

## What counts as the right behaviour

"Correct" is not left to the model, or to whoever wrote the code that day.
It comes from two places.

**Published dementia-care guidance.** A guideline pack,
[`tests/decision_bench/guidelines.md`](tests/decision_bench/guidelines.md),
holds 25 short clauses, each a paraphrase of one source passage with a
link. The sources are NICE guideline NG97, the Alzheimer's Association's
caregiver guidance, a Cochrane review of validation therapy, Kitwood's
person-centred care, the DICE approach to behaviour, and a BMJ cohort study
of falls in people over 90. A person has read the source behind 23 of the
clauses and ticked them as fair; the other two cannot be cited until they
are. The agent's veto rules cite the clause each one comes from, and the
decision benchmark's labels may cite only ticked clauses.

**People.** Benchmark labels are drafted by an isolated Claude annotator
that sees only the guidelines and the scenario, never the agent's code, and
a human reviews every one; disagreements are kept on file. The caregiver
writes the phrases the agent speaks. Where the guidance is silent (how many
seconds to wait, when to alert), the number is a caregiver or owner
setting, labelled as such, never presented as evidence. And the fix lists
from simulated nights put open policy questions to the owner rather than
settling them in code.

None of this makes Night Companion clinically validated. It makes each rule
traceable to a source or to a named human decision.

## Privacy

Everything runs on hardware in the home. Camera frames and audio never
leave the box and are not written to disk unless the caregiver turns on a
documented debug option; only structured events (state changes, what was
said) are stored. An optional cloud fallback for hard reasoning cases is
off by default, and when enabled it only ever sees text, never images or
audio.

The repository contains no recordings of anyone other than the project
owner. The demo videos in [`docs/media`](docs/media) show the owner acting
out night-time scenarios, with their consent. Test fixtures are event logs
and synthetic scenarios; recorded footage used for evaluation lives outside
the repository and is never committed.

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

Behaviour has been tested in simulated nights (`scene_lab`, above): five
director hours with automatic triage, 74 scenes, 387 flagged moments and 37
ranked fixes, measured against the real stack and its local model. The
method and the first measured runs are in
[tests/scene_lab/BASELINE.md](tests/scene_lab/BASELINE.md).

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
