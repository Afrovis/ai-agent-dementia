# Night Companion

Night Companion is a local AI agent that keeps company with a person living
with dementia when they wake up at night. When someone with dementia wakes
in the dark, they can become disoriented — unsure of the time, the place, or
why they're awake — which is when wandering and falls happen. Night
Companion notices the wake-up with a camera, greets the person with a calm
face and voice on a bedside screen, reminds them where and when they are,
and gently guides them back to bed. If a real need comes up (the bathroom,
water, pain) it adapts instead of just repeating itself, and if nudging
doesn't work — or something looks genuinely wrong, like a fall — it wakes
up the family caregiver instead of the person.

It is **not a medical device** and **not a substitute for supervision**.
It's an assistive layer for nights when no one else is awake in the room.

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
3. If they've left the bed, a bedside screen shows a calm animated face,
   large text, and a soft voice — reorienting them ("It's 3am. You're at
   home in your bedroom. Let's go back to bed.") and, if they respond,
   listening and adapting instead of repeating a script.
4. A short list of strategies runs in order: a quiet clock, a gentle
   greeting, orienting text with a familiar photo, listening and
   responding, step-by-step guidance back to bed, lighting the hallway to
   the bathroom, and — if nothing works or something looks wrong — a push
   notification to the caregiver's phone that repeats until acknowledged.
5. Every session is logged for a caregiver dashboard and a morning summary,
   so the family can see what happened without watching video.

Everything runs on hardware in the home. Camera frames and audio never
leave the box by default; only structured events (state changes, what was
said) are stored. An optional cloud fallback exists for hard reasoning
cases, and it only ever sees text, never images or audio.

## Current status (v0.1, in progress)

**Working today:**
- The full session pipeline: wake detection → screen/voice reorientation →
  strategy escalation → caregiver alert.
- 7 of 10 caregiver strategies (quiet clock, greeting, orient with photo,
  listen-and-respond, guided return, hallway light, phone escalation).
- Voice in and out (speech-to-text, text-to-speech) with barge-in — the
  agent stops talking the moment the person starts speaking.
- The caregiver dashboard: live status, session history, profile and
  strategy editors, a zone-drawing tool, morning summaries.
- A 50-scenario automated dialogue test suite, all passing.
- An offline video-evaluation pipeline that scores the camera perception
  against hand-checked reference recordings, so accuracy work doesn't
  require a live camera or a person to test with.

**Not built yet:**
- Playing a family member's recorded voice, or music/story playback
  (strategies 6 and 7).
- A softer in-home chime escalation step, which needs v2 hardware.
- Multiple rooms, multiple people, or languages other than English.
- Bed or door pressure sensors — perception is camera-only for now.

**In flight:** fixing a video aspect-ratio bug in the browser camera
bridge, and finishing accuracy work on floor-fall detection (below).

## Results so far

Perception has been benchmarked offline against recorded bedroom video
(RGB, lamp-lit, hand-labelled reference timelines) rather than trusted on
sight. Two clips, more on the way:

- **Per-frame state agreement** (in bed / sitting / standing / walking /
  on floor / absent) with the best model and rule set: **0.78–0.90**
  across the two test clips, up from 0.56–0.65 with rules off.
- **Scripted events matched:** 9/9 and 11/11, with zero false events.
- **Fall / floor detection:** a new "height ratio" rule — comparing a
  detected person's box height against what a calibrated standing person
  should measure at that camera position — catches both scripted floor
  events with **zero false floor alarms**, versus 0% recall from the
  original geometry-only rule.
- **Latency:** perception runs in 12–60 ms per frame depending on model,
  well inside the ~500 ms budget at the 2 fps the camera streams.

The honest caveat: this is two clips of evidence, not a validated
accuracy rate. It shows the pipeline and rules move in the right
direction; more recorded nights are needed before treating these numbers
as reliable. Details are in
[docs/PERCEIVE_ACCURACY_2026-09-13.md](docs/PERCEIVE_ACCURACY_2026-09-13.md)
and [docs/FLOOR_DETECTION_HANDOFF.md](docs/FLOOR_DETECTION_HANDOFF.md).

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
| `embodiment` | The bedside screen: face, text, voice, camera/mic bridge. |
| `notify` | Caregiver alerts (ntfy by default; Pushover/Telegram pluggable). |
| `store` | SQLite persistence and nightly summaries. |
| `dashboard` | Caregiver web UI. |

Safety-relevant decisions — state transitions, escalation timers, when to
call the caregiver — live in deterministic code, never in the language
model. The LLM composes language and interprets intent; it doesn't decide
whether something is an emergency.

## Technology stack

- **Runtime:** Docker Compose, targeting a Mac mini/MacBook (M-series,
  16GB) with Ollama running on the host.
- **Language:** Python 3.12 across every service.
- **Perception:** YOLO11s (recommended) or MediaPipe pose estimation, plus
  an Ollama vision model for harder cases.
- **Voice:** faster-whisper for speech-to-text, Piper for text-to-speech.
- **LLM:** a local Ollama model by default (`gemma4:e4b-mlx`), with an
  optional text-only Claude fallback for hard reasoning, off by default.
- **Storage:** SQLite via SQLModel.
- **Web:** FastAPI for services, HTMX for the dashboard, plain HTML/JS for
  the bedside page.
- **Messaging:** Redis streams, at-least-once delivery, Pydantic-typed
  events.

## Where to read more

- [PLAN.md](PLAN.md) — full design and guiding principles.
- [HANDOFF.md](HANDOFF.md) — non-negotiable rules and execution brief;
  read this before picking up an issue.
- [ARCHITECTURE.md](ARCHITECTURE.md) — generated, as-built map of
  services, streams, and events. Do not edit by hand.
- [docs/VIDEO_EVAL.md](docs/VIDEO_EVAL.md) — how the offline video
  evaluation pipeline works.
- [docs/PERCEIVE_ACCURACY_2026-09-13.md](docs/PERCEIVE_ACCURACY_2026-09-13.md)
  and [docs/FLOOR_DETECTION_HANDOFF.md](docs/FLOOR_DETECTION_HANDOFF.md) —
  the latest accuracy investigations.
- [AGENTS.md](AGENTS.md) — how to run the stack locally and test it without
  hardware; shared by Claude Code (`CLAUDE.md` imports it) and Codex.
- [docs/TLS.md](docs/TLS.md) — certificates for the camera/mic bridge.
