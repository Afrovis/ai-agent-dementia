# Night Companion: an embodied agent for night-time reorientation

Status: implementation in progress, v0.1 (updated 2026-09-13)

Progress through 2026-09-13:

- M5 issues #48 and #49 are implemented on branches `issue-48-pin-mediapipe`
  and `issue-49-video-eval-prepare-predict`.
- The perceive MediaPipe extra is pinned to `>=0.10,<1.0`; the slim image
  includes the OpenCV runtime libraries required to construct the backend.
- `tools/video_eval` now provides idempotent `prepare` and offline `predict`
  commands, including squash/letterbox bridge frames, provenance metadata,
  private-root indexing, MotionGate replay, zones, StateTracker hysteresis,
  and backend timing records.
- The local bedroom sample produced 488 review frames and 488 frames in each
  bridge variant. A real pinned-MediaPipe run completed in 9.3 seconds over
  488 frames (466 backend calls, 22 gate drops); no images or derived data are
  committed here.
- Issue #50 adds fail-closed YOLO/MediaPipe privacy blurring, verified 3x3
  contact sheets, adaptive local Ollama labels, and human-gated Codex labels.
- Remaining M5 work: reconcile/score/replay (#51), and the evidence-based
  bridge aspect-ratio decision and fix (#52). The
  sample has no caregiver-authored zones or confirmed reference timeline yet,
  so state accuracy and end-to-end replay remain pending.

## 1. Purpose

A person living with dementia at home often wakes at night, is disoriented in time and place, and gets up. This leads to wandering, falls, and exhausted caregivers. Night Companion is a local, always-on agent in the bedroom that:

1. notices when the person wakes and gets up,
2. opens a **session** with a goal (default: *get safely back to bed*),
3. talks with the person through a calm screen face and voice to reorient them in time and space,
4. adapts the goal when the person has a legitimate need (restroom, water, pain),
5. escalates to the family caregiver when nudging fails or a risk is detected.

## 2. Decisions taken (from the design interview)

| Topic | Decision |
|---|---|
| Person and operator | Person with dementia at home. A family caregiver configures the system and receives alerts. |
| Scope v1 | Single bedroom. One person. Night hours only (configurable window, e.g. 22:00 to 07:00). |
| Hardware v1 | The M4 Mac (16 GB) is the box. Everything runs in Docker except Ollama, which runs on the host for Metal GPU access. |
| Patient-facing device (MVP) | A MacBook screen. The embodiment is a web page served over the LAN, so any laptop or tablet can be the bedside device by opening one URL. For the MVP the browser also provides the mic, speaker, and webcam. USB IR camera and speakerphone are added in M1 and M3 as alternative sources. |
| Sensing | Camera with a local vision model, plus two-way voice. Bed sensor and door sensors are deferred to v2 but the event model already accounts for them. |
| LLM locality | Local by default (Ollama). Cloud fallback for hard reasoning only, text only, never images, behind an explicit switch. |
| Embodiment | Simple animated face with very large text (time of night, one sentence, optional family photo). |
| Language | English only in v1. |
| Caregiver interface | Local web dashboard on the LAN plus push notifications. |
| Escalation | Ordered strategies, then a phone notification to the caregiver. |
| Implementation language | Python 3.12. Chosen because the vision, speech, and Ollama tooling is Python-first. |
| Cloud fallback model | Claude Opus 5 (`claude-opus-5`) through the official `anthropic` Python SDK. |

## 3. Guiding principles

- **Calm over clever.** Every interaction must lower arousal. Slow speech, warm tone, one idea per sentence, no questions that require memory ("do you remember...?").
- **Never argue with the person's reality.** Validate the feeling, then redirect. If they say they must go to work, the response is "It is the middle of the night, work starts in the morning. Let's rest first."
- **The person is not a problem to solve.** The agent helps, it does not restrain. If the person is safe and awake, it is fine to sit with them.
- **Privacy by construction.** Camera frames never leave the box and are never stored by default. Only structured events are logged. Cloud fallback sends text only.
- **The caregiver is in control.** They write the phrases, choose the photos, set the strategies, and can turn any feature off.
- **Fail safe.** If a model crashes or the camera dies, the system falls back to a plain night clock and notifies the caregiver. A dead agent must never look like a quiet night.

## 4. System overview

```
                ┌──────────────────────────────────────────────────────────┐
                │ Host (macOS, M4)                                         │
                │   Ollama: text LLM, vision LLM (Metal)                   │
                └──────────────┬───────────────────────────────────────────┘
                               │ http://host.docker.internal:11434
┌──────────────────────────────┼───────────────────────────────────────────┐
│ docker compose               │                                           │
│                                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ camera   │──▶│ perceive │──▶│              │──▶│ embodiment       │  │
│  │ capture  │   │ (vision) │   │  session     │   │ (face + text +   │  │
│  └──────────┘   └──────────┘   │  agent       │   │  TTS)            │  │
│  ┌──────────┐   ┌──────────┐   │  (goal, plan,│   └──────────────────┘  │
│  │ audio in │──▶│ listen   │──▶│  strategies) │──▶┌──────────────────┐  │
│  └──────────┘   │ (STT)    │   │              │   │ notify           │  │
│                 └──────────┘   └──────┬───────┘   │ (ntfy / pushover)│  │
│                                       │           └──────────────────┘  │
│                    ┌──────────────────┴─────────┐                        │
│                    │ event bus (Redis streams)  │                        │
│                    └──────────────────┬─────────┘                        │
│                    ┌──────────────────┴─────────┐  ┌───────────────────┐ │
│                    │ store (SQLite: events,     │  │ dashboard (web,   │ │
│                    │ sessions, config)          │  │ LAN only)         │ │
│                    └────────────────────────────┘  └───────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

Services communicate over a small event bus. Each service is a separate container so a crash in vision does not take down the face.

This drawing is the intended design. See [ARCHITECTURE.md](ARCHITECTURE.md) for
the generated, as-built picture of the services and streams in the current code.

### 4.1 Services

| Service | Responsibility | Tech |
|---|---|---|
| `capture` | Receives frames from a source and publishes them at low frame rate (2 to 5 fps at night), motion-gated. Sources: browser webcam streamed from the embodiment page (MVP), USB or RTSP camera (M1). | OpenCV, WebRTC or WebSocket receiver |
| `perceive` | Person detection and pose classification: `in_bed`, `sitting_up`, `standing`, `walking`, `on_floor`, `absent`. Fast model on every frame, vision LLM only on scene changes to describe what the person is doing ("holding a coat", "at the door"). Publishes `PersonState` events. | YOLO or MediaPipe for pose, Ollama vision model (e.g. `moondream` or `qwen2.5-vl` small) for descriptions |
| `listen` | Voice activity detection, speech to text, publishes `Utterance` events. Only active while a session is open plus a short grace window. Audio arrives from the browser mic on the embodiment page (MVP) or a local speakerphone (M3). | faster-whisper (small.en) |
| `agent` | The core. Owns sessions, goals, strategies, and the LLM conversation. Consumes state and utterances, emits `Say`, `Show`, `Notify`, and `GoalChanged` events. | Python, Ollama text model, optional Claude fallback |
| `embodiment` | Fullscreen web page reachable over the LAN: animated face, big text, photo. Plays TTS audio. In the MVP it also captures mic and webcam with getUserMedia and streams them to `listen` and `capture`. Consumes `Say` and `Show`. Served over HTTPS with a mkcert certificate because getUserMedia requires it off localhost. | FastAPI app, WebSocket updates, Piper TTS, any modern browser in fullscreen |
| `notify` | Sends caregiver alerts. Pluggable backends: ntfy (default, self-hostable), Pushover, Telegram. Repeats until acknowledged for critical alerts. | HTTP |
| `store` | SQLite database of events, sessions, and configuration. Nightly summary generation. | SQLite via SQLModel |
| `dashboard` | Caregiver web UI: configure person profile and phrases, upload photos, review night timelines, acknowledge alerts, watch a live status (not a live video by default). | FastAPI + HTMX |
| `bus` | Redis with streams, gives at-least-once delivery and replay for debugging. | Redis |

### 4.2 Hardware bill

MVP: the box plus a MacBook running one browser tab in fullscreen. The tab is the screen, mic, speaker, and camera. Testing happens with a dim lamp on because a laptop webcam has no night vision.

v1 additions:

- USB camera with IR night vision (e.g. a 1080p USB camera with IR LEDs, or a Wyze v3 in RTSP mode).
- USB conference speakerphone (mic array plus speaker, echo cancellation in hardware). This matters: the agent must hear the person while it is speaking.
- Screen: the MacBook, a 10 to 15 inch monitor on a mini PC, or an old iPad in kiosk mode. All of them just open the embodiment URL.
- Soft warm light controllable via a smart plug (optional but strongly helps: light up the path to the bathroom).

## 5. The session model

A session is the unit of interaction. It starts when the person leaves the `in_bed` state at night and ends when they are back in bed and calm, or when a caregiver takes over.

### 5.1 Session lifecycle

```
IDLE ──(sitting_up/standing/walking at night)──▶ OBSERVING
OBSERVING ──(still up after 20 s, or speaks)──▶ ENGAGED
OBSERVING ──(back in bed)──▶ IDLE
ENGAGED ──(goal reached)──▶ COOLDOWN ──(5 min calm)──▶ IDLE
ENGAGED ──(strategies exhausted or risk)──▶ ESCALATED
ESCALATED ──(caregiver acknowledges or resolves)──▶ COOLDOWN
```

- `OBSERVING` is deliberate. Many wake-ups resolve on their own. Speaking too early is intrusive.
- `COOLDOWN` prevents a second session from firing when the person rolls over.

### 5.2 Goals

A session has exactly one active goal. Goals form a small tree so the agent can switch and return:

| Goal | Success condition | Typical sub-goal |
|---|---|---|
| `return_to_bed` (default) | `in_bed` for 2 min | none |
| `restroom` | person went to bathroom and came back, then `return_to_bed` | `return_to_bed` after |
| `drink_water` | drank, then `return_to_bed` | `return_to_bed` after |
| `comfort` | distress reduced (calm voice, sitting still) | `return_to_bed` after |
| `wait_for_caregiver` | caregiver present | none |

The goal changes when:

- the person states a need ("I need the toilet") detected by STT plus LLM intent classification,
- perception sees the person heading to the bathroom door,
- the LLM planner proposes a change and the rule layer allows it.

Goal changes are events, so the dashboard timeline shows "22:41 goal: return_to_bed → restroom".

### 5.3 Strategies

Strategies are the agent's toolbox for reorientation. They are ordered and tried in sequence within a session. Each has a cost (intrusiveness) and a cooldown. The caregiver can enable, disable, reorder, and edit the text of each.

| # | Strategy | What happens | Intrusiveness |
|---|---|---|---|
| 1 | `ambient_orient` | Screen brightens gently, shows the time, "It is night", and the person's name. No speech. | 1 |
| 2 | `soft_greeting` | Face appears, one calm sentence: "Hello Anna, it's Tom's night helper. It's 3 o'clock at night." | 2 |
| 3 | `orient_time_place` | Explains where they are and what time it is, with a photo of the room in daylight or a familiar object. "You are home, in your bedroom. Everyone is sleeping." | 2 |
| 4 | `validate_and_redirect` | Listens, acknowledges the stated concern, redirects: "You're looking for your mother. You miss her. It's night now, let's rest and talk about her in the morning." | 3 |
| 5 | `guided_return` | Step by step instruction with the light: "The bed is behind you. Let's sit down on it." | 3 |
| 6 | `familiar_voice` | Plays a recorded message from a family member (caregiver uploads). | 3 |
| 7 | `music_or_story` | Plays a chosen calming track or a short familiar story. | 2 |
| 8 | `path_light` | For restroom goal: turns on the hallway light and says where the bathroom is. | 2 |
| 9 | `escalate_gentle` | In-home chime in the caregiver's room (v2, needs hardware). | 4 |
| 10 | `escalate_phone` | Push notification with the situation summary, repeated until acknowledged. | 5 |

Rules the agent must respect:

- Never ask questions that test memory.
- Never say "no" or "you can't". Redirect instead.
- Max one sentence per turn, then wait at least 8 seconds.
- If the person is distressed (raised voice, crying, pacing), skip to `validate_and_redirect` and shorten the escalation timer.
- If the person is on the floor or leaves the room for more than N minutes, skip straight to `escalate_phone`.

### 5.4 Where the LLM sits

The LLM is **not** in charge of safety. A deterministic rule layer owns state transitions, escalation timers, and hard limits. The LLM does three things:

1. **Interpret**: classify utterances into intent (`need_restroom`, `looking_for_person`, `wants_to_leave`, `confused_time`, `pain`, `fine`, `unclear`) and sentiment.
2. **Compose**: write the actual sentence for the chosen strategy, given the person's profile, the caregiver's preferred phrases, and the current context. Output constrained to one short sentence.
3. **Plan**: propose the next strategy or a goal change, as a structured JSON decision. The rule layer validates it against the allowed transitions before acting.

Local model: an instruction-tuned 7B to 8B model in Ollama (e.g. `llama3.1:8b` or `qwen2.5:7b`) with a 4-bit quant fits comfortably in 16 GB next to the vision model. Target: under 2 s to first token for a compose call.

Cloud fallback: when the local model returns `unclear` twice in a row, or when the planner's confidence is low, the agent may call Claude Opus 5 with the text transcript and structured state only. It is off by default, enabled per person in the dashboard, and every fallback call is logged in the timeline.

### 5.5 Person profile

Configured by the caregiver, injected into every prompt:

- preferred name, how they like to be addressed,
- who the caregiver is and how to refer to them,
- known recurring themes at night (looking for a deceased spouse, wanting to go to work, thinking children are small),
- what calms them (music, a specific photo, a phrase),
- what to avoid (mentioning certain people, loud sounds),
- physical notes (uses a walker, unsteady at night),
- restroom location relative to the bed.

## 6. Perception details

### 6.1 Pipeline

1. `capture` grabs frames at 2 fps. If the frame difference is under a threshold for 30 s, drop to 0.5 fps.
2. `perceive` runs a person detector on every frame it receives, then a pose classifier that maps keypoints to `PersonState`. This is cheap and runs on CPU.
3. When the state changes, or every 60 s while `ENGAGED`, `perceive` sends one frame to the vision LLM with a fixed prompt: "Describe in one sentence what the person is doing and whether they look distressed. Do not describe their identity." The output is attached to the `PersonState` event as `scene_note`.
4. Frames are dropped after processing. No video recording in v1. A caregiver-toggled option to keep a low-resolution thumbnail per session is a v2 item.

### 6.2 Night-specific issues

- IR video is grayscale and noisy. Test the detector on IR captures from the actual room before trusting it.
- A blanket covering the person confuses pose models. `in_bed` should be inferred from "no person standing or sitting in the bed zone" plus a caregiver-drawn bed zone on the frame.
- The bed zone, door zone, and bathroom-direction zone are drawn once in the dashboard on a daylight frame.

## 7. Voice details

- Wake-free. The mic is only active during `OBSERVING` and later states. This avoids constant listening.
- STT with faster-whisper `small.en` on CPU is around real time on the M4, good enough for short utterances.
- TTS with Piper, a warm, slow voice at 0.85 speed. Pre-render all fixed phrases at startup so the ambient and greeting strategies have zero latency. Implemented in issue #19: `embodiment` serves locally cached WAVs to the bedside browser without putting audio on the event bus.
- Barge-in: if the person speaks while the agent is speaking, stop TTS and listen.
- The speakerphone's hardware echo cancellation is what makes barge-in feasible. Do not attempt software AEC in v1.

## 8. Embodiment details

- One fullscreen page, dark background, warm amber accents. Nothing moves except the face's gentle blink and breathing animation.
- Layout: face in the upper half, text in the lower half at 72 px or larger. Time shown as words and a clock, "3 o'clock at night".
- Face states: `asleep` (dim, eyes closed), `awake`, `speaking`, `listening`. No exaggerated emotions.
- A caregiver photo fades in behind the face when a strategy asks for it.
- Brightness follows strategy intrusiveness. `IDLE` state shows only a dim clock.

## 9. Caregiver dashboard and notifications

- LAN-only web app, protected by a single password. No cloud account.
- Pages: Tonight (live status, current session, acknowledge button), History (timeline per night, session details, what the agent said and heard), Person profile, Strategies (enable, order, edit phrases, upload voice clips and photos), Zones (draw bed and door zones), System (model health, camera preview for setup only, cloud fallback toggle, export and delete data).
- Notifications through ntfy by default. Levels: `info` (session opened and resolved, sent in the morning summary only), `attention` (session escalated), `critical` (fall or left room, repeats every 60 s until acknowledged).
- Morning summary at a configured time: number of wake-ups, durations, what helped, anything to look at.

## 10. Data and privacy

- Stored: events, session transcripts (agent speech and STT text), configuration, model call metadata.
- Not stored by default: video, audio, frames.
- Retention: 90 days, configurable, one-click delete.
- Cloud fallback: opt-in, text only, logged, and the dashboard shows exactly what was sent.
- Threat model for v1: a device on the home LAN. Not hardened against a hostile home network. Document this clearly.

## 11. Safety and ethics notes

- This is an assistive tool, not a medical device and not a substitute for supervision. State this in the README and in the dashboard.
- Consent: the person with dementia may not be able to consent to a camera. The caregiver decides, ideally with prior expressed wishes. Document this and keep the camera indicator light visible.
- Deception: the face is presented as a "helper", never as a real person. `familiar_voice` uses real recordings by real family members with their consent, never cloned voices in v1.
- Failure disclosure: any night with a system fault is reported to the caregiver, never hidden.

## 12. Evaluation plan

Before any real use:

1. **Perception bench**: record consenting volunteers in the actual room at night doing scripted actions (sit up, stand, walk to door, lie on floor). Measure state accuracy and latency per state. Target: over 95 percent on `standing` and `on_floor`, under 2 s latency.
   - **Recorded-video evaluation** (added 2026-09-13): before volunteers, the project owner records themselves in their own bedroom doing the scripted actions. Each clip is downsampled to exactly what the browser bridge sends (320 by 240, squashed to 4:3, 2 fps), scored offline against a reference timeline built from the recorder's own scenario card, a local vision-language model, and a cloud model that only ever sees face-blurred contact sheets. The same clips are then replayed through the live stack to check agent transitions. RGB in lamp light first; infrared later. Tooling, runbook and privacy rules are in `docs/VIDEO_EVAL.md`.
2. **Dialogue bench**: 50 scripted scenarios with utterances such as "where is my husband", "I have to catch the train", "I need to pee". Judge the local model's intent classification and compose output for tone rules. Keep this as a regression suite.
3. **Dry run**: two weeks with a volunteer, no caregiver notifications, review timelines daily.
4. **Supervised pilot**: with the actual family, caregiver present the first nights.

## 13. Roadmap

### Milestone 0: Skeleton (week 1 to 2)
- Repo, docker compose, Redis bus, event schemas (pydantic), SQLite store.
- `embodiment` page with the face and text, served over HTTPS on the LAN, driven by a fake agent that cycles states. Tested on the MacBook.
- Browser media bridge: getUserMedia in the page streams webcam and mic to the box.
- `notify` with ntfy.
- Dev tooling: event replay from a JSONL file so every service can be tested without a camera.

### Milestone 1: Perceive (week 3 to 4)
- `capture` and `perceive` with pose classification and bed/door zones.
- Dashboard zone editor.
- Perception bench with recorded IR clips.

### Milestone 2: Sessions and strategies (week 5 to 6)
- `agent` state machine, goals, strategies 1 to 5 and 10.
- Local LLM interpret, compose, plan with structured outputs.
- Dialogue regression suite.

### Milestone 3: Voice (week 7 to 8)
- `listen` with faster-whisper and VAD, Piper TTS, barge-in.
- Restroom goal end to end with path light via smart plug.

### Milestone 4: Caregiver (week 9 to 10)
- Full dashboard: profile, strategies, history, morning summary.
- Cloud fallback behind a switch.
- Dry run.

### Milestone 5: Recorded-video evaluation (after the dry-run prep)
- [x] Fix the MediaPipe dependency defect: pin `mediapipe>=0.10,<1.0` and install the slim-image runtime libraries needed by OpenCV.
- [x] Build the first `tools/video_eval/` CLI stage: `prepare` and `predict`, as specified in `docs/VIDEO_EVAL.md`.
- [x] Build `blur`, `sheets`, `label-local`, `label-codex`, `reconcile`, `score`, and `replay` (#50 and #51), plus `visualize` (#54).
- Fill in perception bench tier 3 so confirmed clips score in `python -m perception_bench`.
- [x] Run the backend, squash vs letterbox, resolution, hysteresis and gate experiments on the RGB clips and decide the bridge fix and the pose backend with numbers: letterbox at 640x480 (#52, `docs/VIDEO_EVAL.md` section 7) and YOLO11s-pose at 640 input (#55, `docs/FLOOR_DETECTION_HANDOFF.md`).
- Record the scenario set in section 8 of `docs/VIDEO_EVAL.md` in RGB; infrared clips are a later round.

### Later
- Bed pressure sensor and door sensor via Zigbee or Home Assistant.
- Multi-room and hallway camera.
- Gentle in-home escalation hardware.
- Dutch language.
- Appliance build on a Linux mini PC or Jetson.

## 14. Repository layout

```
ai-agent-dementia/
  PLAN.md
  README.md
  docker-compose.yml
  .env.example
  services/
    capture/
    perceive/
    listen/
    agent/
    embodiment/
    notify/
    dashboard/
  shared/          # event schemas, config models, bus client
  config/
    person.example.yaml
    strategies.example.yaml
  data/            # sqlite, photos, voice clips (gitignored)
  tests/
    perception_bench/
    dialogue_bench/
  docs/
```

## 15. Open questions

- Which pose model handles IR and blankets best? Evaluate MediaPipe Pose vs YOLOv8-pose on real captures.
- Which local text model gives the warmest, most rule-following one-sentence outputs? Compare 3 candidates on the dialogue bench.
- Is a smart plug for path lighting in scope for v1, or manual night light?
- What does the caregiver want to see in the morning summary? Interview before building the dashboard.
