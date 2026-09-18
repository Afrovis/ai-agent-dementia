# HANDOFF: working on Night Companion as an agent

Read this before touching any issue. It contains everything not in the code yet: the fixed decisions, the conventions, the contracts between services, and the rules that must never be broken. `PLAN.md` is the design rationale. This file is the execution brief.

Last updated: 2026-09-17. If you change a decision below, update this file in the same PR.

## 1. What this project is, in three sentences

A person with dementia wakes at night and gets up. A local agent notices, opens a session with the goal *return to bed*, and talks to them through a calm on-screen face and voice to reorient them, adapting the goal if they need the restroom. If nudging fails or there is a risk, it alerts the family caregiver by push notification.

## 2. Fixed decisions, do not re-litigate

| Area | Decision |
|---|---|
| Runtime | Docker compose on an M4 Mac, 16 GB. Ollama runs on the **host**, not in a container, reached at `http://host.docker.internal:11434`. |
| Language | Python 3.12 everywhere. One `pyproject.toml` per service, shared code in `shared/`. |
| Bus | Redis streams. At-least-once. Consumer groups per service. |
| Store | SQLite via SQLModel. One file at `data/night.db`. |
| Web | FastAPI. Dashboard uses HTMX. Embodiment page is plain HTML, CSS, and JS, no framework. |
| Bedside device (MVP) | A MacBook running the embodiment page in a fullscreen browser tab. The tab supplies screen, speaker, mic, and webcam. |
| Perception | Person detection plus pose classification on CPU on every frame. Vision LLM through Ollama only on state change. Frames are never stored. |
| Speech | faster-whisper `small.en` for STT. Piper for TTS. English only. |
| Text LLM | Ollama, small 4-bit instruct model. Model name is config, default `gemma4:e4b-mlx`, the same model the floor check uses. An OpenAI-compatible local server (`AGENT_LLM_BACKEND=openai`) is supported but not recommended on 16 GB. |
| Cloud fallback | Claude Opus 5, model id `claude-opus-5`, via the official `anthropic` Python SDK. Text only, never images. Off by default. |
| Notifications | ntfy by default. Backend interface allows Pushover and Telegram later. |
| Embodiment | Animated face plus very large text. No realistic human face, no voice clones. |
| Scope v1 | One room, one person, night window only. |

## 3. Non-negotiable rules

These come from the domain, not from taste. A PR that violates one is wrong even if it works.

1. **The LLM never owns safety.** State transitions, escalation timers, and hard limits live in deterministic code in the `agent` service. The LLM interprets, composes, and proposes. Every LLM proposal passes through `rules.validate()` before it has any effect.
2. **Camera frames and audio never leave the box and are never written to disk** unless a caregiver has explicitly enabled a debug option in the dashboard. Cloud fallback sends structured text only. If you need frames for a test, use the perception bench fixtures under `tests/perception_bench/`, recorded from consenting volunteers.
3. **Spoken output is one sentence, then silence for at least 8 seconds.** No questions that test memory. Never the words "no", "you can't", "you're wrong". Validate, then redirect.
4. **Fail loud to the caregiver, fail quiet to the person.** A crashed service shows the person a dim clock and sends the caregiver an `attention` notification. A crash must never look like a quiet night.
5. **`on_floor` or `absent` from the room beyond the configured limit skips every strategy and goes straight to `escalate_phone`.**
6. Everything the caregiver can see or change is in the dashboard. No hidden behaviour, no undocumented config keys.

## 4. Repository layout and conventions

```
ai-agent-dementia/
  PLAN.md                 design rationale
  HANDOFF.md              this file
  docker-compose.yml
  .env.example            every env var with a comment, no secrets
  shared/                 pip-installable package `nc_shared`
    nc_shared/events.py   pydantic event models (section 5)
    nc_shared/bus.py      Redis stream publish/subscribe wrapper
    nc_shared/config.py   settings loader (env + yaml)
  services/<name>/
    Dockerfile
    pyproject.toml
    <name>/               package
    tests/
  config/
    person.example.yaml
    strategies.example.yaml
  data/                   gitignored, sqlite, photos, voice clips, certs
  tests/
    perception_bench/
    dialogue_bench/
    fixtures/events/*.jsonl   recorded bus traffic for replay
  docs/
```

Conventions:

- Each service is a package with a `main.py` exposing `run()`. Docker runs `python -m <name>`.
- Config precedence: env var, then `config/*.yaml`, then defaults in code. All keys are documented in `.env.example` or the example yaml.
- Logging: structured JSON to stdout, one line per event, with `service`, `session_id`, and `event_type` fields.
- Tests: `pytest`. Every service must be testable with no camera, mic, Ollama, or Redis by using the replay fixtures and a fake bus (`nc_shared.bus.FakeBus`).
- Formatting and lint: `ruff format` and `ruff check` with the repo config. Type hints on all public functions.
- Commits: short imperative subject. Reference the issue as `#N` in the body. Do not commit anything under `data/`.
- Branch per issue: `issue-<N>-<slug>`. Open a PR against `main`.

## 5. Event contracts

All events are pydantic models in `shared/nc_shared/events.py`, serialised as JSON on Redis streams. Every event carries `ts` (UTC ISO 8601), `source` (service name), and `session_id` (nullable outside a session).

| Stream | Event | Producer | Key fields |
|---|---|---|---|
| `frames_raw` | `RawFrame` | `embodiment` (browser bridge) | `jpeg: bytes`, encoded `width`/`height`, intrinsic `source_width`/`source_height` (nullable for old replays), `source_kind: browser or usb or rtsp`. Ungated, pre-motion-gate frames. Only the browser source goes over the bus; `capture`'s own USB/RTSP cameras are read in-process and never reach this stream. Never persisted. Short retention (`MAXLEN ~ 50`). |
| `frames` | `Frame` | `capture` | `jpeg: bytes`, encoded `width`/`height`, intrinsic `source_width`/`source_height` (nullable for old replays), `source_kind: browser or usb or rtsp`. `capture` is the sole producer: it reads `frames_raw` (browser) or its camera directly (USB/RTSP), applies the motion gate, and republishes what it admits here. Short retention (`MAXLEN ~ 50`). |
| `person` | `PersonState` | `perceive` | `state: in_bed, sitting_up, standing, walking, on_floor, absent`, `confidence`, `zone: bed, door, bathroom_path, other`, `scene_note: str or None` |
| `speech_in` | `SpeechStarted`, `Utterance` | `listen` | onset: no payload beyond base fields; utterance: `text`, `confidence`, `duration_s` |
| `session` | `SessionState` | `agent` | `phase: IDLE, OBSERVING, ENGAGED, COOLDOWN, ESCALATED`, `goal`, `strategy_index` |
| `session` | `GoalChanged` | `agent` | `from_goal`, `to_goal`, `reason` |
| `cloud` | `CloudCall` | `agent` | `task: interpret or plan`, `model`, `payload: JSON text data only` |
| `say` | `Say` | `agent` | `text`, `strategy`, `interruptible: bool`, `clip_id: str or None` (when set, embodiment plays the consented caregiver-uploaded clip while still showing `text`) |
| `show` | `Show` | `agent` | `face: asleep, awake, speaking, listening`, `headline`, `body`, `photo_id or None`, `brightness: 0 to 1` |
| `notify` | `Notify` | `agent`; any service on fault (currently `embodiment` and `listen`) | `level: info, attention, critical`, `title`, `body`, `repeat_until_ack: bool` |
| `ack` | `Ack` | `dashboard`, `notify` | `notify_id` |
| `light` | `LightCommand` | `agent` | `light: hallway`, `state: on or off`, `reason` |
| `audio_in` | `AudioChunk` | `embodiment` (browser bridge) | `pcm16: bytes`, `sample_rate: 16000` |
| `health` | `Health` | every service, every 30 s | `service`, `ok: bool`, `detail` |

Rules:

- Never add a field that carries image or audio data to any stream other than `frames`, `frames_raw`, and `audio_in`.
- `frames`, `frames_raw`, and `audio_in` are capped streams. Everything else is persisted to SQLite by `store`.
- Add new events by editing `events.py` and this table in the same PR.

## 6. The agent core, summarised for implementers

Phases: `IDLE → OBSERVING → ENGAGED → COOLDOWN → IDLE`, with `ENGAGED → ESCALATED → COOLDOWN`. This is the summary for the ordinary, nudging path; rule 5 (below) can escalate straight to `ESCALATED` from any of `IDLE`, `OBSERVING`, `ENGAGED`, or `COOLDOWN`, and outranks this summary when the two disagree.

- Enter `OBSERVING` on `sitting_up`, `standing` or `walking` inside the night window. Wait 20 s. If back `in_bed`, return to `IDLE`. The night window gates starting this nudging session only; it does not gate rule 5.
- Enter `ENGAGED` after 20 s up, or on any `Utterance`. Goal defaults to `return_to_bed`.
- In `ENGAGED`, run strategies in configured order (issue #14, `agent.strategies.StrategyEngine`). Each strategy: emit `Show`, optionally `Say` (validated by `rules.validate_say()`, HANDOFF.md rule 3), wait its dwell time, evaluate. Move to the next on no progress -- "progress" is currently just heading to or reaching the bed (`zone == "bed"` or `state == "in_bed"`), the one signal issue #14 can observe.
- Goal switches: `restroom`, `drink_water`, `comfort`, `wait_for_caregiver`. `restroom`/`drink_water`/`comfort` return to `return_to_bed` when satisfied; `wait_for_caregiver` does not, by design (see below). Every switch goes through `rules.validate_goal()` (issue #13, the goal-change equivalent of `rules.validate()`) and emits `GoalChanged`. Perception seeing a confirmed `PersonState.zone` of `door` or `bathroom_path` enters `restroom`, and it returns to `return_to_bed` once the person is seen back at the bed or a configured timeout (`AGENT_RESTROOM_TIMEOUT_SECONDS`) elapses. Entering the goal selects `path_light` and emits an idempotent hallway-light `on` command; returning to `return_to_bed` emits `off` and selects `guided_return` when enabled. An escalation during the trip keeps the light on until the person is stably back in bed. Issue #15 adds structured local-LLM interpretation and planning: stated restroom needs and pain propose `restroom`/`comfort`, and planner goal proposals enter only through `Session.propose_goal()`. Invalid proposals are inert. Issue #16 loads the caregiver-authored `config/person.yaml` (`PERSON_PATH`, with example/default fallbacks) and injects every profile field into interpret, compose, and plan prompts. Other goal success conditions remain limited by what the current sensors can observe.
- `ESCALATED` when strategies are exhausted (issue #14: implemented -- every configured strategy disabled or on cooldown, either at `ENGAGED` entry or mid-session), distress is detected in two consecutive structured interpretations (issue #15), or rule 5 fires. Rule 5 fires on `on_floor`/`absent` beyond its configured limit from *any* phase, session already running or not, and regardless of the night window (HANDOFF.md rule 5 has no "only if a session is live" qualifier). Every path into `ESCALATED` forces `escalate_phone` selected (issue #14) and emits `Notify(critical or attention, repeat_until_ack=True)`, and sets the goal to `wait_for_caregiver`. That goal's PLAN.md success condition, "caregiver present", is not observable by this system (an `Ack` on a phone is not the same fact as someone being in the room), so it is never satisfied by code; `escalate_phone` stays selected and the session ends the ordinary way below instead, and the goal resets to `return_to_bed` on that return to `IDLE`.
- `COOLDOWN` for 5 min after `in_bed` is stable for 2 min. No new *nudging* session during cooldown; rule 5 can still escalate out of it.

LLM calls, all with structured JSON output validated by pydantic:

| Call | Input | Output | Budget |
|---|---|---|---|
| `interpret` | last utterance, last 3 turns, profile | `intent` enum, `distress: 0 to 3` | under 1 s |
| `compose` | strategy name, caregiver phrase template, profile, time, scene_note | one sentence, max 20 words | under 2 s to first token |
| `plan` | full session state | `next_strategy or goal_change`, `confidence` | under 2 s |

`plan` output is advisory. `rules.validate()` rejects any transition not in the table above and any strategy that is disabled or on cooldown (issue #14: `rules.validate_strategy()`, called by `agent.strategies.StrategyEngine` for every candidate it considers).

Cloud fallback: only for `interpret` and `plan`, only when `enable_cloud_fallback` is true for the person, only after two consecutive `unclear` intents or `confidence < 0.4`. Log a `CloudCall` row with the exact payload.

Issue #25 implements that fallback through the official Anthropic Python SDK
using `claude-opus-5`. It is disabled by default in `person.yaml`; when enabled,
two consecutive local `unclear` interpretations or a local planner confidence
below `0.4` retry that one operation in Claude. Composition always stays local.
The exact structured text data sent is published as a text-only `CloudCall`
before each request, persisted by `store`, and displayed in History. Missing or
failed cloud access preserves the local result and never bypasses the rule layer.

Issue #17's dialogue regression suite lives in `tests/dialogue_bench/`: 50
synthetic scenarios score overall/per-class intent accuracy and every composed
sentence against the runtime speech rules. It can compare three or more local
Ollama models without Redis, camera, microphone, or stored personal data.

Issue #18 implements `listen`: it consumes `SessionState` plus the browser
bridge's capped `audio_in` stream, discards audio in `IDLE`, and uses WebRTC VAD
in `OBSERVING`, `ENGAGED`, `COOLDOWN`, and `ESCALATED`. Complete utterances are
kept only in memory, transcribed locally by faster-whisper `small.en` on CPU,
and published as session-attributed `Utterance` events. A USB speakerphone is
selected as the browser/OS input device and therefore uses the same media
bridge; the service never writes audio to disk. Model weights live under
`data/models/faster-whisper`, and a transcription failure publishes one
`attention` notification and unhealthy heartbeats until a later transcription
succeeds.

Issue #19 implements speech output in `embodiment`: every `Say` is synthesized
locally with Piper's `en_US-lessac-medium` voice at 0.85 speed, cached under an
opaque WAV id in the container's ephemeral storage, and played by the bedside
browser over the same HTTPS origin.
At service startup it expands and pre-renders every configured fixed strategy
phrase, including all twelve possible greeting hours, so their first use does
not wait for inference. LLM-composed phrases are cached after their first use.
Generated audio never crosses Redis or enters retained `data/`; a synthesis
failure preserves the text display and emits an `attention` notification.

Issue #20 implements barge-in: `listen` publishes a transcript-free
`SpeechStarted` as soon as WebRTC VAD sees an onset, before the utterance is
complete. `embodiment` forwards it to the browser, which immediately stops the
current `Say` only when `interruptible=True` and switches the face to listening.
The browser requests hardware/OS echo cancellation; there is no software AEC.

Issues #22 and #23 implement the caregiver dashboard's Tonight, History,
Profile, Strategies, Media, and Zones pages. Profile and strategy edits are
validated and atomically written to `config/person.yaml` and
`config/strategies.yaml`; the agent processes reload them at startup rather
than watching files. Photo uploads are validated JPEG/PNG/WebP files under
`PHOTO_DIR`. Family voice uploads are explicit, consent-labelled, validated
PCM WAV files under `VOICE_CLIP_DIR`; they are referenced by opaque `clip_id`
on `Say` events while the audio bytes stay off Redis. Upload bytes and profile
text are never logged.

Issue #24 implements the morning summary in `store`, which owns the complete
SQLite event history. At `MORNING_SUMMARY_TIME` in `TZ`, it publishes one
informational `Notify` for the prior night with wake-up count, time-to-settle
durations, the last spoken strategy before each non-escalated resolution,
health faults, and escalation count. A SQLite delivery marker prevents
duplicate summaries across restarts.

Issue #26 applies `DATA_RETENTION_DAYS` (90 by default) to persisted events and
morning-summary delivery markers. The store prunes expired rows at startup and
hourly. The authenticated dashboard System page shows the effective period and
retained row count, downloads a no-cache JSON history export, and securely
deletes all retained history on request. Manual deletion deliberately leaves
the caregiver's profile, strategies, zones, photos, and voice clips intact;
new live events continue to be stored immediately afterward.

Issue #27's two-week volunteer evaluation is prepared by `DRY_RUN=true`. In
that mode `notify` suppresses all outbound delivery even when `NTFY_URL` is
configured, while `Notify` events continue through the bus into retained
History for daily review. The dashboard System page makes the active mode
visible. This is an evaluation-only override of rule 4 and must not be used for
real care; follow `docs/DRY_RUN.md`, and return the flag to `false` before a
supervised pilot.

Issue #51 implements the private video-evaluation `reconcile`, `score`, and
`replay` stages. Reference timelines are never scored before an explicit human
confirmation stamp. Reports include exact and upright-collapsed frame metrics,
detection and event metrics, and explicit measured/met/not-measurable PLAN.md
gates. End-to-end replay rebuilds the checkout, uses an always-on night window,
retains only non-frame bus outputs, and cleans capped frame streams and the
ephemeral stack afterward.

Issue #52 resolves the browser bridge's aspect-ratio defect. The complete
camera image is letterboxed into the bridge JPEG instead of being
horizontally squashed, and frame events retain both encoded and intrinsic
camera dimensions. The encoded size is 640 by 480 (#55); the page's
`FRAME_WIDTH`/`FRAME_HEIGHT` in `services/embodiment/embodiment/static/script.js`
are the authority. The numerical detector comparison and its limitations are
recorded in `docs/VIDEO_EVAL.md`.

## 7. Strategy catalogue

Implement in this order. Numbers match `PLAN.md` section 5.3.

| # | id | Show | Say | Dwell | Status |
|---|---|---|---|---|---|
| 1 | `ambient_orient` | brightness 0.3, time as words, "It is night", name | none | 30 s | implemented, issue #14 |
| 2 | `soft_greeting` | face awake, brightness 0.5 | greeting template | 20 s | implemented, issue #14 |
| 3 | `orient_time_place` | room photo behind face | place and time template | 30 s | implemented, issue #14 |
| 4 | `validate_and_redirect` | face listening | composed from utterance | 20 s | implemented, issues #14/#15/#16 -- local structured composition with the full person profile, deterministically validated, with the caregiver template as the failure fallback |
| 5 | `guided_return` | brightness 0.7, bed direction text | step template | 30 s | implemented, issue #14 |
| 6 | `familiar_voice` | family photo | plays uploaded clip | clip length + 15 s | implemented, disabled by default; requires a configured `clip_id` whose WAV exists |
| 7 | `music_or_story` | dim, photo | plays track | track length | not implemented -- needs a caregiver track-upload path that does not exist yet |
| 8 | `path_light` | bathroom direction | one sentence | restroom goal | implemented, issue #21 -- goal-specific rather than part of the ordinary ladder; `light` controls a feature-flagged local Shelly plug |
| 10 | `escalate_phone` | dim clock | "Someone is coming to help." | until `Ack` | implemented, issue #14 -- selected unconditionally on entering `ESCALATED` and stays selected; see `agent.strategies.StrategyEngine.force` |

Strategy 9 (`escalate_gentle`) needs hardware and is out of scope. `agent.strategies.py` is the catalogue and selection engine; ordering, enable/disable, cooldown, dwell, and every phrase for the eight implemented strategies live in `config/strategies.example.yaml`, editable by the caregiver (`STRATEGIES_PATH`, falling back to code defaults, same convention as `zones.yaml`). The `light` service defaults to hardware disabled and supports a Shelly Gen2+ local RPC switch when `LIGHT_ENABLED=true`; all keys are documented in `.env.example`.

## 8. Milestones and issues

Board: https://github.com/users/Afrovis/projects/2. Repo: https://github.com/Afrovis/ai-agent-dementia. Pick the lowest-numbered open issue in the lowest open milestone unless told otherwise.

| Milestone | Issues | Done when |
|---|---|---|
| M0 Skeleton | 1 to 6, 28 | `docker compose up` shows the face on a LAN browser, the fake agent cycles all states, the browser bridge streams webcam and mic, ntfy receives a test alert, replay tooling works |
| M1 Perceive | 7 to 11 | `PersonState` events are correct on the perception bench fixtures at the targets in `PLAN.md` section 12 |
| M2 Sessions | 12 to 17 | full session runs end to end on replayed fixtures with a real local LLM, dialogue bench passes |
| M3 Voice | 18 to 21 | a person can say "I need the toilet" and the goal switches, light turns on, agent guides back |
| M4 Caregiver | 22 to 27 | caregiver can configure everything in the dashboard, morning summary arrives, two-week dry run completed |
| M5 Video eval | 11, 48 to 52 | every confirmed clip under `../data-ai-agent-dementia/` scores through `tools/video_eval` and perception bench tier 3, the three defects are fixed, and the pose backend and bridge aspect decisions are made with numbers |

Status on 2026-09-17: M0 to M3 and M5 are met. M4 is met except issue 27, the
two-week volunteer dry run, which is the only open issue on the board. Each
M5 stage is scoped by the matching section of `docs/VIDEO_EVAL.md`; read that
file rather than the closed issues.

Work since M5 has been accuracy and tooling rather than new milestones:
floor detection (#55, `docs/FLOOR_DETECTION_HANDOFF.md`), bed occupancy (#60,
`docs/BED_OCCUPANCY_2026-09-15.md`), pose backends (#61, #63), walking and
sticky bed occupancy (#62, `docs/WALKING_BED_OCCUPANCY_PLAN.md`), the agent's
text model (#66, `EXPERIMENTS.md`), and the volunteer recording site
(`volunteer/PLAN.md` and `volunteer/HANDOFF.md`, which carry their own brief).

Two habits from the closed work still apply to anything new:

- Build LLM work against the `FakeLLM` in `shared/` first, then the real
  model. Every test must pass with no Ollama.
- An evaluation number is only trusted once the build that produced it is
  sound; a stale or broken container silently produces plausible numbers.

## 9. Local development

Prerequisites on the host: Docker Desktop and Ollama, with `ollama pull
gemma4:e4b-mlx` (the agent's text model, shared with the floor check) and
`ollama pull moondream` (scene notes).

```
cp .env.example .env
docker compose up --build
```

`embodiment` is on `EMBODIMENT_PORT` (8443) and `dashboard` on
`DASHBOARD_PORT` (8444). `embodiment` serves HTTPS when `CERT_FILE` and
`CERT_KEY` both exist and plain HTTP otherwise, which is enough on
`localhost` to see the face. The camera and microphone bridge is stricter:
browsers block `getUserMedia` on any page with a certificate error, so a
certificate the browser does not already trust leaves the face rendering and
the bridge dead. `CLAUDE.md` has the Tailscale procedure that works from
other devices, including the macOS sandbox trap and how to verify trust
without `curl -k`.

Without hardware, record and replay real bus traffic from inside a container,
where `nc_shared` is already installed:

```
docker compose exec store python -m nc_shared.replay record redis://bus:6379 /app/data/rec.jsonl
docker compose exec store python -m nc_shared.replay play redis://bus:6379 /app/data/rec.jsonl --speed 10
```

`./data` is mounted into every container, so a file written to `/app/data`
appears in `data/` on the host.

Run tests for one service: `cd services/agent && pytest`.

## 10. Definition of done for any issue

- The acceptance line in the issue is demonstrably met, and the PR description says how it was verified.
- Unit tests exist and pass with no hardware and no Ollama.
- `ruff check` and `ruff format --check` pass.
- New config keys are documented in `.env.example` or the example yaml.
- New events are in `events.py` and in section 5 of this file.
- `ARCHITECTURE.md` is regenerated when events, streams, subscriptions, or
  outside-world connections change.
- Nothing under `data/` is committed. No frame or audio bytes are logged or persisted.
- If a fixed decision or rule in this file changed, this file changed in the same PR and the PR title starts with `decision:`.

## 11. Things that look like shortcuts but are not allowed

- Letting the LLM return the next phase directly. It returns a proposal, rules decide.
- Storing frames "just for debugging" without the dashboard toggle.
- Using a cloud vision model. Vision is local only, always. This is about the running system. Offline evaluation may send face-blurred, verified contact sheets of the project owner's own test recordings to a cloud model for reference labelling, under the rules in `docs/VIDEO_EVAL.md` (decision by the project owner, 2026-09-13). Nothing from a real person's night ever leaves the box.
- Asking the person a question that tests memory, even in a test fixture. Fixtures are reviewed against rule 3.
- Skipping the `OBSERVING` delay because it makes demos slower. Make it configurable, default 20 s.
- Software echo cancellation. The MVP relies on the MacBook's built-in AEC, v1 on a speakerphone.

## 12. Open questions, and who decides

| Question | Default until decided | Decider |
|---|---|---|
| Pose model: MediaPipe Pose vs YOLOv8-pose | Decided with numbers on RGB: `PERCEIVE_POSE_BACKEND=yolo` with YOLO11s-pose at 640 input (PR #55, `docs/FLOOR_DETECTION_HANDOFF.md`). MediaPipe and an MLX-native backend (#61) stay available behind the flag | decided (#55); reopen when an infrared clip exists |
| Bridge frame format: 320 by 240 squashed vs letterboxed, and resolution | Decided with numbers: letterbox with source dimensions sent alongside (`docs/VIDEO_EVAL.md` section 7, #52), at 640x480 (#55) | decided (#52, #55) |
| Local text model | Decided 2026-09-17 on the dialogue bench: `gemma4:e4b-mlx` in Ollama (#66, `EXPERIMENTS.md`, `tools/llm_speedtest`) | decided (#66) |
| Walking as a product state, distinct from standing | detection ships opt-in (#62, `docs/WALKING_BED_OCCUPANCY_PLAN.md`); nothing downstream depends on it yet | project owner |
| Smart plug for path light in v1 | manual night light, plug behind a feature flag | project owner |
| Morning summary contents | count, durations, what helped, faults | project owner after a caregiver interview |

## 13. Lessons from fixed defects

Resolved defects are not kept here once they have a test; git history has the
detail. These two cost real evaluation data and are easy to repeat.

- Constrain structured LLM output at generation time, never police it
  afterwards. `tools/video_eval label-local` once sent a bare
  `"format": "json"` to Ollama, so `person_visible` could arrive as a string
  and fail a strict `is bool` check: 27 of 488 frames on
  `2026-09-13_bedroom-sample-01` became null records, including a 20-frame
  block that swallowed a scripted `sitting_up` event outright. It now sends
  `LABEL_SCHEMA` as the `format` and coerces only unambiguous spellings.
  Covered by `tools/video_eval/tests/test_labellers.py`.
- Any sampling shortcut that propagates one result forward to the frames it
  skips turns a single bad response into a multi-second gap. Adaptive
  labelling does exactly this, so a failure there is never a single-frame
  failure. Weigh that before adding another such shortcut.
