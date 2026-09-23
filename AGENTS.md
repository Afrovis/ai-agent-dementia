# AGENTS.md

Guidance for coding agents and contributors. Claude Code and Codex both read
this file directly; there is deliberately no `CLAUDE.md` (see "Keeping these
instructions current").

Night Companion is a local, embodied AI agent that helps a person with dementia
return safely to bed at night. It is not a medical device or a substitute for
supervision. Changes that affect what the person sees or hears at night deserve
more care than the code alone suggests.

## Before making changes

- [HANDOFF.md](HANDOFF.md) holds the fixed decisions, event contracts, safety
  rules and the definition of done. It wins if anything here conflicts with it.
- [PLAN.md](PLAN.md) has the design rationale; [ARCHITECTURE.md](ARCHITECTURE.md)
  is the generated as-built service map; [EXPERIMENTS.md](EXPERIMENTS.md) is the
  dated log of measured results.
- Working in one service or bench? Read its own `AGENTS.md` or README first
  (listed under "Where the detail lives").

## Working rules

- Keep safety-critical state transitions, escalation timers and limits in
  deterministic code. LLM output is advisory and is always validated.
- Camera frames and audio stay on the device and are not persisted unless the
  caregiver enables the documented debug option.
- Services communicate only through Redis streams, never by calling each other.
- Add or change event schemas in `shared/nc_shared/events.py`, and update the
  event-contract documentation in the same change.
- Document every configuration key in `.env.example` or an example YAML file.
- Preserve the person-facing language rules in `HANDOFF.md`, especially the
  one-sentence output limit and the silence between spoken prompts.
- Log one structured JSON line per event to stdout, with a `service` field.
- Nothing under `data/` or `../data-ai-agent-dementia/` enters git:
  certificates, the SQLite database, recordings, frames and labels live there.
  Only face-blurred, verified frames may be sent to an outside model (Codex).

## Architecture

| Service | Role |
| --- | --- |
| `bus` | Redis streams broker. The only shared dependency. |
| `capture` | Motion-gates raw frames and publishes `Frame`. The only producer of `frames`. |
| `perceive` | Person detection and pose classification on frames; gaze point for the eyes. |
| `listen` | Voice activity detection and speech to text, publishes `Utterance`. |
| `agent` | Session state machine. Emits `Say`, `Show`, `Notify`, `GoalChanged`, `LightCommand`. |
| `light` | Feature-flagged local Shelly smart-plug control for the restroom path. |
| `embodiment` | Fullscreen HTTPS page: glowing eyes, big text, photos, media bridge. |
| `notify` | Caregiver alerts. ntfy by default. |
| `store` | SQLite persistence and nightly summaries. |
| `dashboard` | Caregiver UI for live status, history, profile, strategies, media, and zones. |

`volunteer/` is a separate public recording site that feeds `tools/video_eval`
with footage; it is not part of the bedside stack and never touches its bus.

Event schemas and the bus wrapper live in `shared/nc_shared`; `events.py` holds
the pydantic models and the registry mapping each event class to its stream.
`ARCHITECTURE.md` is generated, so do not edit it by hand. After changing an
event, stream, subscription or declared outside connection, regenerate and
check it from the repository root:

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
`DASHBOARD_PORT` (8444), Redis on 6379. `embodiment` serves HTTPS when
`CERT_FILE` and `CERT_KEY` both exist and plain HTTP otherwise. Plain HTTP works
on `localhost`, but the camera/microphone bridge needs a trusted certificate
from any other device: see [docs/TLS.md](docs/TLS.md).

`./data` is mounted into every container at `/app/data`. Run Python tooling
inside a container, where `nc_shared` is already installed, rather than in a
host virtualenv.

## Validating changes

Every service must be testable with no camera, microphone, Ollama or live
Redis. Run the narrowest relevant tests first, then before handoff:

```sh
ruff format --check . && ruff check . && pytest
```

Per-service tests run in that service's image, for example:

```sh
docker compose build listen
docker compose run --rm --no-deps listen \
  sh -c "pip install -q pytest ruff && pytest -q && ruff check . && ruff format --check ."
```

Record and replay live bus traffic to test without hardware:

```sh
docker compose exec store python -m nc_shared.replay record redis://bus:6379 /app/data/rec.jsonl
docker compose exec store python -m nc_shared.replay play redis://bus:6379 /app/data/rec.jsonl --speed 10
```

Benches and evaluation tools, each with its own README:

| Path | What it checks |
| --- | --- |
| `tests/dialogue_bench` | 50 dialogue scenarios across local models. Unit tests need no Ollama; `python -m dialogue_bench --model gemma4:e4b-mlx --model llama3.1:8b` does. |
| `tests/decision_bench` | Agent decisions against human-reviewed labels (annotation runbook in its `PLAN.md`). |
| `tests/session_replay` | Regressions extracted from a live session export. Commit only reviewed, text-only scenarios; raw exports stay in `data/`. |
| `tests/scene_lab` | Label-free turn-taking, timing and state bugs: `--invariants` on the benches above, or a simulated night on the separate `nightsim` compose project. Take the manual stack down first; both share host Ollama. Baseline in `BASELINE.md`. |
| `tests/perception_bench` | Perception accuracy on recorded clips. |
| `tests/classifier_bench` | Small local decision classifiers; plan in `docs/CLASSIFIER_BENCH.md`. |
| `tools/video_eval` | Perception and agent against recorded bedroom videos; process in `docs/VIDEO_EVAL.md`. Read its "Rendering review videos" section before rendering: most clips need an explicit `--pipeline-tag`, existing videos are skipped without `--force`, and unmapped marker names silently render as `upright`. |
| `tools/llm_speedtest` | One-turn latency (`speedtest.py`) and `run_bench.sh`, which runs the dialogue bench against one MLX server per model. |

The agent and the benches can use an MLX model served on the host by
`mlx_lm.server` (`AGENT_LLM_BACKEND=openai`, or `--backend openai --base-url
http://127.0.0.1:11435`). MLX cannot constrain output to a JSON schema, so the
schema goes in the prompt and replies are validated as strictly as Ollama's.

## Operational gotchas

- `frames_raw` and `audio_in` are what the browser bridge writes; they sit at
  zero with no page attached, which is correct. `capture` gates `frames_raw`
  onto `frames` (2 fps while the room moves, 0.5 fps after 30 s of stillness;
  tune with `CAPTURE_FPS`, `CAPTURE_IDLE_FPS`, `CAPTURE_STATIC_SECONDS`,
  `CAPTURE_MOTION_THRESHOLD`). Both streams are capped at 50, so `XLEN` stops
  there; a length below the cap that never moves is the real fault signal.
- `capture` needs OpenCV only for `CAPTURE_SOURCE=usb` or `rtsp`. Set the source
  and install the `camera` extra together, or the service fails at startup.
- `dashboard` (8444) answers 503 on every route until `DASHBOARD_PASSWORD` is
  set, then uses HTTP Basic auth. Its saves replace YAML under `config/`, and
  nothing hot-reloads: restart `perceive` after a zones save, `agent` after a
  profile or strategy save, and `embodiment` after a strategy save.
- `notify` logs alerts to `docker compose logs notify` when `NTFY_URL` is empty.
  `Notify` requires a `source` field, which is easy to miss when publishing one
  by hand.
- Per-install configuration (`config/zones.yaml`, `phantoms.yaml`, `person.yaml`,
  `strategies.yaml`) is gitignored; only the `*.example.yaml` templates are
  committed.

## Where the detail lives

Service-specific behaviour, calibration and traps are kept next to the code so
they load only when relevant:

- `services/embodiment/AGENTS.md`: speech (Piper), barge-in handling on the
  page, the eyes, the debug overlay and desk-test controls, photos, familiar
  voice.
- `services/perceive/AGENTS.md`: phantom-box filter, bed-zone calibration, gaze.
- `services/listen/AGENTS.md`: VAD/STT gating, Whisper weights, barge-in.
- `docs/TLS.md`: Tailscale certificates for the media bridge.

## Working with Claude Code

Project skills live in `.claude/skills/`; keep this list in sync with it:

- `run-stack`: start the local stack, check the bridge and pages are alive,
  tear it down. Use it before claiming a dashboard or embodiment change works.
- `decision-bench-annotate`: label decision_bench scenarios with the isolated
  Opus annotator, hand flagged ones to the human, apply reviewed labels.
- `demo-creation-video`: polished 30 fps demo clips from the bedroom recordings
  (not the 2 fps `tools/video_eval` review renders).

Tooling that calls Claude (the decision_bench annotator, the scene_lab person
and director) runs `claude -p` on the claude.ai subscription, never the API.
Long scene_lab runs can exhaust the subscription session limit, so check
before starting several in a row.

Delegation in this repo:

- Read the files you know you need directly. Hand a subagent only broad
  sweeps (for example "which services subscribe to `gaze`"), and give it the
  absolute path of your worktree: many `.claude/worktrees/*` checkouts sit
  under the main one, and a search from the wrong root answers about stale
  code.
- A Codex or subagent brief must be self-contained: it does not see this
  conversation. Name the files, the expected change, how to verify it, and
  the privacy rule above. Recordings, frames and labels outside the repo are
  never passed on; only face-blurred, verified frames may be.
- Verify delegated results yourself before reporting them: run the tests,
  read the cited lines, look at the rendered page (the `run-stack` skill).

## Keeping these instructions current

These files are part of the code: a change that makes a line here wrong fixes
that line in the same commit. Stale instructions are worse than missing ones,
because agents act on them with confidence.

- When you change a command, flag, env var, port, path or service behaviour
  that one of these files describes, update the file.
- When something non-obvious costs you time (a trap, a silent failure, a
  required restart), add it to the most specific file that fits (a service's
  `AGENTS.md` before this one) with one line on why.
- Delete lines that are no longer true instead of annotating them.
- Leave out what the code already shows (file listings, signatures), dated
  results (they go in `EXPERIMENTS.md`) and plans (they go in `docs/`).
- Keep this file under about 200 lines; move detail that only one area needs
  into that area's `AGENTS.md` and list it under "Where the detail lives".
- Do not add a `CLAUDE.md` anywhere. Claude Code (2.1.277 and later, default
  "Project instructions" setting) reads `AGENTS.md` only where no `CLAUDE.md`
  exists, so one would silently hide the `AGENTS.md` beside it. Put
  Claude-only notes in "Working with Claude Code" above.
- When a skill's workflow changes (a command, path, flag or output location),
  update its `SKILL.md` in the same change. When you repeat a multi-step
  procedure with traps a second time and no skill covers it, propose a skill
  rather than growing this file. Remove a skill when the tool it drives goes.
- Check what actually loads with `/context`.
