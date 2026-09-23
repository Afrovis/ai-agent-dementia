# HANDOFF: building scene_lab

Read [PLAN.md](PLAN.md) first; this is the execution brief. The repository's
own `HANDOFF.md` and `CLAUDE.md` still apply in full. If a decision below
changes, update this file in the same PR.

Last updated: 2026-09-22.

## 1. Fixed decisions

| Area | Decision |
| --- | --- |
| Package | `tests/scene_lab/`, an installable package `scene_lab` with its own `pyproject.toml`, like `decision_bench`. On-demand, no CI. |
| Offline formats | Reuse decision_bench YAML and session_replay JSONL + `expect.yaml`. Do not invent a third scenario format. |
| Fake-live stack | `docker compose -p nightsim -f docker-compose.yml -f tests/scene_lab/compose.sim.yml`. No `capture`, no `perceive`, always-night, `DRY_RUN=true`, empty `NTFY_URL`, `LIGHT_ENABLED=false`, its own published ports and Redis. |
| Person injection | Body: `PersonState` on `person`, with `source="scene_lab"`. Voice and page: the embodiment websocket, exactly as `services/embodiment/embodiment/static/script.js` speaks it. Never publish `Say`, `Show`, `SessionState` or anything else the system under test produces. |
| Claude calls | `claude -p` on the claude.ai login only. No tools, empty temp cwd, `ANTHROPIC_API_KEY` and related variables stripped. Copy the isolation from decision_bench's annotator and judge; do not re-implement it. Mind: `--model sonnet`. Director: `--model opus`. |
| Clock | Fake-live runs in real time. Offline runners use simulated time, with the latency-faithful option from phase 1. |
| Output location | `../data-ai-agent-dementia/analysis/scene-lab/runs/<run-id>/`, with `<run-id>` = `<YYYY-MM-DD>T<HHMM>-<kind>`. Nothing from a run is committed except promoted, text-only scenarios. |
| Run length | `python -m scene_lab run --hours N`. Scenes are capped at 600 s. No new scene starts with under 600 s left. |
| Reply deadline | `R` = 5 s, from utterance end to reply playback start. Measured on every reply. |
| 8 s gap | A direct reply to the person is exempt from the 8 s gap (owner, 2026-09-22). The agent change that implements this is a separate `decision:` PR that also updates rule 3 in the root `HANDOFF.md`. Until that PR merges, TT-5 already applies the exemption, and TT-1 will report late replies. |
| Answer or redirect | Flag only. TT-2 writes `review` entries and never passes or fails. |
| Decision records | The agent publishes `Activity(kind="decision")` for every pending-say drop (with its reason) and every veto (with its rule). This is an `events.py` change: regenerate `ARCHITECTURE.md`. |
| Bug list | Every run, live or offline, writes `bugs.jsonl` and `bugs.md` at its root, appended per scene. |

## 2. Rules

1. scene_lab observes and never fixes. It must not modify agent config,
   strategies or code at run time. A scene may mount its own `person.yaml` and
   `strategies.yaml` into the `nightsim` stack only.
2. Synthetic only. No real person's audio, frames or night ever enter a scene.
   `replay export` runs without `--include-media`. Synthesised audio is not
   written to disk; the voice script in `mind.jsonl` is the record.
3. Persona cards and mind lines are fixtures and follow HANDOFF rule 3's
   spirit for the agent, not the person: the person may say anything a
   confused adult says at night, but cards must not caricature dementia.
4. Invariant checks are deterministic Python. Only TT-2 and the existing
   wording patterns may call the judge.
5. Every invariant has a unit test on a hand-built trace that passes and one
   that fails, with no Redis, Ollama or Claude.
6. Before a fake-live batch, check that nothing else is using Ollama (`ollama
   ps`) and that the manual stack is down; results under contention are not
   comparable and must be marked as such in `report.json`.
7. A number is trusted only after checking that the stack is running the
   commit it claims to run (`git rev-parse HEAD` recorded in every
   `report.json`, images rebuilt for that commit).

## 3. Contracts

### Trace (`scene_lab/trace.py`)

One JSON object per line, time-ordered:

```json
{"t": 12.40, "ts": "2026-09-22T23:01:12.400Z", "kind": "input|output|activity|decision",
 "type": "Utterance", "data": {"text": "what time is it", "confidence": 0.91}}
```

`t` is seconds from the scene or replay start. Adapters: `from_decision_bench(trace)`,
`from_session_replay(timeline)`, `from_export(jsonl)` (bus `replay export`), and
`from_agent_log(lines)` for runs recorded before decision records existed
(for example the desk captures).

### Invariant result

```json
{"id": "TT-1", "severity": "critical|major|minor|info", "passed": false,
 "window": [12.4, 30.0], "evidence": ["Utterance@12.4 'what time is it'",
 "decision@18.1 pending_say dropped: strategy_changed"], "reason": "reply dropped"}
```

Thresholds (`R: 5.0`, settle time, silence limits, `max_scene_s: 600`) come
from one `thresholds.yaml` and are printed in every report.

### Bug entry (`runs/<run-id>/bugs.jsonl`)

```json
{"run": "2026-09-23T2200-live", "scene": "conv-gap-question-01", "t": 18.1,
 "check": "TT-1", "severity": "critical|major|minor|review|info",
 "origin": "agent|harness", "fingerprint": "TT-1|ENGAGED|return_to_bed|orient_time_place|strategy_changed",
 "summary": "reply to 'what time is it' dropped: strategy_changed",
 "evidence": ["..."], "report": "conv-gap-question-01/report.md#t=18.1",
 "commit": "<sha of HEAD>", "model": "gemma4:e4b-mlx"}
```

Append one line as soon as a scene is scored, and never rewrite earlier lines.
`bugs.md` is regenerated from `bugs.jsonl` after every scene, grouped by
fingerprint and sorted by severity, then count. `runs/index.jsonl` gets one
line per run: its id, kind, commit, model, hours, scene count and bug counts
by severity. `python -m scene_lab bugs <run-id>...` merges runs and labels
each fingerprint as `new`, `persisting` or `gone` relative to the earliest
run given.

### Scene card (`scenes/<id>.yaml`)

```yaml
id: conv-gap-question-01
category: conversation          # decision_bench categories + conversation
start: "02:40"
duration_s: 420
profile: profiles/anna.yaml     # person.yaml shape
strategies: default             # or a path
persona:
  summary: "Retired teacher, thinks it is morning, wants to find Tom"
  hidden_need: restroom         # or null
  hearing: normal|poor
  patience: low|medium|high
  interrupts: true
opening:                        # scripted beats before the mind takes over
  - {at: 0, move: {state: sitting_up, zone: bed}}
  - {at: 25, say: "Is it time to get up?"}
stressors: [question_in_gap]
mind: claude                    # or "script" (phase 2: beats only)
noise: {flicker: 0.0, low_confidence: 0.0}
```

### Mind I/O (`scene_lab/mind.py`)

Input: the persona card, the scene so far as plain text, what the person heard
(played agent sentences only; an interrupted sentence is cut at the
interruption), what they see on screen (headline), and elapsed time. Output,
validated by pydantic, invalid output retried once and then logged and ended:

```json
{"beats": [{"wait": 2.0}, {"say": "But where is Tom?", "style": "normal|mumble|trailing"},
 {"move": {"state": "walking", "zone": "bathroom_path", "over_s": 8}},
 {"interrupt_if_agent_speaks": true}], "end_scene": false,
 "note": "wants reassurance about Tom before lying down"}
```

### Body cadence (`scene_lab/body.py`)

Emulates `services/perceive/perceive/main.py`: a published reading only after
the new state has held for 3 frames at 2 fps, a heartbeat of the last reading
every 60 s, and `zone` on every published reading. Read the perceive code for
the exact rules before writing this and cite the lines in a comment.

## 4. Build order and acceptance

Each phase is one PR. Delegate well-scoped pieces to the coder, per project
memory; verification and threshold choices stay with the coordinator.

**Phase 0: trace and invariants.**
- `trace.py`, `invariants.py`, `thresholds.yaml`, `bugs.py` (entry, fingerprint,
  `bugs.md` rendering, `index.jsonl`), unit tests per rule 5.
- `--invariants` on `python -m decision_bench` and `python -m session_replay`.
- Agent: publish pending-say drops and vetoes as `Activity(kind="decision")`.
  Regenerate `ARCHITECTURE.md`.
- Acceptance: a baseline report over all decision_bench scenarios (1 run,
  `gemma4:e4b-mlx`) and the 3 desk scenarios, listing every invariant failure
  with its evidence and a `bugs.jsonl`/`bugs.md` for that run. A text-only
  summary is committed as `tests/scene_lab/BASELINE.md`.

**Phase 1: latency-faithful offline.**
- `--llm-latency recorded|fixed:<s>` in both runners; session_replay reads
  `Activity` durations from its captures.
- About 7 `conversation` decision_bench scenarios, timelines only; labels
  wait for the owner's answer-or-redirect policy. Until then TT-2 flags
  them.
- Acceptance: the baseline re-run with `fixed:2.5` shows which TT and TM
  results change, and the change is explained per scenario.

**Phase 2: scripted fake-live.**
- `compose.sim.yml`, `body.py`, `voice.py` (Piper, a voice other than the
  agent's `en_US-lessac-medium`, 16 kHz, real-time chunks over room noise),
  `page.py` (virtual bedside page), `run.py` (`python -m scene_lab run
  scenes/<id>.yaml`), recorder and `report.md`.
- `python -m scene_lab from-bench <decision_bench id>` turns a timeline into a
  scripted scene.
- Acceptance: three decision_bench scenarios run live and in-process; the
  report diffs the two traces and every difference is explained. A manual
  check with a real browser page open next to the virtual one shows the same
  playback `Activity` sequence for one scene.

**Phase 3: Claude mind.**
- `mind.py`, 5 hand-written persona scenes covering: question in the gap,
  hidden restroom need, interrupting talker, silent wanderer, repeated
  question.
- Acceptance: each scene runs to its end unattended three times; mind latency
  p50/p95 is reported; no scene ends from mind errors.

**Phase 4: director.**
- `director.py`, coverage matrix in `coverage.yaml`, `python -m scene_lab
  run --hours N`. The director sees the time left and the run's bug list so
  far.
- Acceptance: a `--hours 4` run ends within 4 h plus shutdown, with no scene
  over 600 s. Its `bugs.md` can be acted on without opening the traces.
  `scene_lab bugs` over two runs labels fingerprints `new`, `persisting` or
  `gone` correctly.

**Phase 5: promotion.**
- `python -m scene_lab promote <run-dir> --at <t> [--to session_replay|decision_bench]`.
- Acceptance: two promoted failures replay offline, fail on the current
  agent, and would pass on the corrected behaviour described in their
  expectation.

## 5. Where to look

| Need | Place |
| --- | --- |
| In-process agent driving | `tests/session_replay/session_replay/core.py`, `tests/decision_bench/decision_bench/runner.py` |
| Pending say, drops, direct replies | `services/agent/agent/main.py` (`_flush_pending_say`, "keep talking" around 807-878), `session.py` |
| Veto rules | `services/agent/agent/veto.py`, `docs/VETO.md` |
| Page protocol, playback reporting | `services/embodiment/embodiment/static/script.js`, `app.py` (`publish_playback`) |
| Listen grace and barge-in | `services/listen/listen/main.py` (`on_playback`, Silero verifier) |
| Bus capture | `python -m nc_shared.replay export` |
| Claude isolation | decision_bench `annotate` and `--judge` implementation |
| Wording checks | `tests/dialogue_bench/dialogue_bench/checks.py` |
