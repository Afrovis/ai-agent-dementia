# scene_lab plan

Status: plan, 2026-09-22. Nothing is built yet. The execution brief is
[HANDOFF.md](HANDOFF.md).

## Goal

Find the agent's decision, timing and turn-taking bugs without a person
standing in the bedroom at night. There are two ways to run it:

- **Fake-live.** The real stack runs in real time against a simulated
  person. A Claude "mind" plays the person, a "body" publishes what the
  camera would report, and a "voice" speaks real audio into the microphone
  stream. A director agent chooses which scenes to try, one after another,
  and pushes harder where the agent breaks.
- **Offline.** A fixed timeline of events is fed through the agent once, and
  the result is checked against known-correct behaviour. `decision_bench`
  (labelled, guideline-based) and `session_replay` (captured regressions)
  already do this. scene_lab adds label-free invariants to both, a timing
  mode that is faithful to LLM latency, and a path that promotes interesting
  fake-live moments into either suite.

## Why the existing benches miss what manual testing finds

Manual testing on 2026-09-22 found agent-decision bugs, timing and cadence
bugs, and turn-taking bugs. In turn-taking bugs the person asks something and
the agent stays silent, or the agent redirects when it should answer.

| Mechanism in the live agent (main, 2026-09-22) | Visible in decision_bench / session_replay? |
| --- | --- |
| LLM `interpret`, `plan` and `compose` run synchronously inside `run_once` and block the loop, so person readings wait behind a slow compose. | No. Both runners give LLM calls zero simulated time. |
| A reply inside the 8 s gap becomes a `PendingSay` and is dropped when it is older than 30 s, or on a session, goal or strategy change, when the person is absent, or on a veto. Drops are only logged as warnings. | Partly. A missing `Say` shows up only if a label or expectation happens to ask for one. |
| "Keep talking to the person" (1a5517d) replies directly only to wanting bed, a time question, a toilet need, or any speech while `ESCALATED`. Every other question goes through the ordinary strategy path, which may redirect. | Only for the three recorded desk scenarios. |
| The agent has no notion of an unanswered utterance. | Nothing checks for it. |
| Silence and barge-in depend on page playback `Activity` events (`started`, `ended`, `interrupted`), and `listen` widens its grace window from them. | No. There is no page in either runner. |
| `perceive` publishes on a confirmed change (3 frames) or a 60 s heartbeat, and the agent confirms a zone after 3 readings. | decision_bench timelines are written by hand and don't follow this cadence. |

Most failures live in timing and in what did *not* happen. Neither is easy
to label per scenario. That is why this plan leans on invariants first.

## Decisions taken (interview, 2026-09-22)

| Question | Decision |
| --- | --- |
| Where the simulated person plugs in | **Hybrid.** The body is `PersonState` on the `person` stream, and `perceive` and `capture` do not run. The voice is real synthesised speech sent as `AudioChunk` through the page's audio path, so VAD, Silero, Whisper and barge-in run for real. `agent`, `embodiment`, `listen`, `notify` (dry run) and `store` are real. |
| What drives the person | Claude through `claude -p`, on the subscription and never the API, isolated like the decision_bench annotator: no tools, an empty working directory, API-key variables stripped. |
| Clock | Real time. The services don't change, and latency is measured rather than modelled. |
| When it runs | On command for a set number of hours (`--hours N`). A scene is capped at 10 minutes, and no new scene starts unless it can finish before the time is up. |
| Answer or redirect | Not decided per question yet. Every question or request the person makes, and what the agent did with it, is flagged for later human review (TT-2), and does not pass or fail. |
| 8 s gap versus a direct answer | A direct answer to the person may break the 8 s gap (owner, 2026-09-22). The agent change that implements this updates rule 3 in the root `HANDOFF.md` in the same PR, as a `decision:` PR. |
| Reply deadline | 5 s from the end of the person's utterance to the start of reply playback, and measured on every reply. |
| Decision records | Accepted: the agent publishes pending-say drops and vetoes as `Activity(kind="decision")`. |
| Bug list | Every run writes one flat list of every issue it found (`bugs.jsonl`, plus a readable `bugs.md`), so runs can be analysed and compared later. |
| Offline format | No new format. Labelled scenarios are decision_bench YAML; captured regressions are session_replay JSONL plus `expect.yaml`. |
| Plan location | `tests/scene_lab/PLAN.md` and `HANDOFF.md`. |

## Design

### 1. One trace, one set of invariants

Every run, whether fake-live, session_replay, decision_bench or a raw
`replay export` of a manual session, is converted to one **trace**: a
time-ordered list of inputs (`PersonState`, `SpeechStarted`, `Utterance`),
agent outputs (`SessionState`, `GoalChanged`, `Say`, `Show`, `Notify`,
`LightCommand`), `Activity` telemetry (LLM durations, playback phases) and
agent decision records (pending-say drops, vetoes). `invariants.py` checks
the trace and needs no labels. Each check has an id, a severity, and a
window that points at the evidence.

Turn-taking (`TT`):

- **TT-1 reply.** Every final `Utterance` in an active phase gets a reply.
  A reply is a `Say` whose playback starts within `R` = 5 s of the end of
  the utterance and is not superseded by a newer utterance. A reply that
  arrives later than 5 s is a `major` issue; no reply at all is `critical`.
  A failure records why: dropped pending say (with reason), vetoed, never
  composed, or phase. The latency of every reply is kept for TM-2.
- **TT-2 answer or redirect (flag only).** For utterances that are questions
  or requests, the evidence judge rates the reply as `answered`,
  `validated_then_redirected`, `ignored_and_redirected` or `no_reply`. This
  never passes or fails. Every case is written to the bug list with severity
  `review`, so the owner can decide question by question later. The agreed
  policy then turns these ratings into pass or fail.
- **TT-3 no talk-over.** No playback starts between `SpeechStarted` and the
  matching `Utterance`, or within 1 s after it.
- **TT-4 barge-in.** An interruptible playback stops within 0.5 s of
  `SpeechStarted`.
- **TT-5 silence gap.** At least 8 s from playback `ended` to the next
  playback `started`, measured on playback and not on publish time. A direct
  reply to the person's utterance is exempt. Only unprompted speech must keep
  the gap.
- **TT-6 no empty room.** No `Say` while the latest reading is `absent`.
- **TT-7 stale reply.** No reply is published after the person has spoken
  again, and no reply answers an utterance older than the latest one.

State machine (`SM`):

- **SM-1 settled means silent.** Once `in_bed` has held for 30 s, there is
  no strategy advance, no non-escalation `Say` and no new `Notify`. This is
  the phase-2 "ladder runs on after back in bed" bug.
- **SM-2 no escalation from calm.** An `attention` or `critical` `Notify`
  needs rule 5, distress, or an exhausted ladder while the person is still
  up.
- **SM-3 no silent session.** While the person is up in `ENGAGED` or
  `ESCALATED`, no silence longer than 60 s. Every session ends in `IDLE` or a
  justified `ESCALATED` by the end of the tail.
- **SM-4 light pairing.** Every hallway `on` has an `off` once the person is
  stably back in bed.
- **SM-5 veto regression.** No published action matches a `veto.py` rule.
  This catches paths that bypass `_vetoed`.

Timing (`TM`, reported, with thresholds as warnings):

- **TM-1 loop lag.** Time from an input's `ts` to its first effect. Flag
  anything over 3 s, and attribute it to the LLM `Activity` that was running
  at the time.
- **TM-2** reply-latency distribution (utterance end to playback start).
- **TM-3** zone-to-goal latency (first `bathroom_path` reading to
  `GoalChanged`).

Wording checks from `dialogue_bench/checks.py` and the decision_bench
evidence judge (`unsupported_claim`, `correction_of_reality`) run on every
`Say` unchanged.

### 2. Offline tier

- **Invariants on existing suites.** decision_bench and session_replay gain
  `--invariants`, which runs the checks on their traces. The first run over
  49 decision_bench scenarios and 3 desk scenarios is the baseline.
- **Latency-faithful time.** Both runners gain `--llm-latency recorded|fixed:<s>`.
  Each interpret, plan and compose advances the simulated clock by the
  recorded `Activity` duration (session_replay captures) or a fixed budget
  (decision_bench). The loop stays blocked for that time, as it is live, so
  TM-1, TT-1 and TT-5 become visible offline and repeatable.
- **Conversation scenarios.** A seventh decision_bench category,
  `conversation`, with about 7 scenarios. Each is a direct question at a
  hard moment: inside the 8 s gap, during agent speech, during a strategy
  change, while `OBSERVING`, two questions in a row, a question while
  walking to the bathroom, and a question the profile cannot answer. They
  are labelled by the existing annotator flow once the answer-or-redirect policy has a
  guideline clause.

### 3. Fake-live tier

A separate compose project, `nightsim`, runs from an override file:

- always-night (`AGENT_NIGHT_START == AGENT_NIGHT_END`),
- `DRY_RUN=true` with `NTFY_URL` empty,
- `LIGHT_ENABLED=false`,
- no `capture` or `perceive`,
- its own ports and Redis,
- scene-specific `person.yaml` and `strategies.yaml` mounted read-only.

It shares host Ollama, so it does not run alongside a manual session.

The simulated person has four parts, in one host process:

- **Body.** Turns intended movements ("stand, walk to the bathroom path over
  8 s") into `PersonState` readings with perceive's real cadence: publish on
  a change after 3 frames at 2 fps, a 60 s heartbeat, and a zone on every
  published reading. Optional noise comes from decision_bench's noisy-variant
  catalogue (flicker, low confidence).
- **Voice.** Synthesises the person's lines with a Piper voice that differs
  from the agent's, at 16 kHz PCM. It streams them in real-time chunks
  inside continuous low-level room noise, so VAD and Silero behave as they
  do live. Mumbling and trailing-off are rendered through the audio, so
  Whisper errors are real errors. Echo leakage of the agent's own speech is
  a later option.
- **Virtual page.** Connects to embodiment's websocket as the bedside page
  does and receives `say` messages. It fetches the WAV and "plays" it for
  its real duration, reports playback `started`, `ended` and `interrupted`
  exactly as `script.js` does, sends the voice's audio up the page's audio
  path, and applies barge-in with the page's rule (stop only when
  `interruptible`). `listen`, `agent` and `store` therefore see what a real
  page would produce. A headless Chromium page with injected audio is a
  later fidelity check of this stand-in.
- **Mind.** Claude plays a persona card: who they are, what they believe
  tonight, a hidden need, hearing, patience and whether they interrupt. It
  is called at decision points: agent playback ended, the person finished
  speaking, N seconds of silence, or the body reached a waypoint. It returns
  a short JSON plan of beats (`say`, `move`, `wait`, `interrupt_if_agent_speaks`,
  `end`). The body and voice keep executing the current plan while the mind
  thinks, and mind latency is logged. Sonnet by default for speed.

The **director** is the "tries different things in sequence" loop. It is a
Claude call (Opus) between scenes. It sees a coverage matrix (category ×
persona × stressor × noise), the report of every scene so far, and the time
left in the run. It writes the next scene card. When a scene fails an
invariant it tries a smaller or harsher variant to confirm and isolate the
bug before moving on. It never edits code or config outside the scene card.
Stressors include a question in the silence gap, a question during
speech, rapid-fire questions, long silence, moving while talking, going
absent mid-conversation, a stated need without the keyword, and repeating
the same question.

A run is started on command: `python -m scene_lab run --hours N`. Every scene
has a hard cap of 10 minutes (`max_scene_s: 600`); a scene that has not
ended by then is stopped, and the stop itself is recorded as an issue
(SM-3). The run starts a new scene only while at least 10 minutes remain, so
it ends within `N` hours plus stack shutdown. Ctrl-C finishes the current
scene's recording and bug list before exiting.

After each scene, the recorder writes
`../data-ai-agent-dementia/analysis/scene-lab/runs/<run-id>/<scene>/`. It
holds `scene.yaml`, `export.jsonl` (`replay export`, no media), `agent.log`,
`mind.jsonl`, `trace.jsonl`, `report.json` and a human-readable `report.md`
timeline.

### 3b. The bug list

Every run, of any kind, writes one bug list at the run's root:
`runs/<run-id>/bugs.jsonl`. That covers a fake-live batch, one scripted
scene, or an offline `--invariants` pass over decision_bench or
session_replay. There is one line per issue, appended as soon as the scene
that produced it is scored, so a crashed run still keeps its list. Every
invariant failure, every TT-2 `review` flag, every wording-check or judge
hit, every scene that hit the 10-minute cap, and every harness error (mind
failure, stack crash) becomes an entry. A harness error is marked
`origin: harness`, so it is never mistaken for an agent bug.

Each entry has a stable `fingerprint` made of the invariant id, the agent
phase, goal and strategy at the time, and the drop reason if there is one.
The same bug seen in many scenes then groups into one row. A readable
`bugs.md` is regenerated after every scene. It groups by fingerprint, sorts
by severity and count, and links every occurrence to its scene report and
timestamp. `python -m scene_lab bugs <run-id> [<run-id> ...]` merges lists
across runs, so you can see which bugs are new, which persist and which have
gone since an earlier run. `runs/index.jsonl` records every run's id, kind,
commit, model, hours and bug counts, for later analysis.

### 4. Promotion: from a live moment to a replayable test

`python -m scene_lab promote <run> --at <t>` takes a flagged moment and
produces:

- a **session_replay scenario** (`extract` on the window, plus an
  `expect.yaml`). Claude drafts it from the failing invariant, a human
  approves it, and it carries recorded interpretations and recorded LLM
  latencies, so it replays deterministically and in faithful time;
- optionally, a **decision_bench scenario** (the input timeline relative to
  scene start). Its checkpoints go through the existing
  `decision-bench-annotate` flow.

Promoted timelines are open-loop: the person no longer reacts. After a fix
changes the agent's behaviour, anything after the first divergence stops
being meaningful. Expectations therefore cover only the window up to the
flagged moment. The scene card is kept too, so the closed-loop scene can be
re-run live as a (non-deterministic) regression.

## Build order

0. **Trace, invariants and bug list.** Build the trace adapters for
   decision_bench, session_replay and `replay export`, the invariants, the
   `--invariants` flag, and the run's `bugs.jsonl` and `bugs.md`. Record pending-say drops and vetoes as `Activity`
   (`kind: decision`) so traces see them without scraping logs. Run over all
   existing scenarios and write the baseline. This alone should show the
   turn-taking bugs.
1. **Latency-faithful offline mode** and the `conversation` scenarios, with
   stub labels.
2. **Scripted fake-live.** The `nightsim` stack and the body, voice and
   virtual page driven by a fixed beat script, with no LLM mind. Replay
   decision_bench timelines live as a first use, then compare against the
   in-process result for the same timeline. The differences are the timing
   bugs.
3. **Claude mind** and persona cards, with 5 hand-written scenes.
4. **Director** loop for `--hours N`, with the per-run bug list and `bugs` merge.
5. **Promotion** into session_replay and decision_bench.
6. Later: headless Chromium page, echo leakage, and a perception tier that
   splices recorded clip frames into `frames_raw` for body movements.

## Open questions

- **Answer or redirect, per question.** Deferred by the owner. TT-2 collects
  the cases as `review` entries in each run's bug list. The owner decides
  from those, and the policy then becomes a guideline clause and a TT-2
  pass rule.
- **Latency under the 5 s target.** Synchronous LLM calls in the loop may
  make 5 s unreachable. Phase 0 and phase 2 measure it, before anyone
  decides whether the agent needs to change.
