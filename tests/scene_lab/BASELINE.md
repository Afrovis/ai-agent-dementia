# scene_lab baseline, 2026-09-22/23

The first measurements with scene_lab, and the evidence for each phase's
acceptance in [HANDOFF.md](HANDOFF.md) §4. Text only: the runs themselves
live in `../data-ai-agent-dementia/analysis/scene-lab/runs/` and are not
committed. Run ids below name those folders.

Agent model everywhere: `gemma4:e4b-mlx` on host Ollama, unless a row says
`stub`. Every live `report.json` says `contention: true`: another compose
project (`majority-vote`, a manual stack from another worktree) was up the
whole time. Its agent sat in `IDLE` and made no LLM calls in the windows
checked, so the contention was nominal, but HANDOFF rule 6 applies: treat
the live latencies as indicative until they are repeated with that stack
down.

## What scene_lab finds in the current agent

These hold across the offline baseline and the live runs.

1. **Questions go unanswered.** A question or statement that the
   interpreter does not map to a goal (`unclear`, `looking_for_person`,
   `wants_to_leave`) gets no reply. Often the next thing said is a scheduled
   ladder step that happens to follow, which the first version of TT-1
   counted as a reply. Now that the agent records why it spoke (`said`
   records, see below), 17 of 58 checked utterances in the offline
   baseline have no reply. For 10 the only thing that followed was a
   scheduled ladder step: "Where am I?", "Tom? Are you there?", "Please, I
   just want to go home.", "It really hurts, I can't sleep.", "I'm so
   frightened." Six were never composed at all, among them "Which way is
   the bathroom?", "My chest hurts. I feel strange." and "Where's the loo?
   This isn't right." One ("I can't get warm.") was vetoed. Live,
   a persona asking about Tom four times got four ladder steps and no
   answer.
2. **Replies miss the 5 s deadline live.** Measured from the end of speech
   to the start of playback over 39 replies in one scene
   (`persona-interrupting-talker`, `2026-09-23T0034-live`): p50 5.5 s,
   p95 7.5 s, max 7.9 s; 27 of the 39 were over 5 s. Single replies in
   scripted scenes took 7.0 s and 6.7 s. The
   budget splits into listen's time from end of speech to the published
   transcript (1.2 to 4.1 s; 8.7 s once on a fresh stack, which included
   the whisper model download), one or two LLM calls of 1.5 to 2.5 s each,
   and page fetch plus playback start (about 0.4 s). Offline with `fixed:2.5`, an utterance waits 5 s for
   interpret plus compose (TM-1), or 7.5 s when a second question arrives
   during the first.
3. **The restroom goal does not start from the camera in time.** perceive
   publishes a `PersonState` only on a confirmed state change or its 60 s
   heartbeat, never for a zone change alone
   (`services/perceive/perceive/main.py`, `classify.py` tracker). The agent
   confirms a zone after 3 consecutive readings with no time component
   (`AgentConfig.zone_confirm_readings`, `Session._confirmed_zone`). A
   person who walks steadily to the bathroom path therefore has the zone
   confirmed about two heartbeats later, roughly 140 s. In every live
   restroom scene the goal started only when the person *said* they
   needed the toilet. decision_bench timelines repeat readings by hand and
   hide this; the body model in `body.py` follows perceive's real cadence.
4. **Nothing is said to a person who keeps talking on the restroom path.**
   With the `restroom` goal active, "I really need to use the bathroom."
   and "Left, you said." got no reply (live, `2026-09-23T0033-live`).
5. **Escalation comes fast and then repeats.** A worried, repeating person
   exhausted the strategy ladder and escalated to the caregiver within
   44 to 78 s (`strategies_exhausted`). After that the agent said
   "Someone is on their way, Jean, and you're safe here." about 15 times
   in 6 minutes while the person asked a yes-or-no question about their
   children. In that run one composed reassurance said "Jean, your
   children are fine", a claim the profile does not support, and one said
   "consider the phrase 'Tom is nearby and everything is settled'", the
   caregiver template text leaking into speech.
6. **Long silence while escalated.** SM-3 flags up to 475 s without a word
   after "Someone is coming to help" while the person is still up (19 of
   33 offline SM-3 hits are `ESCALATED/wait_for_caregiver/escalate_phone`,
   8 are `ENGAGED/restroom/path_light`). This may be intended; see the
   questions in the PR.
7. **`escalate_phone` is spoken into an empty room.** TT-6 finds "Someone
   is coming to help" said while the latest reading is `absent` (fall-02,
   fall-05, silent-wander-04, silent-wander-06). The agent exempts this
   strategy on purpose (`_flush_pending_say`).
8. **A person on the floor is directed to the bathroom.** In the one
   director-chosen scene (fall, hip pain, poor hearing;
   `2026-09-23T0050-live/fall-poor-hearing-floor-1`) the fall itself was
   handled at once (`rule5_on_floor`, Notify, "Someone is coming to
   help."). Then "Oh dear silly me, I'm just down here a moment." and "My
   hip's giving me a bit of jip down here." were interpreted as
   `need_restroom`, and the agent said "The restroom is through the bedroom
   door and immediately to the left, Jean." three times, once right after
   "No, no bathroom, I'm on the floor, dear." It later offered "while you
   enjoy a photo of the garden" to "my hip hurts something awful". No veto
   rule forbids `path_light` while the person is `on_floor`, so SM-5 has
   nothing to flag.
9. **Talk-over and barge-in.** With an interrupting persona, playback
   started over the person's speech 12 times in one scene (TT-3), and one
   interruptible sentence stopped 0.64 s after speech began against a
   0.5 s deadline (TT-4).

## Phase 0: trace, invariants and bug list

Runs: `2026-09-23T0044-decision_bench` (56 scenes: 42 scenarios, their
noisy variants and the 7 `conversation` scenarios) and
`2026-09-23T0050-session_replay` (the 3 desk captures, `--llm recorded`),
both at commit `a300dbe`.

| Check | decision_bench, gemma | desk captures |
| --- | --- | --- |
| TT-1 no reply (critical) | 17 of 58 | 4 of 5 |
| TT-2 review | 33 | 6 |
| TT-5 short gap (minor, estimated playback) | 1 | 0 |
| TT-6 Say while absent | 4 | 0 |
| SM-3 silent session gap | 33 | 2 |
| SM-4 light left on at trace end (minor) | 4 | 1 |
| SM-5 published action a veto would deny | 0 of 372 | 0 of 23 |
| WORD states_clock_time (minor) | 34 | 3 |
| SM-1, SM-2, TT-3, TT-4, TT-7, TM-1 | 0 | 0 |

Desk captures: "Hey, can you hear me?" (a ladder step followed), "It is
not yet time to go to sleep.", "And we bet now." and "Okay, I'm back from
the restroom." get no reply.

TT-3 and TT-4 need `SpeechStarted` and real playback, so offline they are
"not applicable"; they only fire live. Playback offline is estimated from
word count (reported as `playback estimated: True`).

### False positives removed while building the baseline

Each was traced to its cause on a real trace before the check changed;
HANDOFF §3 records the settled rules.

- SM-3 "session did not end" fired whenever a bench timeline ended with
  the person still up; it now needs the person back in bed for
  settle + tail.
- TM-1 flagged every deliberate confirmation delay (68 of 75 hits); it is
  now loop lag in the strict sense, time spent blocked behind an LLM call.
- SM-5 first skipped everything (missing context), then flagged
  `orient_time_place` (session_replay forces an always-on night window)
  and `guided_return` after "Can I go back to bed?" (the agent resolves the
  restroom need on `interpreted_wants_bed`). All SM-5 criticals so far were
  such reconstruction errors; there are none now.
- TT-6 flagged a reply to someone who had just spoken while the camera had
  lost them; recent speech now counts as presence, as in the agent.
- `addresses_by_name` is a positive requirement, not a violation, and no
  longer runs on every Say.

### Decision records

The agent publishes `Activity(kind="decision")` for pending-say drops,
vetoes, interpretations (`interpreted`: intent, distress, text) and every
Say (`said`: trigger, direct, deferred time, `reply`). The last two were
added during the build: promotion needs recorded interpretations to replay
deterministically, and TT-1 needs `reply` to tell an answer from a ladder
step. They are telemetry only; the agent says exactly what it said before.

## Phase 1: latency-faithful offline

Deterministic pair, stub LLM (`2026-09-23T0050-decision_bench` against
`-2`): with `--llm-latency fixed:2.5` everything is identical except
TM-1, which goes from 0 to 8. Each hit is an utterance whose effect waits
for interpret plus compose (5.0 s): conversation-01 and -03, distress-pain-01,
fall-01 and its low-confidence variant, fall-03, fall-06, and conversation-05,
where the second of two questions 3 s apart waits 7.5 s behind the first.
Everything else only moves in time by the call durations.

gemma pair (`2026-09-23T0044` against `2026-09-23T0047`): TM-1 0 → 11, TT-1
critical 17 → 12 plus 3 major, SM-3 33 → 31. The TM-1 change is the
latency. The TT-1 and SM-3 changes mix latency with gemma's own
run-to-run variation (the same scenario's interpretation differs between
runs, e.g. `distress-pain-01` against its `-typo` variant), which is why
the stub pair is the attribution.

Desk captures carry no recorded LLM durations; `--llm-latency recorded`
falls back to `fixed:2.5` with a warning and adds one TM-1 hit
(desk-time-question at 237 s). Captures made from now on keep the timing
rows (`session_replay extract`).

The 7 `conversation` scenarios are timeline-only with stub checkpoints and
stay unlabelled until the answer-or-redirect policy exists.

## Phase 2: scripted fake-live

Three decision_bench scenarios, converted with `scene_lab from-bench`, ran
live on the `nightsim` stack and in-process with the same model, then
`scene_lab diff`:

| Scene | Live run | Matched | Shifted | Only one side | Explanation |
| --- | --- | ---: | ---: | ---: | --- |
| restroom-01 | `2026-09-22T2338-live` | 9 | 7 | 0 | In-process the utterance lands at its scripted start (12.0 s). Live the line ends at 13.1 s, listen publishes at 17.1 s (4.1 s) and interpret takes 1.8 s, so the goal change, light and Say move to 19.1 s. |
| disorientation-01 | `2026-09-22T2348-live` | 4 | 17 | 0 | One cascade: the first utterance arrives about 5.6 s later live, and the ladder's dwell timer restarts at the utterance, so every later step moves by about 6 s. |
| conversation-01 | `2026-09-22T2359-live` | 19 | 1 | 0 | The time answer waits for transcription (2.8 s) and interpret (1.5 s). |

Every decision appears on both sides; every difference is time, and each
is explained by transcription and LLM latency. The manual check with a
real browser page next to the virtual one is not done (it needs a person
at a browser).

## Phase 3: Claude mind

The five persona scenes ran live, each once, after two harness fixes found
in the first runs:

- The mind spoke the device's lines ("Tom's alright. He's safe at home...")
  because the prompt labelled device speech "Agent heard:". It is now
  "The bedside device said to you:", with an explicit role and a person's
  pace.
- Finishing a line re-planned the mind every 2 to 3 s, a nonstop monologue.
  Only device speech and silence re-plan now.

After the fixes: 68 mind calls, 0 failures, latency p50 5.2 s, p95 7.9 s,
max 12.0 s (sonnet via `claude -p`). The mind stays in role and paces
itself; the interrupting persona calmed down over 6 minutes. Scenes that
ended early did so by the mind's own `end_scene` (hidden-restroom at 102 s
once Jean reached the bathroom).

Not met yet: HANDOFF asks for each scene to run three times unattended.
Each ran once after the fixes (repeated-question also before them).

## Phase 4: director

The first `run --hours 1` (`2026-09-23T0020-live`) could not direct at all:
every director call failed on a JSON schema with unresolvable `$defs`, and
every scene came from the least-covered-card fallback. That run was
stopped with Ctrl-C, which recorded the scene in progress, `bugs.md` and
one `index.jsonl` line, as intended. After the fix a director call takes
about 20 s and picks an uncovered cell (for example distress_pain ×
poor-hearing × need_without_keyword).

The second `run --hours 1` (`2026-09-23T0050-live`, commit `a300dbe`)
started at 00:50:31 and ended at 01:40:32: within the hour, 28 scenes, no
scene near the 600 s cap. It is only a partial test of the director. The
first scene was chosen by Opus (the fall scene in finding 8). From the
second call on, every `claude -p` call returned HTTP 429 "You've hit your
session limit", so the director fell back to existing cards and each
scene's mind failed at once: 28 director and 23 mind harness entries, all
`origin: harness`. The run's `bugs.md` still grouped and ranked everything
correctly, but it measured the harness more than the agent. A batch now
stops at the first usage-limit failure and records one harness entry with
the reset time, rather than running scenes whose mind cannot start.

A third `run --hours 1` (`2026-09-23T0552-live`, commit `580086d`, after the
session limit reset) exercised the director fully: 05:52 to 06:42, 10
scenes, every one chosen by Opus, no harness errors, no cap stops. One card
failed validation (a beat with both `say` and `wait`) and the retry fixed
it. The director followed the plan's confirm-and-isolate loop on its own:
- Three fall scenes all failed in `ESCALATED/wait_for_caregiver`. It then
  moved to a pain scene that reaches the same state without a fall.
- Two restroom scenes with different personas both hit
  `TT-1|ENGAGED|restroom|path_light` (finding 4). It then tried a third,
  near-silent persona to find out whether the person has to speak at all.

The run's live bug list had 25 criticals. Re-checked with
`scene_lab rescore` after two check corrections, it has 18, all TT-1 "no
reply": 15 on the restroom path and 3 in the `comfort` goal. The
corrections:
- SM-1 had flagged replies to a person talking from bed. Speaking from
  bed now restarts the settle clock, as in the agent's NICE-05 rule.
- An SM-5 critical turned out to be the blocked loop again. She lay down
  at 14.5 s while an untracked `plan` LLM call held the loop, and the
  greeting at 16.5 s was decided before the agent read that reading. A
  denial that rests only on a reading under 3 s old is now a minor
  "possibly unseen state". Replaying the captured window in session_replay
  confirmed the order of events. `plan` calls publish no Activity, so
  TM-1 cannot see these blocks yet.

`scene_lab bugs 2026-09-22T2348-live 2026-09-23T0020-live` labels the
600 s cap stop fixed in between as `gone`, the director failures as `new`
and the unanswered utterance as `persisting`.

Not met yet: the `--hours 4` acceptance run. On the subscription, one
session window did not cover an hour of Sonnet mind calls (about one every
6 s of scene time with a talkative persona) plus Opus director calls, on
top of this build session's own use.

## Phase 5: promotion

Two live failures promoted with `scene_lab promote` are in
[promoted/](promoted/README.md). Both replay offline with recorded
interpretations and latencies, fail on the current agent, and pass once
the agent replies within 5 s:

- "I don't like being here on my own." (`unclear`, distress 1): only the
  scheduled `soft_greeting` follows.
- "Where are my car keys? They'll be waiting." (`unclear`): nothing
  answers it.

## Harness bugs found by running it

Recorded so the next reader trusts the numbers: flushing Redis under running
services deleted consumer groups (now stop, flush, start); scripted scenes
lost their beats at the first decision point; the `activity` stream's 200
cap dropped early playback reports and decision records from end-of-scene
exports (now a live `StreamTap`); a scripted end at exactly 600 s counted as
a cap stop; the live-vs-in-process diff paired by time rather than content.

## Triage after the director hours, 2026-09-23

Both director hours re-checked with `scene_lab rescore` at the commit that
introduced this section. `0050-live` is mostly harness: 27 of its 28 scenes
are fallback replays of 8 cards with no mind, and 50 of its 76 majors are
the session-limit harness entries. `0552-live` is the measurement to use:
18 criticals, all TT-1 "no reply".

Fixed on branch `scene-lab-fixes`:

| finding | change |
| --- | --- |
| 5, repeats | While escalated, two reassurances per escalation; after that only a direct question or distress 3 is answered, at most once a minute. A repeated sentence is swapped for the approved phrasing said longest ago. Deliberate silence publishes a `no_reply` decision; TT-1 reports it as info, or as review when the utterance was a question. |
| 4, restroom path | The first plain progress remark on a restroom trip gets "Good, take your time."; later ones stay quiet; a question about the way gets the directions again, without a second sentence in the same tick. Distress on the path no longer gets `validate_and_redirect` ("let's rest now"), which contradicted TOIL-01. |
| pain (new) | "Oh my hip really hurts. I need someone, please." got "Hello Jean, it's night-time." (a `plan` pick), and "Is someone coming? It really does hurt." got nothing. One clear pain statement (intent `pain`, distress ≥ 2) now escalates at once (`pain_reported`, one attention Notify) and says "I'm sorry it hurts, Jean; I'm letting someone know now." Milder pain gets one comfort line with no promise of help. |
| 8, floor | Veto `no_directions_from_floor` (FALL-01) denies `path_light` while the person is `on_floor`; while escalated the reply is a reassurance instead, and a `need_restroom` reading on the floor no longer moves the goal to `restroom`. |

Replayed offline with recorded interpretations and latencies
(`scene_lab promote` then `session_replay run --invariants`), windows from
`0552-live`:

| scene | main | this branch |
| --- | --- | --- |
| `fall-poor-hearing-1`, 45 to 425 s | 33 Says, 32 of them "Someone is on their way, Jean, and you're safe here." | 7 Says, six phrasings rotated; TT-1: 7 review (paced answers to repeated questions), no critical |
| `distress_pain-teacher-silence-4`, 0 to 193 s | "Hello Jean, it's night-time." to the first pain statement, escalation on the second, then the same reassurance to every remark | pain acknowledged and escalated on the first statement, two reassurances, then quiet; TT-1 clean |
| `restroom-wanderer-silentpath-isolate-10`, 35 to 240 s | no reply to any progress remark | one "Good, Jean, take your time.", then `no_reply`; the 2 TT-1 criticals left are remarks made before the restroom goal started (open item 1) |

Also fixed on this branch, after the owner's decisions on the first triage:

| item | change |
| --- | --- |
| unmapped intents (finding 1) | `looking_for_person` / `wants_to_leave` get a composed `validate_and_redirect` (name the feeling, redirect gently, never correct; `acknowledge_feeling` in bed). `unclear` gets "Is there something you need, Jean?" once per session. While escalated, "Tom, is that you?" gets "I've let Tom know, Jean, and help is on the way." Replaying `0050-live` conversation scenes: the flagged question is answered in 3.3 s instead of a greeting 30 s later. |
| silence while escalated (finding 6) | A check-in after 120 s without speech while escalated, if the person is present and not settled. SM-3 allows 150 s for a gap that starts in `ESCALATED`. |
| clock time in speech | Spoken time rotates night phrases by hour ("the middle of the night", "late in the evening", "nearly morning and still dark"); the screen keeps the hour (AA-02). Composed speech with a clock time falls back to the template. |

Still open, in suggested order:

1. **Latency (finding 2).** Still 5 to 7.7 s live. Listen's transcript lag
   varies most (1.3 to 4.8 s in these runs); interpret and compose are two
   sequential calls, and the new `validate_and_redirect` replies add a
   compose call. Re-measure without the contending stack first.
2. **Composed replies make claims (finding 5).** Live examples: "Tom is
   nearby and everything is settled", "while you enjoy a photo of the
   garden" with no photo on screen, "your children are fine".
   `validate_composition` catches clock times, "but" and caregiver-arrival
   phrases, not reality corrections or other claims; those rest on the
   prompt. A TT-2 judge pass over a live run would show how often.
3. **`validate_and_redirect` fallback on the restroom path.** If the
   composition for "Where is Tom?" on the restroom goal is rejected, the
   caregiver template ("let's rest now and talk more in the morning") points
   away from the toilet (TOIL-01).
4. **Talk-over (finding 9)** and **restroom goal from the camera
   (finding 3)**: unchanged, need live or perceive work.
5. **Harness.** Both director hours ran with host Ollama's MLX prefix cache
   pinned at 16.3 GiB (from 2026-09-22 23:34), so the host was swapping and
   part of the measured latency is that, not the agent. `scene_lab` now resets
   the model before each scene; re-measure latency on a fresh run.
   One director hour does not fit a subscription session window.
   `compose.sim.yml` sets no `TZ`, so the sim agent's night window and
   screen clock use UTC. `plan` calls publish no Activity, so a blocked loop
   is invisible to TM-1 (the SM-5 minor in `0552-live`). `session_replay
   --llm live` against a busy host Ollama times out every call and silently
   replays with no model.
