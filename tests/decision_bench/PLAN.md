# decision_bench plan

Status: phase 2 complete, 2026-09-21. The real-agent replay harness, scoring,
reports, CLI, deterministic stub and tests exist for the seven unlabelled
pilots. Labels remain phase 3 work, and the clause text awaits human review.
The user-facing description is in [README.md](README.md).

## Goal

A small, on-demand benchmark that feeds the agent scripted position and
speech timelines and checks whether it makes the right redirect decisions.
The labels rest on evidence-based guidelines. A model drafts them and a
human settles them.

## Decisions taken

| Question | Decision |
| --- | --- |
| What is scored | Decisions plus key wording. The real LLM is in the loop, and `dialogue_bench` checks run on every `Say`. |
| Scenario source | Hand-authored YAML timelines. A model drafts variations and a human prunes them. Recorded-video tracks are out of scope for now. |
| Labels | Per checkpoint: an `acceptable` set, a `must_not` set and an optional `escalate_by` deadline. |
| Size | About 30 scenarios in 6 categories (about 5 each), plus about 8 noisy variants. |
| Noise | Some noisy variants of clean parents: state flicker, low confidence, speech-to-text typos, missed speech. |
| Evidence | NICE NG97, Alzheimer's Association guidance, validation therapy and person-centred care, DICE, and nighttime falls and toileting evidence. |
| Disagreements | The model annotates first with citations. The human reviews every label and makes the final call, and disagreements are logged. |
| Annotator | Headless `claude -p --model opus` with no tools, no settings and an empty temp directory as cwd, run by `python -m decision_bench annotate` and wrapped by the project skill `decision-bench-annotate`. The isolation is enforced, not requested: the annotator cannot read the repository. |
| Human review | A self-contained YAML file per scenario (`annotations/review/<id>.yaml`) holding the timeline, questions, the annotator's labels and doubts. `python -m decision_bench apply` merges it and logs disagreements. |
| CI | None. On-demand only, like `dialogue_bench`. |

## What already exists (research, 2026-09-21)

- **`tests/dialogue_bench`** runs one utterance per scenario
  (`fixtures/scenarios.yaml`). It scores intent accuracy and five
  deterministic wording checks in `checks.py` (`conjunction_but`,
  `avoid_terms`, `states_clock_time`, `invents_proper_noun`,
  `addresses_by_name`), with per-scenario `must` and `must_not` overrides.
  It uses the same `--backend` and `--model` switches this bench will reuse.
- **The agent's inputs**, from `shared/nc_shared/events.py`:
  - `PersonState`: `state` is one of in_bed, sitting_up, standing,
    walking, on_floor or absent. It also carries `zone` (bed, door,
    bathroom_path or other), `confidence` and `scene_note`.
  - `Utterance`: `text`, `confidence` and `duration_s`.
  - `SpeechStarted`: marks the start of speech.
- **The state machine**, in `agent/rules.py` and `agent/session.py`:
  - Phases: `IDLE → OBSERVING → ENGAGED → COOLDOWN → IDLE`. `OBSERVING`
    returns to `IDLE` if the person is back in bed early. `ESCALATED` can be
    reached from `OBSERVING` or `ENGAGED`.
  - Triggers: getting up at night → `OBSERVING`. Speech → `ENGAGED`. The
    `need_restroom` intent → restroom goal. Pain → comfort goal. Two
    consecutive distress readings → escalate. Rule 5: on the floor, or
    absent past the timeout → `ESCALATED` from any phase.
- **Where decisions are made.** Mostly in deterministic code: zone
  hysteresis, the Rule 5 timer, and strategy dwell and advance in
  `agent/strategies.py`. The LLM decides the intent reading and composes
  the text; every `Say` is validated before it is published.
- **Strategies**, from `config/strategies.example.yaml`: `ambient_orient`,
  `soft_greeting`, `orient_time_place`, `validate_and_redirect`,
  `guided_return`, `familiar_voice`, `path_light` and `escalate_phone`. The
  first available one in the configured order runs first. The engine moves
  to the next when dwell time passes with no progress, and escalates when
  all are used up.
- **Hook for testing.** Every `Session` method takes `now: datetime`, and
  `services/agent/tests/test_session.py` already drives it directly. The
  harness can replay a timeline without Redis or a wall clock.
- **Guidelines in the repo today.** Only design principles: PLAN.md §3
  ("calm over clever", "validate the feeling, then redirect") and HANDOFF.md
  rule 3 (one sentence then at least 8 s silence; no "no", "you can't",
  "you're wrong", or memory-testing questions). No clinical source is cited
  anywhere. That gap is why the guideline pack comes first.

## Design

### Harness

- `decision_bench/runner.py` builds a `Session` with the scenario's
  profile and strategy config. It walks the timeline in time order. Every
  event calls the matching `Session` method with `now = start + t`. The
  clock also ticks every second between events so that dwell and timeout
  logic fires as it would live.
- Utterances go through the same interpret and compose path that
  `agent.main` uses, against the chosen backend.
- The runner records a trace of every `Transition`, `Say`, `Show`,
  `Notify`, `GoalChanged` and `LightCommand`, each with its time.

### Scoring

- A checkpoint looks at the trace inside its `window`. It passes if at least
  one `acceptable` action occurred and no `must_not` action did.
- `must_not` hits and missed `escalate_by` deadlines count as critical.
- Named wording patterns such as `correction_of_reality` are deterministic
  checks added next to the ones in `dialogue_bench/checks.py`. When a
  pattern cannot be caught with string rules, an isolated Claude evidence
  judge (`--judge`) decides it per sentence from the profile and that
  sentence's context, citing its evidence. A human verdict overrides it, and
  a disagreement is reported. Phase 3 changed this at the reviewer's
  request; the first design kept these patterns human-only.
- The report gives, per model, the pass rate by category and by clean
  versus noisy input, a count of critical violations with scenario ids, the
  escalation latency distribution, and the wording-check failures.

### Guideline pack (`guidelines.md`)

- Numbered clauses (`NICE-nn`, `AA-nn`, `VAL-nn`, `PCC-nn`, `DICE-nn`,
  `FALL-nn`, `TOIL-nn`). Each is a short paraphrase with a source link and
  a section reference. The clause text is checked against the source by a
  human before any label cites it.
- Each clause says what it implies for the agent's actions (for example,
  "do not correct a disoriented belief" means `must_not: say
  correction_of_reality`).
- Explicitly out of scope: numeric thresholds. Those are
  `threshold_source: caregiver`.

### Annotation workflow

1. **Draft scenarios.** The model gets the category table and the timeline
   schema, and proposes several candidates per category. The human keeps
   about 5 per category.
2. **Model labels.** The annotator sees the guideline pack, the action list
   and the scenario, and never the strategy config or agent code. It writes
   `annotations/model/<id>.yaml` with a rationale and citations for each
   checkpoint.
3. **Human review.** The human accepts or edits each label. Accepted labels
   go into `fixtures/scenarios/<id>.yaml`. Each edit is appended to
   `annotations/disagreements.yaml` as `{scenario, checkpoint, model, human,
   reason}`.
4. **Noisy variants.** Derived mechanically from reviewed parents, with the
   parent's labels inherited unchanged.

The annotator's shell (skill, subagent or API script) is chosen before
step 2. The inputs and outputs above stay the same whichever is used.

## Layout

```
tests/decision_bench/
  README.md  PLAN.md  guidelines.md  pyproject.toml
  decision_bench/  __main__.py  schema.py  runner.py  scoring.py  checks.py  report.py
  fixtures/scenarios/<id>.yaml
  annotations/model/<id>.yaml  annotations/disagreements.yaml
  tests/            # schema, runner (stub LLM), scoring
```

## Build order

1. **Guideline pack, schema and 7 pilot scenarios** (one per category
   except `false_alarm`, which gets two). The human reviews the clause text.
   The pilots carry timelines and checkpoint questions only: labelling them
   is step 3, so the first labels come from the annotator and not from
   someone who has read the agent code.
2. ✅ **Harness and scoring**, with unit tests on a stubbed LLM and reports
   suitable for local-model comparison. Phase 2 made `memory_question` and
   `blunt_refusal` deterministic checks; `correction_of_reality` and
   `infantilising` remain human-review patterns.
3. ✅ **Annotator.** Pick the shell, label the pilot, run the human review,
   and adjust the prompt and schema based on the disagreements. See
   "Phase 3" below.
4. **Scale up** to about 30 reviewed scenarios and about 8 noisy variants,
   then run the first comparison across models.

## Phase 2 run on one local model (2026-09-21)

`gemma4:e4b-mlx` on Ollama, all seven pilots, default `AgentConfig` and
`config/strategies.example.yaml`. All 15 checkpoints are unlabelled, so
there is no pass rate yet; the JSON report is in
`../data-ai-agent-dementia/analysis/decision-bench/2026-09-21-gemma4-pilot.json`.
What the traces show, for the annotator and for the agent itself:

- **fall-01** escalates on the first `on_floor` reading with a critical
  `Notify`. **false-alarm-01** goes `OBSERVING → IDLE` with no speech.
- **restroom-01** handles the toilet request well (goal `restroom`,
  `path_light`), but once the person is back `in_bed` at 350 s the
  `guided_return` dwell runs out at 380 s, before `in_bed_stable` (120 s)
  ends the session, so the agent reports "strategies exhausted" and sends an
  `attention` notify although nothing is wrong. The same happens in
  **false-alarm-02**, which runs the whole ladder and notifies.
- **Every night-time `soft_greeting` and `orient_time_place` states an exact
  clock time** ("it's 2 o'clock at night"): 8 `states_clock_time` wording
  failures in one run. AA-02 supports saying it is night, not the clock
  time, so labels citing it will mark these as violations. That is a
  template question (`time_words`), not a model one.
- `validate_and_redirect` produced one "but" (`conjunction_but`).
- **Zone-driven goal switching is almost unreachable live.** The agent
  confirms a zone after 3 consecutive `PersonState` events, but `perceive`
  publishes only on a state change or its 60 s heartbeat. Walking along the
  bathroom path therefore takes minutes to confirm, and every goal switch in
  the pilots came from speech. Silent restroom scenarios will expose this.
- Results vary from run to run: the model sometimes returns output that
  fails validation (`None`; compose falls back to the template) and the
  planner's strategy proposals differ, so a comparison needs several runs
  per model. A cold model timed out on its first calls, which is why the
  CLI now makes one warm-up call.

## Open questions

- ~~Which annotator shell to use.~~ Settled in phase 3: headless
  `claude -p` with no tools.
- ~~Whether `familiar_voice` belongs in `acceptable` sets when the profile
  has no consented clip.~~ Settled in phase 1: a scenario sets
  `voice_clip: true` when a clip exists, and the schema rejects
  `familiar_voice` in `acceptable` otherwise.
- How strict the `window` for "stay quiet" on false alarms should be. That
  depends on the configured `OBSERVING` wait, which labels must not copy.

## Phase 3: annotator and first labelled run (2026-09-21)

**Annotator.** `python -m decision_bench annotate` ran `claude-opus-5`
(effort high) once per pilot: 7 scenarios, 15 checkpoints, every answer
valid on the first attempt, $1.36 total. The drafts, with their prompt,
guideline and scenario hashes, are in `annotations/model/`.

**Human review.** The reviewer accepted all 15 checkpoints unchanged, so
`annotations/disagreements.yaml` is empty. There is nothing to learn from
disagreements yet, and the prompt and schema are unchanged. The annotator's
own `uncertain` and `scenario_notes` fields raised these points, which are
still open:

- `escalate_by` is met by a notification at any level, so an on-time `info`
  for someone on the floor is only an ordinary miss (fall-01). A level on
  the deadline (for example `escalate_level: critical`) would close that.
- No citable clause forbids `guided_return` while the person is on the
  floor. That gap is in the guideline pack, not the labels.
- NICE-06 (pain) is unchecked, so the pain labels rest on DICE-01 and
  NICE-01.
- The `guidelines.md` status line still says no clause has been checked.
  Three annotator runs flagged it.

**First scored run.** `gemma4:e4b-mlx` against the labelled pilots
(`../data-ai-agent-dementia/analysis/decision-bench/2026-09-21-gemma4-labelled-run1.json`):
4 pass, 4 critical, 7 review, which is 27% overall. Two findings concern
the bench, not the agent:

- **Escalation already sent before `trigger` reads as missed.** In
  disorientation-01 (`stays-at-door`, trigger 200) and silent-wander-01
  (`caregiver`, trigger 360), the agent had already escalated with an
  `attention` notify in the previous window and stayed `ESCALATED`. Scoring
  only counts a `Notify` at or after `trigger`, so both are reported as
  critical "escalation missed".
- **Review-only patterns swamp the pass rate.** Any checkpoint with a
  `must_not: say correction_of_reality` or `infantilising` and at least one
  `Say` becomes `review`, which is 7 of 15. None of the texts in this run
  looks like either pattern, but only a human can say so.

Both are fixed. **Escalation:** a `Notify` sent earlier in the same episode
now meets an `escalate_by` deadline with latency 0, provided the agent is
still `ESCALATED` at the trigger. **Review patterns:** the human's verdict
on each distinct sentence goes in `annotations/say_verdicts.yaml`, and
scoring reuses it. `--collect-verdicts` appends the sentences that still
need one, and a sentence with no verdict stays `review`. Neither change
touched the reviewed labels. Two more runs (`...-run2.json`, `...-run3.json`)
scored 40% with 6 review checkpoints before any verdicts were filled in. The
two false "escalation missed" criticals are gone, and one run produced a
real `conjunction_but` in disorientation-01 ("I know you worry about the
children, but Tom is here").

The agent failures are real and match phase 2: false-alarm-02 runs the
whole ladder on a contented person and notifies; restroom-01 goes back to
`guided_return` and escalates after the person is back in bed; every
greeting states the clock time (9 `states_clock_time`). One reply, "Tom is
here now, so let's settle down", asserts something the agent cannot know.
No wording pattern covers that yet.

**Hallucination screening (review feedback).** The reviewer asked to screen
for invented facts, such as giving directions when no restroom location is
set. There are two new wording patterns, and both run on every `Say` like
the other wording checks, so the reviewed labels are unchanged:

- `invents_directions` (check): direction or place language is invented
  unless all its content words are in the profile's `restroom_location`;
  with no location set, any direction is invented. The agent itself does
  this: `agent/strategies.py` falls back to "The restroom is just outside
  the bedroom" when `restroom_location` is empty.
- `unsupported_claim` (review): a fact the agent cannot know that is not in
  the profile or input ("Tom is here"). It is settled by the human verdict
  file, and a `true` verdict on any review pattern now counts as a wording
  failure on every run.

No pilot scenario leaves `restroom_location` empty, so the direction check
has no live exposure yet. A restroom scenario with the location unset
belongs in phase 4. With the first verdicts filled in, run 4 scored 67%.
Distress-pain-01 also varied in that run: `guided_return` during ongoing
pain, and escalation 165 s after the trigger against a 60 s deadline.

**Evidence judge (review feedback).** Rating every sentence by hand without
the context behind it did not work well. The reviewer marked the
restroom-location sentence as unsupported although it repeats the profile
word for word, and could not see whether "12 o'clock" was the real time. So
`--judge` runs an isolated `claude -p` (no tools, subscription only, with
API-key variables stripped) over each new sentence. It sees the profile and
that sentence's context: the time words, the person's last words, the
camera reading, and whether a caregiver has been notified. Its verdicts sit
under `judge:` in `say_verdicts.yaml`. A human value wins, and every
disagreement is printed. In run 5 it checked 14 sentences, flagged none,
and its evidence matched both human corrections. With no checkpoint left in
review, gemma4 scored 80%. The remaining criticals are false-alarm-02,
restroom-01 back-in-bed and one "but".

All Claude calls here (annotator and judge) go through the Claude Code CLI
on the claude.ai login. `list_price_usd` in the annotation drafts is Claude
Code's list-price estimate of subscription usage, not a charge.
