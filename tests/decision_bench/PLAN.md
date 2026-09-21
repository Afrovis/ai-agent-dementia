# decision_bench plan

Status: planned, 2026-09-21. Nothing is built yet. The user-facing
description is in [README.md](README.md).

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
| Annotator | Not decided yet. Leading option: a Claude Code skill that runs an Opus agent on the subscription. |
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
  pattern cannot be caught with string rules, it is flagged for human review
  instead of being guessed at by a judge model.
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

1. **Guideline pack, schema and 5 pilot scenarios** (one per category
   except `false_alarm`, which gets two). The human reviews the clause text.
2. **Harness and scoring**, with unit tests on a stubbed LLM, plus a
   report on one local model.
3. **Annotator.** Pick the shell, label the pilot, run the human review,
   and adjust the prompt and schema based on the disagreements.
4. **Scale up** to about 30 reviewed scenarios and about 8 noisy variants,
   then run the first comparison across models.

## Open questions

- Which annotator shell to use: a Claude Code skill running an Opus agent,
  or something else.
- Whether `familiar_voice` belongs in `acceptable` sets when the profile has
  no consented clip. Probably the profile fixture should state whether a
  clip exists.
- How strict the `window` for "stay quiet" on false alarms should be. That
  depends on the configured `OBSERVING` wait, which labels must not copy.
