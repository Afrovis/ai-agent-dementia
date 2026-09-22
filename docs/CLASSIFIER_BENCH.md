# Classifier benchmark plan

A plan for benchmarking small, local, typed-decision classifiers as the agent's decision
layer, and for producing the training data they need.

Written 2026-09-22, after the local-annotator experiment in
`tests/decision_bench/VOTE_HANDOFF.md` rejected gemma4 as a second annotator. That
experiment produced the finding this plan is built on.

## Why

gemma4 placed an action opposite to Opus on 3 of 397 actions (0.8%). It does not disagree
with Opus about what is safe; it fails to *enumerate* the full set of acceptable and
forbidden actions, omitting 50.4% of what Opus labels. The Opus-vs-Opus baseline shows the
same shape from the other side: 74.0% agreement, zero inversions, and 26.0% of actions
placed by one run and left unmentioned by the other.

So a large part of what `decision_bench` currently measures is enumeration noise, not
judgment. Set generation is the hard part, and it is a part no model does reliably.

Asking instead *"is this specific action acceptable in this specific state?"* is a
classification question, and classification is what the current generation of small typed
decision models does in milliseconds on local hardware. That reframing is the subject of
this plan.

## What we are benchmarking

A **candidate classifier** answers typed questions about a bedside state. Three primitives,
following the Jev and Laya vocabulary:

| primitive | example question | output |
| --- | --- | --- |
| `bool` | "In this state, is `{strategy: path_light}` acceptable?" | probability |
| `choice` | "Which strategy fits this state best?" (≤32 options) | option + probabilities |
| `score` | "How urgent is this state, 0-4?" | ordinal |

We benchmark, on the same question set: agreement with the reference labels, **calibration**,
a cost-weighted error score, and **latency**. Nothing about generated text.

### Candidates

| backend | what it is | role |
| --- | --- | --- |
| `random`, `majority` | trivial baselines | floor; a score is meaningless without them |
| `rules` | the existing state machine's logic, expressed as answers | the bar to beat — if rules win, use rules |
| `opus` | `claude -p`, as in `annotate.py` | reference ceiling, and label source |
| `laya-torch` | `convaiinnovations/laya`, ModernBERT-large 421M, English, torch MPS | v1 local candidate |
| `laya-coreml` | `FluidInference/laya-coreml`, Core ML on the ANE | v2 local candidate |
| `jev` | typesafe.ai API, 70-500 ms | comparison only, never in the live path |

English first, as decided 2026-09-22. Note the tension to resolve in Phase 0: the English
checkpoint is ModernBERT-large (421M, 512 tokens), while the checkpoint FluidInference
converted to Core ML is the 322M multilingual mmBERT one. The ANE path may therefore need a
conversion of the English checkpoint through `FluidInference/mobius`, which is unproven.

`jev` is API-only with no open weights, so it is disqualified from the bedside: the live path
must not send a vulnerable person's night-time speech to a third party, and must not depend
on a home network at 3am. It stays in the benchmark as a quality reference on **synthetic
scenarios only** — enforced in code, see Rules below.

## Non-goals

- Replacing `decision_bench`. That stays the end-to-end agent benchmark. This is a
  unit-level benchmark of the decision layer alone.
- Generating speech. Wording stays with the local LLM and Piper.
- Classifying escalation *timing*. Laya's own documentation calls ordinal `score` its weakest
  primitive, and timing is where a wrong answer either wakes a caregiver needlessly or misses
  a fall. `escalate_by` and `trigger` stay deterministic rules. The classifier answers *what
  is allowed*, not *how many seconds from now*.

## Dataset

What exists today:

| source | size |
| --- | --- |
| scenarios | 49 (42 clean, 7 noisy variants) |
| checkpoints | 106 |
| labelled checkpoints | 47 |
| explicit action labels in fixtures | 317 |
| judged sentences (`annotations/say_verdicts.yaml`) | 66 |

Upstream Laya's finetuning notebook trains on ~30k questions. We have roughly two orders of
magnitude less, and **the data work is the project** — not the finetune, which is a
four-to-five hour Kaggle run on free 2×T4.

### The closed action space

Exactly 32 candidate actions: 5 `phase`, 3 `goal`, 8 `strategy`, 4 `notify`, 12 `say`. Two
consequences. It fits the 32-option capacity of the FluidUse Core ML buckets exactly, with no
headroom. And 47 labelled checkpoints × 32 candidates = 1,504 `bool` questions, against 317
that carry an explicit label today.

### The absent-label problem

The other 1,187 are unmentioned, and "unmentioned" is ambiguous: it can mean *irrelevant
here*, *acceptable but not worth listing*, or *the annotator did not think of it*. Feeding
that ambiguity to a classifier as a negative would teach it noise. Resolving it is the first
data task:

1. Where both Opus runs place an action the same way, take it (310 of 419 compared actions).
2. Where exactly one run places it and the other is silent (109 actions), ask explicitly. A
   closed question per disputed action, not a re-enumeration.
3. Where both are silent, ask once per checkpoint whether the remaining candidates are
   irrelevant or merely unlisted, in one batched call.

Step 2 is the cheap experiment worth running first: gemma4's inversion rate was 0.8%, so if
the reframing works, a *local* model can answer those 109 questions reliably. That is free,
and it tests the entire premise of this plan before any finetuning.

### Soft labels

The vote machinery already produces what training against a proper scoring rule wants.
A 3-0 vote is 1.0, a 2-1 vote is ~0.67, a doubtful single vote is 0.5. The half-point doubt
scoring from `VOTE_HANDOFF.md` becomes the training signal rather than just a report column.

### State serialization

512 tokens forbids raw dialogue history. Each question carries a compact state record built
from the scenario timeline: clock time, seconds since the person left the bed, current
`state` and `zone` plus confidence, the last utterance and its transcription confidence, a
repeat count, the session phase, and the relevant profile flags. This spec is a deliverable
in its own right — the live agent must be able to build the identical record at 3am, or the
benchmark measures something the agent cannot reproduce.

### Splits

- Split **by scenario id**, stratified by category.
- A noisy variant must land in the **same split as its parent** (`noise_of`). Otherwise the
  clean twin leaks the answer.
- The **human-reviewed labels are test-only** and never enter training.
- Report agreement with Opus and agreement with the human holdout as separate numbers. They
  are different claims, and conflating them is how the gemma circularity trap repeats.

## Metrics

Per backend, per question type:

- **Accuracy and macro-F1**, plus Cohen's kappa against the human holdout.
- **Calibration**: Brier score and expected calibration error. A calibrated 0.6 is more
  useful at the bedside than a confident 1.0, because thresholds can then be tuned per path.
- **Cost-weighted error.** Errors are not symmetric: calling a forbidden action acceptable
  (letting the agent contradict a distressed person, or miss an escalation) is far worse than
  calling an acceptable action forbidden. Weight the confusion matrix accordingly and publish
  the weights next to the score.
- **Latency**: p50 and p95 per single question and batched, with the hardware named, against
  a stated budget of 150 ms for one decision on the M4.
- **Footprint**: resident memory and whether it coexists with the Docker stack.
- **The ceiling**, in every report: Opus-vs-Opus and Opus-vs-human agreement on the same
  questions. A candidate scoring 80% against labels whose own agreement is 74% is telling us
  about the labels.

## Phases

### Phase 0 — feasibility, half a day, no labels needed

Fresh venv (torch 2.14 and transformers 5.x will not coexist with the existing venvs). Load
the English checkpoint, run a realistic serialized state through it on MPS, and measure
single and batched latency. Then try `FluidInference/laya-coreml` through `coremltools` to
measure the ANE path from Python, and check output parity against torch.

Gate: under 50 ms for a single question on the M4, and torch/Core ML outputs that agree.
Zero-shot accuracy is expected to be near chance (upstream reports 0.36 against 0.318 for
random) — that is not the gate.

### Phase 1 — harness and the wording task, one to two days

Start with the **wording judge**, not the decision layer: the label space is three booleans,
labelled data already exists, and `judge.py` batches 27 sentences per Opus call, so
augmentation is cheap. Harvest the `Say` sentences from the existing bench runs under
`../data-ai-agent-dementia/analysis/decision-bench/` to grow 66 sentences toward ~600, at
roughly 25 Opus calls.

Build `tests/classifier_bench/`: a JSONL question format, a `Classifier` protocol mirroring
the injectable `Runner` pattern in `annotate.py`, the baselines, the Opus ceiling, and the
report. Unit tests must not need a network, a GPU or Ollama.

Payoff even if nothing is finetuned: the bench stops needing Opus to score wording.

### Phase 2 — finetune on the wording task, two to three days

Upstream's Kaggle notebook, 4-5 hours for 4 epochs. Expect early stopping with far less data.

Gate: beat the `rules` baseline, approach Opus-vs-human agreement, stay calibrated, hold the
latency budget. If it clears that, it also becomes a real-time guardrail — a veto on a
sentence before Piper speaks it, which fits in the latency budget and is a safety win
independent of the benchmark.

### Phase 3 — the decision task

Build the 1,504-question set per the absent-label procedure, with the state serialization
spec. Opus ceiling, then finetune, gated on cost-weighted error and latency rather than raw
accuracy. Timing stays in rules.

### Phase 4 — v2 on the ANE

Convert the chosen checkpoint through `FluidInference/mobius`, verify parity against the
torch run question by question, then wire the classifier into the live path. Pin a commit and
vendor it: FluidUse is at v0.2.0 with 49 commits, and Laya has 36 open issues against 63
commits. Both are young.

## Rules

- **Nothing real leaves the machine.** `jev` and any other hosted backend may only ever
  receive synthetic scenario text. No recordings, no transcripts of real speech, no profile
  of a real person. Enforce it in code with an explicit opt-in flag and an assertion on the
  question source, not with a comment.
- **Never train on the test split**, and never report a score without the ceiling beside it.
- **Keep the candidate injectable**, as `annotate.py` does with `Runner`, so tests inject a
  fake and no test needs hardware.
- **Rules are a candidate, not scaffolding.** If the deterministic baseline matches a 421M
  model on the paths that matter, ship the rules and keep the classifier for the genuinely
  ambiguous judgments.

## Open questions

- Can `mobius` convert the English ModernBERT-large checkpoint, or is the ANE path limited to
  the multilingual one? Decides whether v2 costs a conversion or a checkpoint downgrade.
- Is 32 candidate actions within the Core ML bucket capacity in practice, given it is exactly
  at the documented limit?
- How much does the 512-token context bind once real profile flags are included?
- Do we need more scenarios before a finetune beats rules, and if so, is the cheapest source
  generated variants or state snapshots harvested from the bedroom recordings?

## 2026-09-22 result: the reframing works, with two caveats

`tests/classifier_bench/` implements the first slice: `build` derives the question set from
the 28 two-run annotations, `ask` puts one action at a time to a candidate, `score` reports
accuracy, calibration, inversions and baselines. gemma4 answered all 419 questions in about
25 minutes with no parse or validation failures.

### The premise holds

| asked as | agreement with Opus |
| --- | --- |
| enumerate the full sets (`decision_bench` annotator) | 27.0% |
| one action at a time, where it answered | **92.7%** |
| one action at a time, abstentions counted as wrong | 73.5% |

Baselines: always-acceptable 60.0%, always-forbidden 40.0%, random 47.4%. Brier 0.066,
5-bin ECE 0.027. State records came out at 63-203 rough tokens (median 109), comfortably
inside a 512-token encoder.

So the same model that looked useless as an annotator answers the classification form of the
question about as well as a second Opus run does (74.0% ceiling). The enumeration framing was
hiding a usable judge. That validates the plan's central bet.

### Caveat 1: it abstains, unevenly

20.6% of answers were `irrelevant`, and the abstentions are concentrated: **32 of 63
`notify` questions (51%)** versus 7 of 99 for `strategy`. It will not commit on whether a
notification level is allowed, which is precisely the escalation path. Forced accuracy
therefore lands at 73.5%, and `notify` forced accuracy is 49.2% — no better than a coin flip.

### Caveat 2: the errors run in the unsafe direction, confidently

Inversions are asymmetric: 2 of 186 acceptable actions called forbidden (1.1%), but **16 of
124 forbidden actions called acceptable (12.9%)** — every one of them at confidence 0.8-0.9.
Aggregate calibration looks fine (Brier 0.066) and hides this completely.

The 16 are systematic, not noise, which is the encouraging part for a finetune:

- 7 are `strategy: guided_return` where the person needs the toilet — it does not know that
  steering someone back to bed mid-need is forbidden (the TOIL clauses).
- 3 are `strategy: orient_time_place` at the wrong moment.
- 6 are `say` patterns, including `say: any` where the rule is that the agent must not speak
  at all — it reads "any" as permission rather than a blanket prohibition.

A handful of clause areas account for nearly all of it. That is learnable, and it tells us
where training data should be concentrated.

### Caveat 3: it cannot break ties

On the 109 disputed actions it sided with run 1 in 31 cases and run 2 in 28, abstaining on 35
and matching both or neither on 15. A coin flip. **The disputed items still need Opus**, so
the absent-label procedure in this plan keeps its step 2 as an Opus call.

### What this changes

- Phase 3 (the decision task) is worth doing, and per-action questions are the right form.
- A local model is usable as a **labelling assistant that may abstain**, not as an
  unsupervised labeller and certainly not yet as a safety veto. Abstention plus the unsafe
  inversion rate means every `forbidden` judgment it makes still needs review.
- Report cost-weighted error, not accuracy, as the headline. A 92.7% that misses one in eight
  prohibitions is worse at the bedside than a duller model that never does.
- Training data should over-sample the TOIL clauses, the `say: any` blanket prohibitions and
  the `notify` levels, since that is where both the errors and the abstentions sit.

Artifacts: `../data-ai-agent-dementia/analysis/decision-bench/2026-09-22-action-probe-{questions,answers,report}.{jsonl,json}`.
