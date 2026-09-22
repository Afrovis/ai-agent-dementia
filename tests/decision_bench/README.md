# decision_bench

An on-demand benchmark for the agent's **decisions**, not only its words.

`dialogue_bench` feeds the agent one utterance and checks the intent and the
wording of the reply. `decision_bench` feeds it a scripted night: a timeline
of `PersonState` readings and `Utterance`s spread over several minutes. It
then checks what the agent did at each labelled moment: which phase it
entered, which strategy it ran, which goal it set, whether it notified a
caregiver, and how quickly.

It is not a CI gate. Run it when you change the session rules, the strategy
order, the prompts or the model, and compare the reports.

> Status: phase 3 of [PLAN.md](PLAN.md). The seven pilot scenarios carry
> human-reviewed labels drafted by the isolated annotator, so runs produce
> pass rates. Scaling up to about 30 scenarios is phase 4.

## What it measures

For every scenario the harness drives `agent.main.run_once` in-process with
an in-memory bus and injected clock. There is no Redis, camera or microphone,
and the real LLM backend interprets speech and writes the replies. Each labelled
checkpoint is scored on:

| Measure | Meaning |
| --- | --- |
| Checkpoint pass | The agent's action inside the checkpoint window is in the `acceptable` set. |
| Critical violation | The agent did something in `must_not`, such as correcting the person's reality, sending an urgent page on a false alarm, or saying a clock time unprompted. Reported separately from ordinary misses. |
| Escalation latency | Seconds from the triggering event to `Notify`, compared with the checkpoint's `escalate_by` deadline. Escalating too late is critical. |
| Wording | Every `Say` goes through the `dialogue_bench` rule checks (`checks.py`): no "but", avoid-terms, no invented names or places, and so on. |

Results are broken down by category and by clean versus noisy input, with
models side by side.

## Scenario categories

About 30 scenarios, roughly five per category, plus about 8 noisy variants:

| Category | Example |
| --- | --- |
| `restroom` | Sits up, says "I need the toilet", walks towards the bathroom path. |
| `disorientation` | "I have to pick up the kids", heading for the door. |
| `distress_pain` | Crying, "my leg hurts", repeated distress. |
| `fall` | `on_floor`, or `absent` for too long after leaving the bed. |
| `false_alarm` | Rolls over, sits up for 20 s, lies back down. The agent should stay quiet. |
| `silent_wander` | Gets up and walks with no speech at all. |

A noisy variant has `noise_of: <parent id>` and the same labels as its
parent. It adds flickering states, low confidence, speech-to-text typos or a
dropped utterance, to test hysteresis and robustness.

## Scenario format

One file per scenario in `fixtures/scenarios/<id>.yaml`. The schema is
`decision_bench/schema.py`; the profile every scenario starts from is
`fixtures/profile.yaml`.

```yaml
id: disorientation-01
category: disorientation
summary: Believes they must collect the children and heads for the door.
start: "02:14"                     # quoted local time of t=0; decides night vs day
profile: {}                        # optional overrides of the default profile
voice_clip: false                  # true if the caregiver recorded a consented clip
noise_of: null
timeline:                          # t = seconds from start, in order
  - {t: 0,  person: {state: sitting_up, zone: bed, confidence: 0.9}}
  - {t: 25, person: {state: standing,   zone: bed}}
  - {t: 40, utterance: {text: "I have to pick up the kids from school"}}
  - {t: 55, person: {state: walking,    zone: door}}
checkpoints:
  - id: first-response
    window: [40, 70]
    question: What should the agent do, and what must it not say?
    acceptable:
      - {strategy: validate_and_redirect}
      - {strategy: soft_greeting}
    must_not:
      - {say: correction_of_reality}
      - {notify: any}
    rationale: Validate the wish to care for the children before redirecting.
    cites: [VAL-01, AA-03]
  - id: caregiver
    escalate_by: 300               # notify within 5 min of the trigger
    trigger: 55                    # scenario second the deadline counts from
    threshold_source: caregiver    # a time threshold comes from the caregiver, not the evidence
    rationale: Someone should know the person is up and trying to leave.
```

A checkpoint starts **unlabelled**: just its `id`, `window` and the
`question` the annotator answers. The labels (`acceptable`, `must_not`,
`escalate_by`) come from the annotation workflow below. A labelled
checkpoint needs a `rationale`, and action labels need `cites`. The
`familiar_voice` strategy is only a valid label when `voice_clip` is true.

The set of actions a label can name is exactly what the agent can do:

- **phase**: `IDLE`, `OBSERVING`, `ENGAGED`, `COOLDOWN`, `ESCALATED`
- **goal**: bed, restroom, comfort
- **strategy**: the ids in `config/strategies.example.yaml`
- **notify**: level, and whether one was sent at all
- **say**: `any`, the `dialogue_bench` rule checks, and the named wording
  patterns listed in `guidelines.md`

## How labels are made

Labels come from the guideline pack in `guidelines.md`. Its clauses are
numbered and paraphrased, and each links to its source: NICE NG97, the
Alzheimer's Association, validation therapy and person-centred care, DICE,
and the evidence on nighttime falls and toileting. Every checkpoint cites
the clauses it rests on.

1. A model drafts scenarios from the category list, and a human prunes them.
2. A model annotator labels each checkpoint, with a rationale and citations.
   It sees the guidelines and the list of actions above, but **not** the
   current strategy config or agent code. That keeps the labels from just
   copying what the agent already does.
3. A human reviews every label. Disagreements appear side by side, the human
   makes the final call, and each disagreement is logged in
   `annotations/disagreements.yaml`.

Time thresholds such as "escalate within N seconds" do not come from the
guidelines. They are marked `threshold_source: caregiver` and are never
attributed to the evidence.

### Annotating and reviewing

The project skill `.claude/skills/decision-bench-annotate` runs this loop.
By hand:

```sh
python -m decision_bench annotate --dry-run --scenario <id>   # print the exact prompt
python -m decision_bench annotate --scenario <id>             # two Opus runs + triage;
                                                              # unflagged scenarios are applied
# only if flagged: edit annotations/review/<id>.yaml, set reviewed: true
python -m decision_bench apply <id>                           # labels into the fixture
```

To evaluate a local model as one of the two first-pass annotators, write its
single independent run to a separate directory (the recommended location is
`annotations/local/`):

```sh
python -m decision_bench annotate --backend ollama --model qwen3.5:9b \
  --out-dir annotations/local [--scenario <id> ...]
python -m decision_bench calibrate --local-dir annotations/local \
  [--out calibration.json]
```

The Ollama annotator calls `http://127.0.0.1:11434/api/chat` by default; use
`--ollama-url` to change it and `--no-think` to disable thinking. It requests
JSON mode with temperature zero, includes the schema in the prompt, and
validates replies strictly without retaining thinking text. Existing files in
the local output directory are skipped unless `--force` is given. For safety,
a local run requires `--out-dir`, refuses the
model and review annotation directories, creates no review file, and never
applies labels to fixtures.

`calibrate` compares every available local file with both stored Opus runs
using the same pair-agreement rule as tiebreak triage. Its summary highlights
the local+Opus-1 agreement rate (the fraction that would avoid a second Opus
call), action placement agreement, escalation/trigger agreement, runtime and
parse/validation retries. The optional JSON report includes checkpoint detail
and potentially harmful local+Opus-1 agreements that the two Opus runs would
have sent to a tiebreak.

`annotate` runs `claude -p --model opus` with no tools and no settings from
an empty temp directory. It sends the prompt in
`decision_bench/annotator_prompt.md` plus `guidelines.md`, the profile and
the scenario without any existing labels, so the annotator cannot see the
agent or its config. It may cite only clauses whose `Checked` box is ticked.
Its output must pass the schema, and it gets one retry with the problems
listed. It writes `annotations/model/<id>.yaml` (the untouched draft, with
the model id, cost and hashes of the prompt, guidelines and scenario) and
`annotations/review/<id>.yaml`. The review file is self-contained: the
timeline with clock times, each checkpoint's question, the annotator's labels
and its doubts, each marked as the annotator's. `apply` refuses an
unreviewed file or a changed checkpoint without a `reason`, rewrites only the
fixture's `checkpoints:` block, and logs each change in
`annotations/disagreements.yaml`. `annotate --review-only --force` rebuilds
the review files from the drafts without calling the annotator, overwriting
any edits.

### Triage: which labels need a human

`annotate` runs the annotator twice, independently, and compares the runs
(`decision_bench/triage.py`). A checkpoint is flagged for a human when:

- either run sets `needs_review` (it could reasonably go either way);
- the two runs have different `must_not` sets;
- one run sets an escalation deadline and the other does not;
- their `acceptable` sets have nothing in common.

A different deadline *number* is not flagged, because it is a caregiver
placeholder either way. A scenario with no flags is accepted as the model
labelled it: its review file gets `reviewed_by: model` and `annotate`
applies it straight away (`--no-apply` stops that). A flagged scenario waits
for a human, and its review file marks each flagged checkpoint `NEEDS YOUR
REVIEW` next to the second opinion's labels. Triage never accepts what it
cannot check: a draft without the self-flag or without a second opinion is
always flagged. Editing a model-accepted file requires
`reviewed_by: human`.

### Verdicts on spoken sentences

`correction_of_reality`, `infantilising` and `unsupported_claim` have no
string rule. With `--judge`, each new sentence goes to an isolated Claude
evidence judge: `claude -p` with no tools, on the claude.ai subscription,
never an API key. The judge sees the profile and that sentence's context:
the time words, the person's last words, the camera reading, and whether a
caregiver was notified. It decides each pattern and quotes its evidence.
Its answer is stored under `judge:` in `annotations/say_verdicts.yaml`.

The top-level values in that file are the human's. A human value always
wins, and every place where it disagrees with the judge is printed after the
report. A `true` verdict counts as a wording failure on every run.
Verdicts are per sentence, and the judge sees the context of the sentence's
first occurrence. `--collect-verdicts` adds new sentences without judging
them.

## Running

The scenario clock ticks once per second. Person readings repeat at
`perceive`'s default 60-second heartbeat cadence, so state persistence and
the agent's zone hysteresis follow the live data path. Before the first
scenario, each real model gets one untimed warm-up call, because a cold model
can exceed `--timeout` and those failures would be charged to the first
scenario. `--no-warmup` skips it.

The interface mirrors `dialogue_bench`:

```sh
pip install -e services/agent -e tests/dialogue_bench -e tests/decision_bench[dev]
pytest tests/decision_bench/tests          # unit tests, no LLM needed
python -m decision_bench --backend stub    # deterministic smoke run
python -m decision_bench --backend stub --trace
python -m decision_bench --model gemma4:e4b-mlx --model llama3.1:8b
python -m decision_bench --backend openai --base-url http://127.0.0.1:11435 --model <mlx-model>
python -m decision_bench --category fall --scenario fall-01   # narrow the run
python -m decision_bench --backend stub --json --out decision-report.json
python -m decision_bench --model gemma4:e4b-mlx --judge   # evidence-check new sentences (subscription)
```

## Caveats

The scenarios are synthetic and contain no personal data. The results
measure agreement with guideline-based labels, not clinical safety. This is
not a medical device and not a substitute for supervision.
