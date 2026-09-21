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

> Status: phase 2 of [PLAN.md](PLAN.md). The real-agent replay harness,
> scoring, reports, CLI and seven unlabelled pilot scenarios exist. Reviewed
> labels are phase 3, so the pilots currently produce review summaries rather
> than pass rates.

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
```

## Caveats

The scenarios are synthetic and contain no personal data. The results
measure agreement with guideline-based labels, not clinical safety. This is
not a medical device and not a substitute for supervision.
