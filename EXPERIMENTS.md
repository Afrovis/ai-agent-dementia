# EXPERIMENTS.md

A dated log of measured experiments: what was asked, how it was run, what came
out, and what is still unresolved. Newest first. Numbers here are measurements,
not targets; the gates themselves live in [PLAN.md](PLAN.md) section 12.

Each entry records enough to re-run it. Raw bench JSON is not committed —
`data/` is gitignored and bench output is ephemeral by design — so the tables
below are the record.

---

## 2026-09-17 — The compose prompt's own rules, measured for the first time

### Question

`validate_say` checks five things: non-empty, one sentence, the three forbidden
phrases, question form, and the silence gap. `_COMPOSE_TASK` in
`services/agent/agent/llm.py` instructs the model to do five more things that
nothing scored: avoid "but", respect `things_to_avoid`, state the time only when
asked about it, invent nothing, and address the person by
`profile.preferred_address`. Composition pass rate has always been measured
against the first list only. How well does the shipped model obey the second?

### Setup

- Mac mini, Apple M4, 16 GB. Ollama on `localhost:11434`, `gemma4:e4b-mlx`.
- `tests/dialogue_bench`, the same 50 scenarios, `--timeout 30`. The model was
  pre-warmed with a throwaway generate call first (cold start is ~11 s; see the
  entry below).
- Two runs. n=2 is enough to see a 50-point gap, not to rank anything.
- **The prompt and both payloads are unchanged.** Assertion configuration lives
  at the YAML document level (`default_must`, `default_must_not`,
  `default_avoid_terms`) as siblings of `default_profile`, never inside it,
  because `scenario.profile` is passed verbatim to the model. A new profile key
  would have changed the model's input and made every number here incomparable
  to the entry below.

Five deterministic checks were added in `dialogue_bench/checks.py`. No LLM
judging; pure string work over the composed sentence.

### Headline result

| measure | run 1 | run 2 |
| --- | --- | --- |
| safe compositions (`validate_say`) | 50/50 | 48/50 |
| **prompt-rule assertions passed** | **23/50** | **24/48** |
| `states_clock_time` violations | 17 | 15 |
| `conjunction_but` violations | 13 | 10 |
| `avoid_terms` violations | 0 | 0 |
| `invents_proper_noun` violations | 0 | 0 |
| `addresses_by_name` failures | 0 | 0 |

**Composition scored perfect on the old measure while about half of all replies
broke a rule the prompt states explicitly.** Run 2 had two model calls rejected
outright, which is why its denominators are 48.

### The two violations are real, not check artefacts

`conjunction_but`, 10–13 of 50. The model produces the exact construction the
rule exists to prevent — acknowledge, then cancel the acknowledgement:

```
[person-01] Jean, thinking of your spouse is natural, but let's rest now while Tom helps you settle.
[person-02] Jean, I hear you are looking for your mother, but let's rest now while Tom stays nearby.
[person-03] Jean, I know you're thinking about Tom, but let's rest now while I stay with you.
```

For someone looking for a deceased spouse, "thinking of your spouse is natural,
but let's rest now" is the dismissal the validate-then-redirect strategy is
written to avoid. `validate_say` accepts every one of these.

`states_clock_time`, 15–17 of the 43 scenarios where the check applies. The
model volunteers the clock time to someone who asked about their husband or the
toilet:

```
[restroom-07] Jean, since 3 o'clock, your bladder is full; the restroom is through the bedroom door and immediately left.
[person-06] Jean, it is three o'clock at night, and Tom is nearby and everything is settled; let's return to bed now.
[person-07] Jean, since it is 3 o'clock at night, let's look at a garden photo and settle down to rest.
```

Unprompted reality-orientation at 3am, in a system whose whole premise is not to
do that.

### Suspected cause of the clock violations

`compose` always receives `time_words` in its payload and is then told to mention
the time only if the utterance is about it. The time is handed over and the model
is trusted to decline; it declines about 60% of the time. Withholding `time_words`
from the payload unless the utterance is time-related is the obvious remedy, and
it changes the payload, so it cannot be evaluated on these 50 scenarios — they
are a dev set already tuned against. It needs the held-out set first.

### What the checks do not catch

`invents_proper_noun` scored zero violations, which is weaker evidence than it
looks. It flags a capitalised token absent from the model's input, so it catches
an invented *name* and misses an invented lowercase fact or plan entirely —
"your son will pick you up in the morning" passes clean. Confabulation is not
measured; only one narrow form of it is. `states_clock_time` is likewise narrow
by choice: it matches `o'clock`, `H:MM`, `am`/`pm`, `midnight` and `noon`, and
deliberately ignores vague day-parts, because the shipped caregiver template
says "talk more in the morning" and that is allowed.

### Caveats

- Intent accuracy came out 46/50 and 45/50 across the two runs, against the
  0.880 three-run baseline below. This is **not** an improvement: the payload is
  identical, n=1 per run, and the gap sits inside the sampling spread already
  documented. Do not quote it as a new baseline.
- Latency was higher than the entry below (interpret mean 1.03 s vs 0.29–0.61 s).
  The box was not idle. Not comparable.
- The 50 scenarios remain a dev set. These assertion numbers are a measurement of
  the shipped prompt, not a gate, and tuning against them burns them further.

### Conclusions

1. **The old composition pass rate measured almost nothing about quality.** 50/50
   safe and 23/50 rule-following are the same 50 sentences.
2. **`conjunction_but` and `states_clock_time` are live defects in the shipped
   prompt**, at roughly 25% and 35% of eligible scenarios respectively.
3. Written prompt rules are not self-enforcing. Every rule worth stating in
   `_COMPOSE_TASK` needs either a deterministic check or an accepted blind spot,
   recorded as such.

### Open actions

- [ ] Fix the two measured violations. Withholding `time_words` when the
      utterance is not about time is the first thing to try for the clock case.
- [ ] Write the held-out scenario set before tuning the compose prompt, for the
      same reason the interpret prompt already needs one.
- [ ] Extend the anti-confabulation check beyond invented proper nouns, or state
      plainly in `HANDOFF.md` that confabulation is unmeasured.
- [ ] Multi-turn trajectories: nothing here or below tests a sequence, and both
      cloud-fallback triggers (two consecutive `unclear`, plan confidence < 0.4)
      are sequence properties that no current test can fire.

---

## 2026-09-17 — Confirming `gemma4:e4b-mlx`, and what actually earned the score

### Question

`gemma4:e4b-mlx` was adopted as the local text model in #66. Three things were
unverified: whether its dialogue-bench score is stable rather than sampling
noise, whether a larger model does better, and how much of the score comes from
the model versus the prompt.

### Setup

- Mac mini, Apple M4, 16 GB. Ollama on `localhost:11434`, otherwise idle box.
- `tests/dialogue_bench`, 50 scenarios, three runs per cell.
- `OllamaLLM` sends no `options` block, so temperature is the model default and
  every run is independently sampled. This is why repeats are necessary.
- Models pre-warmed; cold-start figures reported separately.
- Arms were run by injecting a client into `dialogue_bench.scoring.run_model()`,
  which already accepts one. The bench itself was not modified.

**Three prompt variants** were compared, all against the same 50 scenarios:

| variant | what it is |
| --- | --- |
| `pre-#66` | the old one-line task string; the schema reached the model only through Ollama's `format` constraint |
| `main` | the post-#66 prompt, which spells the JSON schema out in the prompt text |
| `main + defs` | `main`, plus a one-line written definition of each of the seven intents |

### Headline result

Mean intent accuracy over three runs of 50 scenarios:

| arm | accuracy | composition pass | interpret mean |
| --- | --- | --- | --- |
| `gemma4` / `pre-#66` | 0.473 (0.46–0.50) | 0.94 | 0.56 s |
| **`gemma4` / `main`** | **0.880 (0.88–0.88)** | 0.83 | 0.61 s |
| `gemma4` / `main + defs` | 0.880 (0.86–0.90) | 0.89 | 0.29 s |
| `qwen3.5:9b` / `pre-#66` | 0.500 (0.48–0.52) | 0.95 | 2.41 s |
| `qwen3.5:9b` / `main + defs` | 0.840 (0.82–0.86) | 0.93 | 3.32 s |

**The prompt is worth about 40 accuracy points; the model is worth about 3.**
Spread within each cell is 0–4 points, so none of this is sampling noise.

The 40 points were already earned by #66. Writing explicit class definitions on
top of it bought nothing overall — 0.880 either way — and, as below, cost
accuracy where it matters most.

### Per-class recall

Mean over three runs. Seven scenarios per class, eight for `unclear`.

| intent | `pre-#66` | `main` | `main + defs` |
| --- | --- | --- | --- |
| `need_restroom` | 1.00 | 1.00 | 1.00 |
| `pain` | 1.00 | 1.00 | 1.00 |
| `fine` | 0.05 | 1.00 | 1.00 |
| `looking_for_person` | 0.10 | 0.86 | 1.00 |
| `confused_time` | 0.38 | 0.86 | 1.00 |
| `unclear` | 0.75 | 0.83 | 0.75 |
| **`wants_to_leave`** | 0.00 | **0.62** | 0.43 |

`wants_to_leave` confusion, summed over three runs (21 attempts):

| arm | predictions |
| --- | --- |
| `pre-#66` | `confused_time` 7, `unclear` 9, `need_restroom` 5, **correct 0** |
| `main` | `unclear` 4, `fine` 4, **correct 13** |
| `main + defs` | `confused_time` 9, `looking_for_person` 3, **correct 9** |

**Do not add the class definitions.** They are neutral on aggregate accuracy and
a clear regression on the one class that matters most for safety. The added line
telling the model that "wanting to go home while already at home is
`wants_to_leave`, not confusion about place" appears to have pushed predictions
*toward* `confused_time` rather than away from it. The reason is not understood;
the measurement is repeatable.

### Why the pre-#66 prompt failed, and why #66 fixed it

The old prompt passed the seven intents to the model only as bare enum strings
inside Ollama's `format` constraint, with no prose anywhere saying what any
class meant. The model inferred each class from its label name alone, and the
results track that exactly: `need_restroom` and `pain` are self-evident from the
name and scored 1.00 untouched, while `fine` collapsed into `confused_time` 13
times out of 21 and `wants_to_leave` was never once identified.

#66's `_prompt` now spells the schema out in the prompt text. Its docstring
states the reasoning, and this experiment independently confirms it: constraint
alone never shows the model the allowed values, and it guesses badly without
them.

### Model comparison: `qwen3.5:9b` is rejected

| | `gemma4:e4b-mlx` | `qwen3.5:9b` |
| --- | --- | --- |
| best accuracy measured | 0.880 | 0.840 |
| interpret mean / max | 0.29–0.61 s / 0.42 s | 2.41–3.32 s / 3.89 s |
| compose mean / max | 0.37–0.40 s / 0.55 s | 3.12–3.14 s / 3.76 s |
| resident size | 8.8 GB | 6.6 GB |

HANDOFF.md section 6 budgets `interpret` under 1 s and `compose` under 2 s.
`gemma4` meets both with an order of magnitude to spare. `qwen3.5:9b` misses
both by a wide margin on an idle box, before `perceive` and `listen` compete for
the same 16 GB — and it is no more accurate.

It also fails in a worse direction. `qwen3.5:9b` classified `unclear`
utterances as `fine` eight times out of 21, meaning the system stays quiet when
it does not understand. `gemma4` never does this.

### Two defects in `services/agent/agent/llm.py`, both present on `main`

**1. A thinking-enabled model scores zero, silently.** `qwen3.5:9b` has thinking
on by default. Ollama returns the whole JSON answer in the `thinking` field and
leaves `response` an empty string. `OllamaLLM._call` reads
`response_body["response"]`, finds nothing usable, and returns `None` on every
call. The service falls back to caregiver templates and logs only
`{"reason": "response had no usable text"}` with no model text — a total failure
indistinguishable from a working system.

Adding `"think": false` to the request body fixes it completely. `OllamaLLM`
has no way to send that field; `OpenAICompatibleLLM` already disables thinking
through `chat_template_kwargs`, so only the Ollama path is exposed. Every
`qwen3.5` figure above was measured with `think:false` patched in. Without it
the model scores 0.000.

**2. The first call of the night always times out.** Cold-start latency measured
12.36 s for `gemma4:e4b-mlx` and 9.61 s for `qwen3.5:9b`. `OllamaLLM.__init__`
defaults `timeout_seconds` to 2.0 and `AgentConfig` passes 10.0. Either way the
first exchange of a session falls back to a template. Independent of model
choice.

### What held up

The deterministic safety layer did its job. One `qwen3.5` composition was
rejected by `validate_say` for the forbidden word "no" inside the phrase
"there's no rush" — conservative, and correct for the rule as written. No
composition reached the caller unvalidated in any run.

### Caveats

- **The 50 scenarios are a dev set, not a test set.** A prompt was tuned against
  them in this experiment. Further prompt iteration needs held-out scenarios or
  the numbers stop meaning anything.
- **Composition figures are not comparable to `main`.** The harness used the
  pre-#66 compose task; `main`'s compose is goal-aware. Only `interpret` is a
  like-for-like comparison. Composition pass rate ranged 0.78–0.98 across all
  fifteen runs, wide enough that no arm's composition score is meaningful.
- `qwen3.5:9b` was never measured under `main`'s exact prompt — its arms used
  `pre-#66` and `main + defs`. Its latency disqualifies it regardless.
- All figures are warm-model, idle-box. Nothing here measures the agent running
  alongside `perceive` and `listen` under real memory pressure.
- n=3 per cell. Enough to separate a 40-point effect from a 4-point spread, not
  enough to rank two models a few points apart.
- `llama3.1:8b`, `qwen2.5:7b` and `mistral:7b` have still never been run on this
  bench. There is no baseline for them.

### Conclusions

1. **Keep `gemma4:e4b-mlx` at `main`'s current prompt.** 0.880, stable to the
   third decimal across three runs, comfortably inside both latency budgets.
2. **Do not adopt `qwen3.5:9b`**, and do not read its bench score as promising:
   it is slower by an order of magnitude, no more accurate, and fails toward
   silence.
3. **Do not add intent class definitions to the prompt.** Measured neutral
   overall and a regression on `wants_to_leave`.
4. **`wants_to_leave` remains the blocker.** Best measured recall is 0.62, and
   it is the exit-seeking case — the person heading for the front door at night.
   No prompt variant tried here fixes it.

### Open actions

- [ ] Dedicated work on `wants_to_leave` before any supervised pilot (#27).
- [ ] Let `OllamaLLM._call` send `think: false`, so a thinking-enabled model
      fails loudly instead of scoring zero in silence.
- [ ] Reconcile the cold-start timeout, or pre-warm the model at session start.
- [ ] Write held-out scenarios before tuning the interpret prompt further.
