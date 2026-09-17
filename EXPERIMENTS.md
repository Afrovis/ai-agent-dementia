# EXPERIMENTS.md

A dated log of measured experiments: what was asked, how it was run, what came
out, and what is still unresolved. Newest first. Numbers here are measurements,
not targets; the gates themselves live in [PLAN.md](PLAN.md) section 12.

Each entry records enough to re-run it. Raw bench JSON is not committed —
`data/` is gitignored and bench output is ephemeral by design — so the tables
below are the record.

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
