# Laya feasibility probe (classifier bench, Phase 0)

Phase 0 of [docs/CLASSIFIER_BENCH.md](../../docs/CLASSIFIER_BENCH.md): is a Laya
typed-decision model fast enough on the M4, and does the Core ML conversion compute the
same thing as torch? No labels, no accuracy claims. Zero-shot answers are recorded in the
JSON only for the record; they are near chance, as upstream warned.

## Setup

torch 2.14 and transformers 5.x do not coexist with the other venvs, so this gets its own:

```sh
/opt/homebrew/bin/python3.12 -m venv .venv-laya
.venv-laya/bin/pip install laya coremltools huggingface_hub
```

Measured with laya 0.3.5, torch 2.14.0, transformers 5.17.0, coremltools 9.0 (which warns
it was tested up to torch 2.7; it loads and runs compiled models fine). Checkpoints are
`convaiinnovations/laya` at revision `1c5edc17` and `FluidInference/laya-coreml` at
`7b8d7a2b`, which was converted from that same laya revision.

## Run

From this directory:

```sh
../../../.venv-laya/bin/python torch_latency.py --device mps --out results/torch_english_mps.json
../../../.venv-laya/bin/python torch_latency.py --device mps --fp16 --out results/torch_english_mps_fp16.json
../../../.venv-laya/bin/python torch_latency.py --device mps --subfolder multilingual --out results/torch_multilingual_mps.json
../../../.venv-laya/bin/python coreml_parity.py --bucket 256 --out results/coreml_L256.json
../../../.venv-laya/bin/python coreml_parity.py --bucket 512 --out results/coreml_L512.json
```

(Paths assume the venv sits at the repository root.) `probe.py` holds the one state and
the questions: restroom-01 at t=35, serialized to about 180 tokens, and one `bool`
question per action in the closed 32-action space.

## Results, 2026-09-22, Mac mini M4 16 GB, Docker stack running

p50 in ms. "Decision" is all 32 `bool` questions for one state.

| backend | 1 question | decision (32 bool) | 32-option choice | resident memory |
| --- | --- | --- | --- | --- |
| English, torch MPS fp32 | 71.6 | 1,775 | 110.5 | 3.1 GB |
| English, torch MPS fp16 | 70.7 | 1,683 | 109.0 | 2.2 GB MPS |
| multilingual, torch MPS fp32 | 30.6 | 709 | 58.6 | 4.6 GB |
| multilingual, Core ML L256, CPU+ANE | **11.1** | 354 | (truncated, see below) | 0.3 GB |
| multilingual, Core ML L256, all units | 11.8 | 378 | | |
| multilingual, Core ML L512, CPU+ANE | 33.5 | 1,073 | 33.5 | 0.3 GB |
| multilingual, Core ML L256, CPU only | 32.7 | 1,049 | | |

Parity, Core ML fp16 against torch fp32 on CPU, 34 questions (32 `bool` and two
`choice`): largest probability difference 0.011 at L256 and 0.008 at L512 on CPU+ANE;
34/34 argmax agree and 32/32 `bool` answers land on the same side of 0.5, on every
compute unit.

## What it means

**Gate: passed on the ANE, failed for the English checkpoint.** The plan's gate is under
50 ms for one question and torch/Core ML agreement. The multilingual checkpoint on the ANE
does one question in 11 ms at L256 and agrees with torch. The English ModernBERT-large
checkpoint does not make 50 ms on torch/MPS at all, and fp16 does not help: at ~150 GFLOP
per question the GPU is compute-bound, not precision-bound. So the English v1 candidate
only becomes viable if `mobius` can convert it, which is still the plan's first open
question and was not attempted here.

**The 150 ms decision budget rules out 32 `bool` questions per decision in the live path.**
Laya encodes one sequence per question and re-reads the state every time, so 32 questions
cost 32 forward passes: 354 ms at best. The 1,504-question `bool` framing is fine for
the benchmark, but at 3am the agent needs one of:

- a shortlist, where rules narrow the 32 candidates to the handful that are live in this
  state (about 10 at 11 ms each fits the budget);
- one `choice` per action family (phase, goal, strategy, notify: 4 × 11 ms), plus `bool`
  only for the wording checks, which run per sentence anyway.

**32 options fit the Core ML bucket, but not with a state at L256.** The 32-option question
fills the whole 256-token option head, so at L256 the state is truncated to nothing. With
the state it is 343 tokens, which needs L512 (33.5 ms). Per-family choices of 3 to 12
options avoid this and stay in L256.

**Context.** This state is about 180 tokens with a short profile. L256 leaves roughly 70
tokens of headroom before a question drops to L512 and triples in cost, which is the real
answer to the plan's 512-token question: the budget that binds is 256, not 512.

**Footprint.** Core ML runs in about 300 MB resident, next to the full Docker stack. torch
needs 3 to 5 GB, which is a lot on a 16 GB machine that also hosts the LLM.
