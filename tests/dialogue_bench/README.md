# Dialogue regression bench (issue #17)

This suite runs 50 synthetic night-time utterances through the agent's real
structured `interpret` and `compose` calls. It reports overall and per-class
intent accuracy, an intent confusion matrix, and the proportion of composed
responses accepted by the same deterministic speech gate used at runtime. It
also records mean and worst end-to-end call latency for each task; this is full
response latency, not the first-token timing targeted in `PLAN.md`.

The composition check enforces the product's reviewable tone rules: one
sentence, at most 20 words, no question, and none of “no”, “you can't”, or
“you're wrong”. A timeout, malformed JSON, invalid output shape, or absent
response counts as a measured failure; it is never silently skipped.

All fixtures are invented for this test. There are no recordings, transcripts,
or details from a real person.

## Run the unit tests

No Ollama, Redis, camera, microphone, or network is needed:

```sh
python3.12 -m venv .venv-dialogue
source .venv-dialogue/bin/activate
pip install -e services/agent
pip install -e tests/dialogue_bench[dev]
pytest tests/dialogue_bench/tests
```

## Compare local models

With Ollama running locally and the model weights already pulled:

```sh
python -m dialogue_bench
```

The default comparison is `llama3.1:8b`, `qwen2.5:7b`, and `mistral:7b`.
Choose another set by repeating `--model`:

```sh
python -m dialogue_bench --model llama3.1:8b --model qwen2.5:7b
python -m dialogue_bench --model llama3.1:8b --json
```

JSON reports omit generated sentences by default so a report can be shared
without unexpectedly carrying text. Add `--include-text` only when reviewing
the actual synthetic outputs locally.

The command does not invent a universal pass threshold or choose a model from
an unavailable run. It exits zero after a completed comparison and exposes all
failures in the report. Model selection is a caregiver/project-owner decision
based on these measured results, latency, and available memory.
