# Promoted scene_lab moments (drafts)

Text-only scenarios promoted from fake-live scene_lab runs on 2026-09-23 with
`python -m scene_lab promote`. The person's lines were played by the Claude mind
over synthetic audio; no real person's audio, frames or night is here.

Both are `status: draft` and wait for owner approval. They fail on the current
agent on purpose: each reproduces a live turn-taking bug, and each passes once
the agent replies to the flagged utterance within 5 s.

| Scenario | Source | Flagged moment | Why it fails now |
| --- | --- | --- | --- |
| `promoted-alone-no-reply` | `2026-09-23T0015-live/persona-repeated-question` at 33.9 s | "I don't like being here on my own." | Interpreted `unclear` (distress 1); the only Say that follows is the ladder's scheduled `soft_greeting`, not a reply. |
| `promoted-car-keys-no-reply` | `2026-09-22T2348-live/disorientation-01` at 103.7 s | "Where are my car keys? They'll be waiting." | Interpreted `unclear`; nothing answers it, the next Say is a scheduled `guided_return`. |

Replay either one deterministically, with recorded interpretations and recorded
LLM latencies:

```sh
python -m session_replay run tests/scene_lab/promoted/session_replay/promoted-alone-no-reply.jsonl \
  --expect tests/scene_lab/promoted/session_replay/promoted-alone-no-reply.expect.yaml \
  --llm recorded --llm-latency recorded
```

The expectation asks for `{type: Activity, decision: said, reply: true}`: the
agent's own record that a Say was caused by the person's speech, not a dwell
timer. Once approved, move the pair into `tests/session_replay/scenarios/`.
The `decision_bench/` copies are unlabelled and go through the
`decision-bench-annotate` flow. Each `*.scene.yaml` is the closed-loop scene
card, to re-run live with `python -m scene_lab run`.
