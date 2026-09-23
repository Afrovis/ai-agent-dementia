# Session replay

Replay a recorded bedside exchange through the current agent without Redis,
camera, microphone, or a real clock. The runner uses `nc_shared.bus.FakeBus`,
publishes each input at its original timestamp, and calls `agent.main.run_once`
at every input and every 0.5 simulated seconds through a 60 second tail.
The replay night window is always on. `--tz` (default `America/New_York`)
sets the local clock used by caregiver phrases. Agent publication timestamps
in the timeline are simulated, including deferred speech.

From the repository root:

```sh
pip install -e shared -e services/agent -e 'tests/session_replay[dev]'
python -m nc_shared.replay export redis://bus:6379 data/sessions/desk-export.jsonl
python -m session_replay extract data/sessions/desk-export.jsonl \
  tests/session_replay/scenarios/desk.jsonl --since 2026-09-22T22:51:40Z --until 2026-09-22T22:53:30Z
```

`extract` keeps `PersonState` from `person` and `Utterance` from `speech_in`.
It also retains `SessionState`, `GoalChanged`, `Say`, `Show`, `LightCommand`,
and `Notify` as `observed: true` reference rows. Agent LLM `Activity` end
rows keep only kind, phase, duration_ms, and timestamp for latency replay;
their detail is removed. It drops speech starts, health, other activity,
pose debug, frames, and audio. The runner ignores observed rows as inputs.
Review the extracted text before committing; raw exports stay in
gitignored `data/`, and only extracted, text-only scenarios belong in git.

Write a sibling `desk.expect.yaml` with recorded interpretations and ordered
expectations, then run:

```sh
python -m session_replay run tests/session_replay/scenarios/desk.jsonl \
  --expect tests/session_replay/scenarios/desk.expect.yaml --out /tmp/desk-timeline.jsonl
pytest tests/session_replay
```

For label-free scene_lab checks, install `tests/scene_lab` and add
`--invariants`. One command can replay several scenarios into one run:

```sh
python -m session_replay run tests/session_replay/scenarios/*.jsonl \
  --invariants --runs-root /tmp/scene-lab-runs
```

Each scenario uses its sibling `*.expect.yaml` and its recorded `llm` mode
when present; `--llm` overrides that mode. `--expect` is allowed with one
scenario. Expectation failures still set the exit status independently of
invariant findings. The run contains `bugs.jsonl`, `bugs.md`, and a trace and
report per scenario. `--thresholds PATH` selects a threshold YAML;
`--tt2-judge` opts into Claude question ratings.

The expect file may contain `llm: recorded`, an `interpretations` map keyed
by exact utterance text, an `expect` list, and a `never` list. Each expectation
anchors on `after: {heard: "..."}` or `after: {person: {state: standing,
zone: bed}}`. `events` are matched as an ordered subsequence of agent
publications following the first matching input, through `within_s` seconds.
Each event match can name `type` and any timeline field, such as `to_goal`,
`state`, `strategy`, or `text`. `absent` lists event patterns forbidden in that
same anchored window; `never` applies to the whole replay. Failures
print the matching window and exit 1. Every scenario with a sibling expect
file is also run by the parametrized pytest regression suite.

Promoted scenarios may also contain an observed `InterpretationMap` row with
the same mapping. The runner uses it in recorded mode, with the expect file's
mapping taking precedence when both are present.

`--llm none` is the default without an expect file and exercises the agent's
deterministic fallbacks. `--llm recorded` uses the interpretation map,
defaults unknown speech to `unclear` with zero distress, and lets compose and
plan fall back to the agent's own code. `--llm live` uses the same local
client constructor as the service; `--backend`, `--model`, and `--base-url`
select its server. `--strategies` and `--person` override the agent's normal
configuration path resolution. `--tail-s` changes the post-input simulation
duration.

`--llm-latency none` (default) gives calls zero simulated time. Use
`--llm-latency recorded` for per-call durations from extracted agent LLM
Activity, matched in order by kind. Missing durations use
`--llm-latency-fallback 2.5` seconds by default and warn with the scenario
name; the checked-in desk scenarios have no duration rows. Alternatively,
`--llm-latency fixed:2.5` advances every interpret, compose, and plan call by
2.5 seconds. Inputs during a call arrive after it ends, and the trace records
the advanced output time and Activity duration.

A promoted draft can use two additional keys on an anchored expectation:
`delay_s` starts matching that many seconds after the anchor (for example,
settling in bed), and `min_spacing_s` requires consecutive `Say` events in
the window to be at least that many seconds apart. Both default to unset;
existing expectation files behave as before.
