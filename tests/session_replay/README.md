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
and `Notify` as `observed: true` reference rows. It drops speech starts,
health, activity, pose debug, frames, and audio. The runner ignores observed
rows. Review the extracted text before committing; raw exports stay in
gitignored `data/`, and only extracted, text-only scenarios belong in git.

Write a sibling `desk.expect.yaml` with recorded interpretations and ordered
expectations, then run:

```sh
python -m session_replay run tests/session_replay/scenarios/desk.jsonl \
  --expect tests/session_replay/scenarios/desk.expect.yaml --out /tmp/desk-timeline.jsonl
pytest tests/session_replay
```

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

`--llm none` is the default without an expect file and exercises the agent's
deterministic fallbacks. `--llm recorded` uses the interpretation map,
defaults unknown speech to `unclear` with zero distress, and lets compose and
plan fall back to the agent's own code. `--llm live` uses the same local
client constructor as the service; `--backend`, `--model`, and `--base-url`
select its server. `--strategies` and `--person` override the agent's normal
configuration path resolution. `--tail-s` changes the post-input simulation
duration.
