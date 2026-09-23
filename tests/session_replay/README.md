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

`extract` keeps `PersonState` from `person`, `Utterance` from `speech_in`,
operator `DebugControl` requests and `ResetSession` requests from `debug`.
It also retains `SessionState`, `GoalChanged`, `Say`, `Show`, `LightCommand`,
`Notify`, and agent `DebugControl` echoes as `observed: true` reference rows.
It drops speech starts,
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

The expect file may contain `llm: recorded`, `interpretations` and `plans`
maps keyed by exact utterance text, an `expect` list, a `never` list, and an optional
`known_bug` reason. Each expectation anchors on `after: {heard: "..."}`,
`after: {person: {state: standing, zone: bed}}`,
`after: {debug: {force_in_bed: false}}`, or `after: {reset_session: true}`.
Debug anchors may name either or both override fields. Add `occurrence: N`
inside `after` to select the Nth matching input (1-based; default 1).
`events` are matched as an ordered subsequence of agent
publications following the selected input, through `within_s` seconds.
Each event match can name `type` and any timeline field, such as `to_goal`,
`state`, `strategy`, or `text`. `absent` lists event patterns forbidden in that
same anchored window; `never` applies to the whole replay. Failures
print the matching window and exit 1. Every scenario with a sibling expect
file is also run by the parametrized pytest regression suite. A `known_bug`
scenario is marked strict xfail there: the suite stays green while the bug
reproduces and fails if the bug is fixed before the marker is removed. The CLI
still exits 1 and prints the reason when its expectation fails.

`--llm none` is the default without an expect file and exercises the agent's
deterministic fallbacks. `--llm recorded` uses the interpretation map,
defaults unknown speech to `unclear` with zero distress, and returns a plan
for the latest interpreted utterance when `plans` contains one. Plan values
use the agent's fields, for example `"Thank you.": {next_strategy: guided_return}`
or `"I need the restroom": {goal_change: restroom}`; omitted confidence
defaults to 1.0. Without a matching plan, and for composition, the agent's
own fallbacks apply. `--llm live` uses the same local
client constructor as the service; `--backend`, `--model`, and `--base-url`
select its server. `--strategies` and `--person` override the agent's normal
configuration path resolution. `--tail-s` changes the post-input simulation
duration.
