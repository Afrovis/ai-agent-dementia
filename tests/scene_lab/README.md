# scene_lab

Install into the local test environment:

```sh
/Users/mathiasserver/.claude/jobs/e96c9b56/tmp/venv/bin/pip install --no-build-isolation --no-deps -e tests/scene_lab
```

Check a bus export with `python -m scene_lab check export.jsonl --format export`, a normalized trace with `--format trace`, or JSON agent logs with `--format agent_log`. Compare runs with `python -m scene_lab bugs RUN_ID [RUN_ID ...]`.

Runs go to `SCENE_LAB_RUNS` if set; otherwise to the sibling `data-ai-agent-dementia/analysis/scene-lab/runs` directory. Each run contains scene traces, JSON and Markdown reports, and an append-only `bugs.jsonl`.

`agent_log` is best effort: the current agent log omits Say text and usually has no timestamp, so the adapter uses line order as its time axis. Bus exports provide the timing evidence for TT and TM checks. SM-5 reports an info skip unless the trace supplies the night-window, profile avoid-list, and restroom-resolution facts in `meta`.

## Scripted fake-live scenes

The three starter cards in `scenes/` were converted from `restroom-01`,
`disorientation-01`, and `conversation-01`. Install `scene_lab`, its `live`
extra for Piper, and the repository's shared, agent, and decision bench packages
in the same Python environment. Set `SCENE_LAB_PIPER_DIR` to a directory with
`en_US-amy-medium.onnx` and its matching JSON file. The simulation uses Amy
for the person and keeps synthesized audio in memory.

```sh
python -m scene_lab from-bench restroom-01
python -m scene_lab run tests/scene_lab/scenes/restroom-01.yaml
python -m scene_lab inprocess tests/scene_lab/scenes/restroom-01.yaml --out /tmp/restroom-inprocess.jsonl
python -m scene_lab diff /path/to/live/restroom-01/trace.jsonl /tmp/restroom-inprocess.jsonl
```

`run` accepts `--no-stack-up`, `--keep-stack`, `--runs-root PATH`, and
`--run-dir PATH`. It builds the isolated `nightsim` compose project, uses Redis
only on port 16379, and records `ollama ps`, other agent stacks, and the Git
commit in `report.json`. The `inprocess` command uses decision bench's
`StubLLM`, so its diff describes both service timing and model differences.
The live run needs Docker, a reachable Ollama model, and a working Piper voice.
The coordinator should check that no manual agent stack or concurrent Ollama
client is active, build the stack at the recorded commit, and inspect the
resulting playback Activity sequence beside a real browser page once.

Each scene folder holds `scene.yaml`, a text-only bus `export.jsonl`,
`agent.log`, `mind.jsonl`, `trace.jsonl`, and reports. The run root has
`bugs.jsonl` and `bugs.md`. Audio is never written to disk.

## Claude persona scenes

Five `persona-*.yaml` cards use `mind: claude`. They use the logged-in
`claude -p --model sonnet` subscription through decision bench's isolated
runner. No API key, tools, or repository working directory is passed to that
call. Install `decision_bench` in the same environment, sign in to the Claude
CLI, and run a card with the `scene_lab run` command above. `mind_silence_s`
defaults to 20 seconds and can be set in a card. Each scene remains capped at
600 seconds. `mind.jsonl` contains each call's trigger, elapsed time, latency,
validated beats, note, and every scripted or generated person line. `report.json`
adds `mind_latency_p50_s` and `mind_latency_p95_s` when calls occurred. If both
validation attempts fail, the scene ends and `bugs.jsonl` records a harness
origin `mind failure` entry. These scenes require the live stack and Piper;
the unit tests inject a fake Claude runner and need neither.
