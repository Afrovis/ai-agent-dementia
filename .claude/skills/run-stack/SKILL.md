---
name: run-stack
description: Start the Night Companion docker compose stack locally, check that the media bridge, bus streams, bedside page and caregiver dashboard are alive, drive it by hand (test alerts, desk-test time shift), and take it down again. Use when asked to run, start, restart or screenshot the app, to confirm a dashboard or embodiment change works in the real page, or to debug "nothing is happening" on the bedside page.
---

# Run and check the local stack

Commands run from the repository root (in a worktree, that worktree's root).
The stack shares host Ollama with `tests/scene_lab`'s `nightsim` project, so
check neither is already running before starting the other:

```sh
docker compose ls
```

## Start

```sh
test -f .env || cp .env.example .env
docker compose up --build -d
docker compose ps
```

Set `DASHBOARD_PASSWORD` in `.env` first if you need the dashboard; without it
every dashboard route answers 503. `EMBODIMENT_DEBUG_CONTROLS=true` adds the
desk-test controls (time shift, force in-bed, detect bed zone) to the bedside
debug overlay; it has no login, so turn it off again afterwards.

## Check it is alive

- Pages: bedside on `https://<tailnet-name>:8443` (or `http://localhost:8443`
  without certificates), dashboard on `:8444` with Basic auth. To look at them
  without a browser tool, use the `headless-browsing` skill and read the
  screenshot. `?debug=1` on the bedside page shows the debug overlay; `?demo=1`
  exposes `window.__eyesDemo` for driving the eyes.
- Page logic without the stack: open
  `services/embodiment/tests/js/eyes_logic_test.html` headless with
  `--allow-file-access-from-files --dump-dom`; the title reads `PASS <n>`.
- Bridge: once a page has granted camera and microphone, these climb above
  zero. They stay at zero with no page attached, which is not a fault. Both
  cap at 50; a value below 50 that never moves is.

  ```sh
  docker compose exec bus redis-cli XLEN frames_raw
  docker compose exec bus redis-cli XLEN audio_in
  docker compose exec bus redis-cli XLEN frames
  ```

- Logs are one JSON line per event with a `service` field:
  `docker compose logs --tail 50 <service>`.

## Drive it by hand

Send a test caregiver alert (`source` is required). With `NTFY_URL` empty it
lands in `docker compose logs notify`:

```sh
docker compose exec store python -c "
import redis
from nc_shared.bus import Bus
from nc_shared.events import Notify
Bus(redis.Redis.from_url('redis://bus:6379')).publish(
    Notify(source='manual-test', level='attention', title='test',
           body='hello', repeat_until_ack=False))
"
```

Replay recorded bus traffic instead of live hardware:

```sh
docker compose exec store python -m nc_shared.replay play redis://bus:6379 /app/data/rec.jsonl --speed 10
```

Config saved from the dashboard does not hot-reload: restart `perceive` after
zones, `agent` after profile or strategies, and `embodiment` after strategies.

## Finish

Report what you checked and how (screenshot, stream counts, logs), and say so
plainly if a check could not be done, for example no camera attached. Then:

```sh
docker compose down
```

If running the stack taught you something this file or `AGENTS.md` gets wrong,
fix it in the same change.
