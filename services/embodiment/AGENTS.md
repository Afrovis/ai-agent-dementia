# embodiment

The fullscreen bedside page and its server. Everything here is what the person
sees and hears at night, so check changes on a real page, not only in tests.

## Speech

`embodiment` turns `Say` events into local Piper WAV files and serves them to
the page under `/speech/<opaque-id>.wav`; speech bytes never enter Redis. The
image downloads `en_US-lessac-medium` at build time. Startup pre-renders the
configured fixed strategy phrases into the ephemeral `PIPER_CACHE_DIR`;
built-in greetings say night-time, while a caregiver template that still uses
`{time_words}` is warmed in the spoken night phrase variants. Generated speech is not
kept under `data/`. To rebuild and exercise the real voice path:

```sh
docker compose build embodiment
docker compose run --rm --no-deps embodiment \
  sh -c "pip install -q pytest ruff && pytest -q && ruff check . && ruff format --check ."
```

Barge-in: `listen` publishes `SpeechStarted` on `speech_in` (see
`services/listen/AGENTS.md`); embodiment forwards it, and the page stops audio
only when `Say.interruptible` is true. The page requests browser/OS echo
cancellation from `getUserMedia`.

If the browser blocks autoplay, a "Tap to turn on the voice" button appears so a
person can unlock and retry a recent clip.

## The eyes

Design and exact values are in `docs/EYES_UPGRADE_PLAN.md` and
`docs/EYES_UPGRADE_HANDOFF.md`. Embodiment picks the expression itself, as a
reflex, in `embodiment/eyes.py`:

- `in_bed` and `sitting_up` are sleepy for a 25 s doze (`DOZE_SECONDS`) while
  the page slowly lowers the lids, then sleeping (with floating z's). Sitting up
  from lying down, or the end of listening or a face override, restarts the doze.
  Anything else is open.
- `SpeechStarted` means listening until the next `Utterance` or 15 s;
  `Show.face` listening/speaking overrides posture for at most 15 s. The page
  switches to speaking while speech audio plays.
- The alert vignette comes on when the session enters `ESCALATED` and fades when
  the caregiver acknowledges that escalation's `Notify` or the phase changes.
  For this, embodiment also reads `person`, `gaze`, `session`, `notify` and `ack`.
- `perceive` publishes `Gaze` on the capped, unpersisted `gaze` stream; the eyes
  follow it mirrored (`1 - x`) through a speed-capped spring.
- The voice pulse uses Web Audio, which starts only after the camera and
  microphone permission click; without it the eyes stay still but speech plays.
- The top-bar clock uses `EMBODIMENT_NIGHT_START`, `EMBODIMENT_NIGHT_END` (moon
  between them, sun otherwise) and `EMBODIMENT_CLOCK_24H`.

The pure page logic has a browser test,
`services/embodiment/tests/js/eyes_logic_test.html`: open it in any browser (or
headless Chromium with `--allow-file-access-from-files --dump-dom`) and the
title reads `PASS <n>` or `FAIL: ...`. `?demo=1` on the page URL exposes
`window.__eyesDemo.receive(msg)` and `setAmp(x)` for driving it by hand.

## Debug overlay and desk-test controls

Press `D` (or open `?debug=1`) to toggle the overlay: local camera preview with
pose, person and session state, media and model activity, Say playback events,
this page's identity and audio state, the number of connected bedside pages,
and a short event log. The `pose_debug` and `activity` streams are capped
telemetry that `store` never persists.

`EMBODIMENT_DEBUG_CONTROLS=true` (then restart `embodiment`) adds operator
controls: shift the agent's time of day by whole hours (3 a.m. at noon) or
reset it, force every `PersonState` to count as in bed, and, when perceive
reports no bed zone, a "Detect bed zone" button that runs `calibrate_bed` on
about 20 live frames, writes `zones.yaml` and reloads `perceive` without a
restart. These travel as `DebugControl`, `CalibrateBed` and `BedZoneStatus` on
the `debug` stream. The offset moves only the night window and spoken time,
never session timers. Overrides live in the agent's memory, so restarting
`agent` clears them; an amber badge shows while any is active. The page has no
login, so leave the flag off anywhere but a desk test.

## Photos and familiar voice

Demo photos ship in `embodiment/demo_photos` and resolve at
`/photos/demo_family` and `/photos/demo_room`. Caregiver uploads under
`PHOTO_DIR` shadow a demo photo with the same id.

`familiar_voice` is disabled by default. Upload a consented PCM WAV on the
dashboard Media page, copy its displayed id into that strategy's `clip_id`,
enable it, and restart `agent`. `agent` and `embodiment` must share
`VOICE_CLIP_DIR` (compose default `/app/data/voice-clips`). It never falls back
to Piper or cloned speech when the clip is unavailable.
