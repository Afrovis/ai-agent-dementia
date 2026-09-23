# Eyes upgrade handoff

Execution brief for [EYES_UPGRADE_PLAN.md](EYES_UPGRADE_PLAN.md). Read the plan
first; this file says where the code goes and what the exact numbers are. Line
numbers were taken on 2026-09-22 from `main` and will drift; search for the
named symbol if a line has moved.

## Design source

Canvas: <https://claude.ai/artifact/BYpWLiTJDJxLFYWgyDmpWo> ("Night Companion").
Read artboards with the Artifact tool (`read`, `path: "project/<name>"`). The ones
this work uses:

| Artboard | What to take from it |
| --- | --- |
| `Expressions.dc.html` | Eye shapes (`eye()` function) for open, sleeping, sleepy. |
| `Warm-RedMono.dc.html`, `Night-Nightlight.dc.html` | Red mono palette, top bar, zzz, breathe animation. |
| `Track-Eyes.dc.html` | Gaze transform (translate plus near-eye scale). |
| `Listen-Demo.dc.html`, `Listen-Cues.dc.html` | Listening glow and fast blink. Ignore the tilt, bob, thinking cue and speaking bounce. |
| `Glow-Vignette.dc.html` | Alert vignette. Ignore its eyes, text panel and buttons. |

The canvas is a design prototype, not production code. Port the values into the
existing plain HTML/CSS/JS page; do not copy the `DCLogic` runtime.

## Exact values

### Palette (red mono)

```
eye    #B8322A   bg     #070202   panel  #120605   border #230C0A
text   #D98A80   muted  #8A4A43   accent #C9463B
```

### Eye shapes (scale 1; `c` = eye colour)

- **open**: 150 x 200 px, `border-radius: 999px`, filled `c`,
  `box-shadow: 0 0 48px c88, 0 0 12px c`.
- **sleeping**: 150 x 62 px, `border: 11px solid c`, `border-top: 0`,
  `border-radius: 0 0 999px 999px`, transparent fill, `opacity: .7`,
  `filter: drop-shadow(0 0 16px c99)`. Breathe animation on the pair.
- **sleepy**: 160 x 72 px, filled `c`, `border-radius: 14px 14px 999px 999px`,
  `margin-top: 50px`, `opacity: .85`, same box-shadow as open.
- **listening**: open shape at 1.08 x.
- **speaking**: open shape.
- The pair sits in a row with a 140 px gap, 240 px tall, centred in the face area.
  Scale everything with the viewport (the canvas is 1440 x 900) rather than using
  fixed px.

### Animations

```css
@keyframes breathe   { 0%,100% { transform: scale(1);    opacity: .55 } 50% { transform: scale(1.04); opacity: .8 } }  /* 5s, sleeping */
@keyframes blink     { 0%,92%,100% { transform: scaleY(1) } 95% { transform: scaleY(.08) } }                        /* 6s, open/speaking */
@keyframes blinkFast { 0%,84%,100% { transform: scaleY(1) } 90% { transform: scaleY(.08) } }                        /* 2.4s, listening */
@keyframes zfloat    { 0% { transform: translate(0,0); opacity: 0 } 25% { opacity: .7 } 100% { transform: translate(40px,-90px); opacity: 0 } }  /* 4s */
@keyframes alertGlow { 0%,100% { opacity: .15 } 50% { opacity: .40 } }                                             /* 4.5s */
```

- Zzz: three `z` in `accent`, sizes 44/32/24 px, delays 0/1.3/2.6 s, placed
  top-right of the eye pair as in `Night-Nightlight.dc.html`.
- Suppress blinking for 2 s after any expression change.
- Expression changes crossfade over 1.5 s (animate size, radius, opacity and glow;
  the sleeping shape is a border arc, so crossfade two layers rather than morphing).

### Glow levels (group `filter: drop-shadow(0 0 Npx eye)`)

| State | N |
| --- | --- |
| open / sleepy | 14 |
| sleeping | 6 |
| listening | 26 + 22 x amp |
| speaking | 14 + 16 x amp |

`amp` is the smoothed loudness in [0, 1] (see Voice pulse). Size pulse is
`scale(1 + 0.03 x amp)` for listening and speaking only.

### Gaze

- Input `gx, gy` in [0, 1] image coordinates from the `eyes` message.
- Mirror: `sx = 1 - gx`. Position `pos = clamp((sx - 0.5) * 2, -1, 1)`.
- Transform on the pair: `translate(pos * R px, |pos| * 10 px)` with
  `R = 220 px` at 1440 px width (scale with viewport). Near eye
  `scale(1 + 0.14 * pos)`, far eye `scale(1 - 0.14 * pos)` (left eye gets `+`).
- Smoothing, run in `requestAnimationFrame`:
  - critically damped spring toward the target, settling in about 1.5 s;
  - velocity capped at 0.15 of the screen width per second;
  - dead zone: ignore target changes smaller than 0.03 in `pos`;
  - `target: none`: hold the last position for 5 s, then ease to centre over
    3 s and dim to 75% brightness.
- `prefers-reduced-motion: reduce`: no gaze motion, eyes stay centred.

### Alert vignette

A fixed full-screen layer above everything, `pointer-events: none`,
`box-shadow: inset 0 0 180px 40px #C9463B`. On: opacity ramps 0 to 0.15 over
3 s, then runs `alertGlow` (4.5 s, ease-in-out). Off: fade to 0 over 3 s from
wherever it is. Reduced motion: steady 0.30, still with the 3 s fades.

### Top bar

Padding 28 px 56 px, 22 px text in `muted`, following `Warm-RedMono.dc.html`.

- Left: 22 px stroke icon plus time. Moon path
  `M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z`; draw a matching sun (circle
  plus eight short rays, same stroke). Icon swap crossfades over 2 s. Time
  updates each minute.
- Right: 10 px dot in `accent` plus label. Opacity of the dot: 1 for Needs
  attention / Camera off / Microphone off, 0.8 for Speaking / Listening, 0.5 for
  Winding down, 0.35 for Resting · monitoring, and a slow 2 s opacity blink for
  Reconnecting. Label changes crossfade over 0.6 s.
- Status priority, highest first: Needs attention (alert on) > Camera off /
  Microphone off (track ended, permission denied, or no bridge) > Reconnecting
  (`/ws` closed) > Speaking (audio playing) > Listening > Winding down (sleepy) >
  Resting · monitoring (sleeping) > nothing.

### Font

The canvas uses Atkinson Hyperlegible (SIL OFL). The bedside page must work
with no internet, so vendor the 400 and 700 woff2 files into
`services/embodiment/embodiment/static/fonts/` with the licence file, and keep a
system sans fallback. Do not load it from Google Fonts at runtime.

## Work items

Do them in this order; each ends with its tests green.

### 1. `shared`: the `Gaze` event

`shared/nc_shared/events.py`, next to `PersonState` (around line 111):

```python
class Gaze(Event):
    target: Literal["face", "bed", "none"]
    x: float | None = None  # 0..1, image coordinates, origin top-left
    y: float | None = None
```

Register it on a new `gaze` stream in the event registry (around line 238).
Validate `x`/`y` in [0, 1] and require both when `target != "none"`. Add a test
in `shared/tests`. Run `python -m nc_shared.archdoc --write` from the repo root
and commit the regenerated `ARCHITECTURE.md` together with the change.

### 2. `perceive`: publish `Gaze`

- The pose (`PoseResult` with `landmarks["nose"]` and normalised `bbox`,
  `services/perceive/perceive/backends.py:60-89`) is available in the main loop
  at `services/perceive/perceive/main.py:363`. `PersonState` is published at
  `main.py:472-479` (state change) and `main.py:522-529` (heartbeat).
- Put the logic in a small pure function (new module `perceive/gaze.py`):
  - state `absent`, or no pose: `target="none"`;
  - state `in_bed`, and a `bed` polygon exists: `target="bed"`, point = polygon
    centroid (`zones.py:39,96` hold polygons as normalised `(x, y)` lists);
  - otherwise `target="face"`: nose if its visibility is at least 0.5, else
    `((x_min + x_max) / 2, y_min + 0.1 * (y_max - y_min))`.
- Publish when the target changes or the point moves more than 0.03 in either
  axis since the last publish, and alongside every heartbeat.
- Unit-test the function (no model, no Redis) including: nose hidden, no bed zone
  while `in_bed` (fall back to face), absent, sub-threshold jitter.

### 3. `embodiment` server: eyes state

- Streams are listed in `broadcast_loop()` at
  `services/embodiment/embodiment/app.py:210-226`. Add `person`, `gaze`,
  `session`, `notify`, `ack`. `speech_in` is already read but `Utterance` is
  filtered out (around line 235); use it now to end listening.
- New pure module `embodiment/eyes.py` holding an `EyesState` that takes events
  and a clock and yields `{"expression", "alert", "gaze"}`:
  - expression from posture: `in_bed` -> `sleeping`, `sitting_up` -> `sleepy`,
    everything else -> `open`;
  - `SpeechStarted` -> `listening` until the next `Utterance`, or 15 s after
    the last `SpeechStarted` if no `Utterance` comes;
  - `Show.face == "listening"` -> `listening`, `"speaking"` -> `speaking`;
    `asleep`/`awake` are ignored for the expression (see plan, `Show.face`);
  - alert: on when a `SessionState` enters `ESCALATED`; remember the ids of
    `Notify` messages seen from 10 s before that entry until it ends; off
    when an `Ack` names one of those ids, or when the phase leaves
    `ESCALATED`. `Ack.notify_id` is the Redis stream id of the `Notify`
    (`shared/nc_shared/events.py:199-202`, produced by the dashboard at
    `services/dashboard/dashboard/app.py:651`). Check that the embodiment bus
    loop exposes the message id; if the wrapper hides it, extend the wrapper in
    `nc_shared`, not the service.
  - gaze: latest `Gaze` as is.
- Send a new WebSocket message whenever the state changes:
  `{"type": "eyes", "expression": ..., "alert": bool, "gaze": {"target", "x", "y"}}`.
  Replay the latest one on connect in `ConnectionManager.send_current_state()`
  (`app.py:115-118`, which today replays only the last `Show`).
- Page settings go in a `{"type": "config", ...}` message on connect:
  `night_start`, `night_end`, `clock_24h`. Environment variables, read in
  `services/embodiment/embodiment/main.py` next to the others (around line 57):
  `EMBODIMENT_NIGHT_START=20:00`, `EMBODIMENT_NIGHT_END=07:00`,
  `EMBODIMENT_CLOCK_24H=false`. Add them to `.env.example`.
- Tests in `services/embodiment/tests`: the whole `EyesState` table (posture,
  listening timeout, `Utterance` ends listening, `Show.face` mapping, alert on,
  Ack with matching id turns off, Ack with other id does not, phase change
  turns off, Notify arriving just before the phase), plus a WebSocket test that
  a new client receives the last `eyes` and the `config` message.

### 4. `embodiment` page

Files: `services/embodiment/embodiment/static/index.html`, `script.js`,
`style.css`.

- Replace the `.face` circle, dot eyes and mouth with the eye pair, the top bar,
  the zzz and the vignette layer. Keep the headline/body text, photos and
  everything about `say` and `speech_started`.
- `FACE_STATES` (`script.js:17`) and the old `applyShow` face handling: `show`
  still sets text, photo and brightness; the expression comes from `eyes`.
- Speaking: set on `play` of the `Audio` created at `script.js:69`, cleared on
  `ended`, `pause` and `error`. Speaking outranks the server's expression.
- Voice pulse through Web Audio:
  - create one `AudioContext` after `getUserMedia` succeeds
    (`script.js:372`), which follows the user's permission click; call
    `resume()` there and on the first pointer event;
  - mic: `createMediaStreamSource(stream)` -> `AnalyserNode`; read it only
    while the expression is `listening`, otherwise `amp` for listening is 0;
  - speech: `createMediaElementSource(audio)` -> analyser -> destination,
    **only if `ctx.state === "running"`**. A media element routed into a
    suspended context plays silently, so otherwise play it plainly and leave
    `amp` at 0;
  - loudness = RMS of the time-domain buffer, mapped to [0, 1] with a noise
    floor; smooth with a 80 ms attack and 300 ms release; update in the same
    `requestAnimationFrame` loop as the gaze.
- Camera off / Microphone off: from `getUserMedia` failure and each track's
  `ended` event. Reconnecting: from the `/ws` close handler in `connect()`
  (`script.js:127-147`).
- Put the pure page logic (status priority, gaze mapping and spring step,
  loudness smoothing, day/night from a time) in one plain script with no DOM
  access, so it can be checked in isolation.

### 5. Docs

Update `CLAUDE.md` where it describes the embodiment page: the eyes and their
states, the new streams embodiment reads, the three new environment variables,
and that the voice pulse needs the permission click. Regenerate
`ARCHITECTURE.md` if anything in item 3 changed the streams again.

## Commands

```sh
# shared
docker run --rm -v "$PWD":/repo -w /repo python:3.12-slim \
  sh -c "pip install -q -e shared[dev] && pytest shared/tests"
python -m nc_shared.archdoc --write

# perceive and embodiment (same pattern for each)
docker compose build perceive embodiment
docker compose run --rm --no-deps perceive \
  sh -c "pip install -q pytest ruff && pytest -q && ruff check . && ruff format --check ."
docker compose run --rm --no-deps embodiment \
  sh -c "pip install -q pytest ruff && pytest -q && ruff check . && ruff format --check ."
```

Visual check with the `headless-browsing` skill against a locally served page:
one screenshot per expression, the alert glow, a day and a night top bar, and a
synthetic gaze sweep (send `eyes` messages with `x` stepping 0 to 1) to confirm
the eyes lag smoothly and never jump.

## Hand-test on the real device

1. `docker compose up --build`, open the page over the tailnet name, grant
   camera and microphone.
2. Lie in bed: sleeping eyes, zzz, looking toward the bed. Sit up: sleepy.
3. Stand and walk left to right: open eyes follow you, not your mirror image,
   and never jump.
4. Speak: listening glow, fast blink, pulse follows your voice. The reply
   pulses with the companion's speech and is always audible.
5. Publish a test escalation (or trigger one with replay): glow fades in, status
   says Needs attention. Acknowledge on the dashboard: glow fades out.
6. Unplug the network: status says Reconnecting; eyes keep their last state.
