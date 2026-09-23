# Eyes upgrade plan

Date: 2026-09-22. Execution brief: [EYES_UPGRADE_HANDOFF.md](EYES_UPGRADE_HANDOFF.md).

Replace the bedside face (circle, dot eyes, mouth) with the glowing eyes from
the "Night Companion" design canvas
(<https://claude.ai/artifact/BYpWLiTJDJxLFYWgyDmpWo>), make them react to
posture, speech and caregiver alerts, and let them gently follow the person.
The canvas is the visual source of truth; the artboards that matter are listed
in the handoff.

## Goals for this round

1. Four eye expressions: **sleeping**, **sleepy**, **listening**, **speaking**,
   plus plain **open** eyes as the resting shape when none of those apply.
2. Gaze that gently follows the person's face, or rests on the bed while the
   person is in bed. Never sudden movement.
3. A soft pulse driven by the voice: the person's voice while listening, the
   companion's own speech while speaking.
4. A caregiver-alert glow around the screen edge while a caregiver alert is
   open.
5. A small top bar: clock with moon (night) or sun (day) on the left, the most
   important status on the right. Floating "z z z" while sleeping.

## Decisions taken

| Topic | Decision |
| --- | --- |
| Palette | Red mono from the canvas: eye `#B8322A` on `#070202`, accent `#C9463B`. |
| Who picks the expression | Embodiment, as a local reflex on posture and speech. The agent needs no changes. |
| `Show.face` | The agent sends `asleep` in IDLE/COOLDOWN and `awake` in OBSERVING/ESCALATED, so it cannot outrank posture without killing the sleepy state. Posture decides. `Show.face` `listening`/`speaking` still map to those expressions. Headline, body, photo and brightness from `Show` keep working. |
| Standing, walking, empty room, `on_floor` | Plain open eyes. Following when someone is visible, centred and dimmed when nobody is. No alarmed face at someone who may have fallen. |
| Listening | Eyes about 8% larger, brighter glow, blink every 2.4 s, soft pulse with the person's voice. No head tilt, no bob. |
| Speaking | Open eyes, glow and size pulse with the speech being played (at most 3% size). No bounce on syllables. |
| Alert glow | Inset vignette in `#C9463B`. Fades in over 3 s, then pulses slowly between about 15% and 40% opacity every 4.5 s. Steady at about 30% under reduced motion. |
| Alert on/off | On when the session enters `ESCALATED`. Off at the first of: the caregiver acknowledges that alert's `Notify` (dashboard `Ack`), or the session leaves `ESCALATED`. Fades out over 3 s. |
| Clock | Moon icon from 20:00 to 07:00, sun otherwise (configurable). 12-hour `2:14 AM`, configurable to 24-hour. Uses the bedside device clock. |
| Status (upper right) | One label, by priority: Needs attention, Camera off / Microphone off, Reconnecting, Speaking, Listening, Winding down (sleepy), Resting · monitoring (sleeping), nothing for open eyes. |
| Zzz | Three small z's float up on a 4 s cycle, only with the sleeping eyes. |
| Gaze mirroring | The webcam image is not mirrored, so the eyes use `1 - x` to look toward the person. |

## Out of scope for this round

Happy, sad and alert eye expressions; the "thinking" cue; the "To caregiver"
panel and the caregiver action buttons from the canvas; amber, ember and red
palettes; software echo cancellation.

## Shape of the change

- **shared**: one new event, `Gaze`, on a new `gaze` stream. It carries only
  normalised coordinates and a target (`face`, `bed`, `none`), never image data.
- **perceive**: publishes `Gaze` next to `PersonState`. In bed, the point is
  the bed polygon's centroid. Otherwise it is the nose landmark, falling back
  to the top of the person's box. Sent only when the point moves more than 3%
  or the target changes, plus on the existing heartbeat.
- **embodiment (server)**: subscribes to `person`, `gaze`, `session`, `notify`
  and `ack` in addition to what it reads today. A pure module turns those into
  an eyes state (expression, alert on/off, gaze target), sent to the page as a
  new `eyes` WebSocket message and replayed to pages that reconnect.
- **embodiment (page)**: new eyes markup and CSS following the canvas, a
  spring-smoothed gaze, loudness-driven pulse through Web Audio, the alert
  vignette, top bar and zzz. Speaking and the connection or device statuses
  are known only on the page, so the page does the final composition.

## Safety and comfort rules

This screen faces a person with dementia at night. Every rule below is a
requirement, not a polish item.

- No step changes in brightness, position or shape. Everything ramps.
- Gaze speed is capped. Jitter below the dead zone never moves the eyes.
- Alert glow stays soft and slow. It must never flash.
- `prefers-reduced-motion` turns off gaze motion, the voice pulse and the zzz
  drift, and holds the alert glow steady.
- If Web Audio is not running, speech must still be heard. The pulse is the
  part that degrades, never the voice.

## Open points for the owner

- "Needs attention" is readable by the person too. "Caregiver called" is the
  gentler alternative if it proves worrying.
- 12-hour clock is the default, following the canvas.

## Verification

- pytest for the new event, the perceive gaze logic, the embodiment eyes state
  machine and the new WebSocket message, run in the service containers.
- `ARCHITECTURE.md` regenerated and the shared tests pass.
- A headless-browser pass that screenshots each expression, the alert glow and
  a synthetic gaze sweep with synthetic loudness.
- Owner's real-room check: walk left to right and the eyes follow you; sit up
  in bed and they turn sleepy; talk and they listen and pulse; the reply
  pulses; a test escalation brings the glow, and acknowledging it fades it.
