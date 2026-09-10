# Recording protocol for tier 3 (the real perception bench)

Tier 3 is currently empty. This system's milestone 1 is not done, per
HANDOFF.md section 8, until someone records this. This is written to be
followed in an afternoon, once consent is in place.

## Consent, first

PLAN.md section 11: the volunteer must actually consent to being recorded
on infrared camera in the room, doing these actions. Explain what the
footage is for (measuring `perceive`'s accuracy and latency), who will see
it (whoever reviews the bench results), and that it is never uploaded
anywhere (HANDOFF.md rule 2: camera frames never leave the box unless a
caregiver explicitly enables a debug option; this recording is that
explicit case, done on purpose, once, for this bench). Get a clear yes
before the camera is on. If the intended volunteer is the person the
system is ultimately for and they cannot meaningfully consent, this
protocol is not for them -- use a different consenting adult standing in,
per PLAN.md section 11's guidance on caregiver-decided consent applying to
the real deployment, not to this recording.

## What to record

Set up the camera exactly where it will live in the real room: same
mount point, same field of view, same IR illuminator, same bed and door in
frame. This is the whole point of tier 3 over tier 2 -- it has to be *this*
room, at night, in IR, or it is not measuring what the system will actually
see.

Lighting: the room as it will actually be at night -- IR illuminator only,
no lamp, no hallway light spilling in, unless that is genuinely how the
room will be used. Note whatever the real condition is in the manifest;
see below.

One continuous recording, or several short ones -- either works, since the
manifest (see `perception_bench/infrared.py`'s docstring) is per-clip. A
single ~10-15 minute take covering the full script below once is enough to
start; more takes, on more nights, only make the bench stronger.

## The script

These are PLAN.md section 12's scripted actions, run in this order, with
pauses in between so each state is unambiguously held for a few seconds
(this makes labelling easier and gives the hysteresis in `StateTracker`
frames to actually confirm on):

1. **Lie in bed**, under the covers, still, for about 30 seconds. This is
   `in_bed`.
2. **Sit up** in bed, for about 10 seconds. `sitting_up`.
3. **Stand up**, next to the bed, for about 10 seconds. `standing`.
4. **Walk to the door**, at a normal pace, then stop. A few seconds of
   `walking`, ending in `standing` or leaving frame (`absent`) depending on
   whether the door is in frame.
5. Return to the middle of the room and **lie down on the floor** (a mat or
   folded blanket on the floor for comfort is fine and does not need to be
   noted specially -- it is not the bed). Hold for about 15 seconds.
   `on_floor`.
6. **Get up and walk out of frame** entirely, through the door. Hold out of
   frame for about 10 seconds. `absent`.

Repeat the whole script two or three times if the volunteer is willing;
repeats do not need to be identical (a slower stand-up on the second pass
is useful, not noise).

## Labelling

While recording, either:

- Call out each action out loud right as it starts ("sitting up... now"),
  and note the wall-clock or recording timestamp, or
- Have a second person with a stopwatch/notes app mark each transition
  time as it happens.

Afterwards, write down the timestamp (in seconds from the start of the
clip) each action began. That is directly what the manifest's
`timeline[].from_s`/`to_s` need -- see the worked example at
`tests/perception_bench/fixtures/ir_manifest.example.yaml` and the format
documented in `perception_bench/infrared.py`.

## After recording

1. Save the video file(s) under `data/perception_bench/ir/` (anywhere
   under `data/` works; `data/` is gitignored, so this never gets
   committed).
2. Copy `tests/perception_bench/fixtures/ir_manifest.example.yaml` to
   `tests/perception_bench/fixtures/ir_manifest.yaml`. **The manifest
   itself is not data** -- it is a small YAML file naming which real clips
   exist and what happens in them, with no image or video bytes in it. It
   is fine, and expected, to commit `ir_manifest.yaml` once it points at
   real (uncommitted) clip files under `data/`.
3. Fill in `zones` for the room the way `config/zones.yaml` is filled in
   (same polygon format), and one `clips` entry per recording with its
   `path` (relative to the manifest), `frame_interval_s` (1 / your camera's
   fps), and the `timeline` from your notes.
4. Run `python -m perception_bench --ir-manifest tests/perception_bench/fixtures/ir_manifest.yaml`
   and read what it says. As of this issue, tier 3 scoring itself (decoding
   the video and running it through the pipeline) is not yet implemented --
   see `perception_bench/infrared.py`'s docstring for why, and pick that up
   as the very next piece of work once real clips exist to test it against.
