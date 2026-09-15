# Volunteer recording site

Design for `upload.mathiasvissers.com`: a public page where
healthy volunteers record themselves moving around a room with their laptop
camera, see what the pipeline makes of the sample-02 recording, and download
the analysis video of their own clip once it has been processed.

The goal is more varied evaluation footage for `tools/video_eval`. This site is
not part of the bedside stack, never touches its Redis bus, and must not slow
it down.

## Decisions

| Topic | Decision |
| --- | --- |
| Hosting | Everything on the Mac mini, one compose project in `volunteer/`. |
| Exposure | Cloudflare Tunnel to `upload.mathiasvissers.com`. No open router port. Fully public. |
| Storage | Local disk, `../data-ai-agent-dementia/volunteer_clips`. No object storage. |
| Recording setup | Laptop placed as far away as possible, ideally the whole floor in view. Video only, no audio. |
| Analysis | `visualize --mode pipeline` only. No `label-local`, no Codex. |
| Demo | Sample 02 (Mathias), unblurred, with a live classification overlay. |
| Retention | Kept indefinitely; deleted on request. |
| Language | English. |
| Later | Face-blur before the clip is saved. |

## Privacy promise and what makes it true

Cloudflare sits in the path: Tunnel terminates TLS at Cloudflare's edge. Plain
HTTPS would therefore let a third party see the video in transit, and "no data
is sent to third parties" would be false. The site closes that gap with
end-to-end encryption between the volunteer's browser and the worker.

1. On consent, the browser generates a random AES-256-GCM key and a random
   deletion token. Both live only in the URL fragment of the private status
   link (`/s/<id>#k=…&d=…`). Browsers never send the fragment to a server.
2. The AES key is wrapped with the site's RSA-OAEP public key and sent once.
   Only the worker holds the private key.
3. Every recorded chunk is encrypted in the browser before upload. `web`
   stores ciphertext only and cannot decrypt it.
4. The worker unwraps the key, decrypts the clip locally, runs the pipeline,
   and encrypts the analysis MP4 with the same AES key.
5. The status page downloads the ciphertext and decrypts it in the browser into
   a normal file download.

Cloudflare therefore carries and briefly sees only ciphertext, in both
directions. The sample-02 demo is public footage and is not encrypted.

Copy for the landing page (keep it accurate if the design changes):

> **Your video stays private.** It is encrypted in your browser before it leaves
> your computer. Only I (Mathias) can see it. The analysis runs on my own
> computer with local AI. Your video is never sent to any AI company or cloud
> service. Cloudflare delivers this website and checks for bots, but it only
> ever handles encrypted data it cannot read. You can ask for your video to be
> deleted at any time with the link you get after recording. No sound is
> recorded. This is a research project, not a medical service.

Hard rules for the implementation:

- The worker never runs `sheets`, `label-codex` or `label-local` on volunteer
  clips. Enforce it in code (an allowlist of subcommands), not by convention.
- No third-party scripts, fonts, analytics or CDNs on the page. Everything is
  served from `web`, except Cloudflare Turnstile.
- No `speechSynthesis` for prompts. Some browser voices are cloud-backed.
  Prompt audio is pre-rendered with Piper, like `embodiment`.
- Do not log IP addresses beyond what rate limiting needs in memory.
- The RSA private key lives in `VOLUNTEER_KEY_DIR`, mounted only into `worker`.

## Architecture

```
browser ──HTTPS──▶ Cloudflare edge ──Tunnel──▶ cloudflared ──▶ web ──┐
                                                                     │ SQLite + files
                                                   worker ◀──────────┘ (volunteer_clips)
```

| Service | Role |
| --- | --- |
| `cloudflared` | `cloudflared tunnel run --token $CLOUDFLARE_TUNNEL_TOKEN`. Ingress only to `http://web:8000`. |
| `web` | FastAPI. Static pages, Turnstile check, submission and chunk API, status and encrypted download. Holds only the public key. No access to the repository or the bus. |
| `worker` | Polls SQLite for finalized submissions, decrypts, runs `video_eval`, encrypts the result. Concurrency 1, CPU-limited. |

A separate compose network, with nothing published on host ports.
`cloudflared` reaches `web`; `web` reaches only Turnstile's verify endpoint;
`worker` has no network at all. SQLite (`volunteer_clips/volunteer.db`,
rollback journal, since WAL is unreliable across containers on a macOS bind
mount) is the queue. A second Redis would
be one more thing to run for a queue that sees a few jobs a day.

`worker` gets `cpus: ${WORKER_CPUS}` and `mem_limit`, so a volunteer job cannot
starve `perceive` or `listen` on the same machine.

## Data layout

`VOLUNTEER_DATA_DIR` doubles as the `video_eval` data root
(`--data-root`), so volunteer clips never mix with the bedroom recordings:

```
volunteer_clips/
├── volunteer.db                      submissions table
├── incoming/<id>/
│   ├── key.wrapped                   RSA-OAEP wrapped AES key
│   ├── chunk_000000.enc …            ciphertext as uploaded
│   └── submission.json               consent version, mime type, markers, duration
├── raw/<id>.mkv                      decrypted, remuxed recording
├── clips/<id>/…                      video_eval prepare/predict outputs
├── analysis/<id>__pipeline.mp4       plaintext render, local only
└── results/<id>.mp4.enc              encrypted render served to the volunteer
```

The submission id is a random 128-bit URL-safe string, which also serves as the
`video_eval` clip id. The database stores `sha256(deletion_token)`, never the
token.

The exact `submissions` schema, status transitions, crypto wire format and HTTP
API are specified in `volunteer/HANDOFF.md` section 5.

## Volunteer flow

1. **Landing** (`/`). A one-paragraph explanation, the privacy copy, the
   sample-02 demo, and a "Record a video" button.
2. **Consent** (`/record`). Checkboxes: 18 or older; understands the video is
   stored and analysed as described; will only film themselves, with nobody
   else in view. A Turnstile check runs in the background
   (managed mode, `appearance: interaction-only`): most visitors see nothing,
   and a regular checkbox appears only if Cloudflare suspects an automated
   agent. Accepting creates the submission
   (`POST /api/submissions` with the wrapped key and Turnstile token) and
   rewrites the URL to include the fragment. The page also tells people to
   bookmark the link.
3. **Setup.** Camera preview at the largest size, with a checklist: laptop as
   far away as possible, whole floor visible, room well lit, whole body in frame
   when standing at the far end. A "Test the voice prompts" button checks that
   the volume is loud enough to hear from across the room.
4. **Recording.** A 10-second countdown to walk into position, then a guided
   script. Prompts are spoken (Piper WAVs) and shown in huge text, because the
   screen is far away. `MediaRecorder` delivers a chunk every 5 s; each chunk is
   encrypted and `PUT` to `/api/submissions/<id>/chunks/<n>` immediately, so
   the footage is on disk even if the tab closes. There is a visible "Stop"
   button and a hard cap at `MAX_RECORDING_SECONDS`.
5. **Finalize.** `POST /api/submissions/<id>/finalize` with the prompt
   timestamps. Submissions still in `recording` with no chunk for 15 minutes
   are finalized by the worker.
6. **Status** (`/s/<id>#k=…&d=…`). Polls
   `GET /api/submissions/<id>` for queued, processing, ready or failed. When
   ready, it fetches `/api/submissions/<id>/result`, decrypts it and offers
   `analysis.mp4`. A "Delete my video" button sends the deletion token.

### Guided script

About three minutes, and every step can be skipped. Each prompt timestamp is
recorded as a marker and stored with the submission. That gives ground truth
for free later, if a clip ever gets a `clip.yaml` script. Marker names must map
to the names in `visualize.py`, because unmapped names render as `upright`
(see `tools/video_eval/README.md`).

| Prompt | Seconds | Marker |
| --- | --- | --- |
| Walk into view and stand still | 10 | `standing` |
| Walk slowly around the room | 20 | `walking` |
| Sit on the edge of the bed or sofa | 15 | `sitting_up` |
| Lie down on the bed or sofa | 20 | `in_bed` |
| Sit up | 10 | `sitting_up` |
| Stand up and walk out of view | 15 | `absent` |
| Come back and stand still | 10 | `standing` |
| Optional: slowly lower yourself to sit or lie on the floor, only if comfortable | 20 | `on_floor` |
| Get up and walk back to the laptop | 15 | `walking` |

The floor step carries an explicit safety line and a "Skip" affordance that is
the default when unsure. The Marker column is each step's expected state; step ids
are defined separately in `volunteer/HANDOFF.md`. Check expected states against
the `visualize.py` mapping when implementing.

## Sample-02 demo

Source: `clips/2026-09-13_bedroom-sample-02` in `EVAL_DATA_DIR`, raw 3840×2160
HEVC at 60 fps, 114 s. Predictions: `predictions/yolo_yolo11s-pose-letterbox640-8a36a3e9-g.jsonl`
(2 fps; `t_s`, `state`, `state_confidence`, `zone`, `gated`, `bbox`, and COCO-17
`landmarks` as `[x, y, visibility]`).

A one-off `python -m volunteer_worker.build_sample` produces the demo assets in
`VOLUNTEER_DATA_DIR/sample/`, which is not in git:

- `sample.mp4`: 1280×720 H.264, 30 fps, no audio, `+faststart`.
- `sample.json`: the prediction rows reduced to `t_s, state, state_confidence,
  gated, bbox, landmarks`, with coordinates converted from letterbox640 space to
  video space. A 16:9 source in a 640×480 letterbox has content height 360,
  offset by 60 px: `y_video = (y * 480 - 60) / 360`, and `x` unchanged. Verify
  this against a rendered frame of `analysis/…__pipeline.mp4` before trusting it.

The page plays `sample.mp4` with a `<canvas>` overlay. On every
`requestVideoFrameCallback` it takes the latest prediction at or before
`currentTime` and draws the skeleton, the box, and a state badge using the
`STATE_COLORS` from `visualize.py`. Beside it sits a timeline strip of state
over time with a playhead. When a volunteer scrubs, the overlay follows. Gated
frames show "motion gate: skipped" rather than a stale skeleton.

## Worker pipeline

For each `finalized` submission, oldest first:

1. Mark it `processing`. Unwrap the key and decrypt the chunks in order into
   one stream.
2. `ffmpeg -i - -map 0:v:0 -c copy raw/<id>.mkv`. This drops any audio track
   defensively and gives browser WebM, which has no duration header, a proper
   container. If the copy fails (Safari MP4 fragments), re-encode with
   `libx264 -crf 23`.
3. `python -m video_eval prepare --data-root $VOLUNTEER_DATA_DIR --video raw/<id>.mkv --clip <id> --variant letterbox640`
4. `python -m video_eval predict --data-root … --clip <id> --backend yolo --yolo-model yolo11s-pose.pt --variant letterbox640`
5. Read the tag from the newest `clips/<id>/predictions/*.meta.json` rather than
   recomputing it. The tag embeds the git SHA.
6. `python -m video_eval visualize --data-root … --clip <id> --mode pipeline --pipeline-tag <tag> --force`
7. Encrypt `analysis/<id>__pipeline.mp4` into `results/<id>.mp4.enc` and mark
   the submission `ready`.

No zones, bed calibration or phantoms exist for an unseen room. `predict`
degrades to an empty zone map, with every point `other`, and the phantom filter
off. The result page says so in plain words: the room was not calibrated, so
labels are rougher than on the sample. Automatic `calibrate-bed` on review
frames is a later improvement, once it has been confirmed to run fully offline.

`predict` on sample 02 (114 s, motion-gated) took 14 s of wall time, so a
3-minute clip should finish in a few minutes including `prepare` and
`visualize`. Docker on macOS runs YOLO on CPU only. Measure this in phase 3 and
show a real estimate on the status page.

The image is `volunteer/worker/Dockerfile`, with build context at the
repository root. It mirrors `services/perceive/Dockerfile` with
`INSTALL_VIDEO_EVAL=true`, and adds `ffmpeg`, `cryptography`, the baked
`yolo11s-pose.pt` (no download at runtime) and `mediapipe==0.10.21`. Pass
`GIT_SHA` as a build argument, because there is no `.git` in the image.

## Cloudflare setup

`volunteer/scripts/setup_cloudflare.py` is idempotent. It reads
`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_ZONE_NAME` from
`volunteer/.env`, and:

1. Creates, or reuses, a remotely managed tunnel named `nc-volunteer`, then
   writes `CLOUDFLARE_TUNNEL_TOKEN`.
2. Sets tunnel ingress `upload.mathiasvissers.com → http://web:8000`, with a
   catch-all 404.
3. Creates, or updates, the proxied CNAME `upload → <tunnel-id>.cfargotunnel.com`.
4. Creates, or reuses, a Turnstile widget for the hostname, then writes
   `TURNSTILE_SITE_KEY` and `TURNSTILE_SECRET_KEY`.

It never prints the token or secrets. The token needs Account › Cloudflare
Tunnel › Edit, Account › Turnstile › Edit, and Zone › DNS › Edit on
`mathiasvissers.com`.

`python -m volunteer_worker.make_keys` generates the RSA-3072 key pair in
`VOLUNTEER_KEY_DIR` (mode 600), and only if none exists. Losing the private key
makes every stored clip unreadable, so back it up outside the Mac mini.

## Abuse and limits

- Turnstile on submission creation. A random per-submission upload token,
  stored hashed, authorizes the chunk calls.
- In-memory rate limits: 5 submissions per IP per hour, and one active
  recording per submission.
- Per-chunk size cap. Total `MAX_UPLOAD_MB` and `MAX_RECORDING_SECONDS` are
  enforced server side.
- `web` accepts only `application/octet-stream` chunks and never parses the
  ciphertext. Only the worker's ffmpeg touches decrypted media, in its own
  container, with no network at all (`network_mode: none`).
- Security headers: strict CSP (self plus the Turnstile origin), `Permissions-Policy: camera=(self)`,
  no referrer.
- Free disk space check. `web` refuses new submissions below a threshold.

## Deletion

"Delete my video" on the status page sends the deletion token. `web` verifies
it against the stored hash and marks the row `deleted`. The worker removes
`incoming/<id>`, `raw/<id>.*`, `clips/<id>`, `analysis/<id>__*`, and
`results/<id>*`. Email requests are handled with `python -m volunteer_worker.delete <id>`.

## Build phases

1. **Plumbing.** Compose skeleton, key generation, Cloudflare setup script, and
   `web` serving a placeholder through the tunnel. Done when
   `curl -s -o /dev/null -w "%{http_code}" https://upload.mathiasvissers.com/`
   answers 200 from outside the tailnet.
2. **Encrypted upload.** Consent, recorder, chunked encrypted upload, finalize,
   and SQLite. Done when a recording made in Chrome and in Safari reassembles
   and decrypts bit-exact in a test harness, and `web` holds no key material.
3. **Worker.** Decrypt, remux, the three `video_eval` steps, encrypted result,
   status and download. Done when a real recording made from across the room
   downloads a playable analysis MP4. Record wall time.
4. **Sample demo.** `build_sample.py`, canvas overlay and timeline. Done when
   the overlay lines up with the person across the whole clip.
5. **Guided script and hardening.** Piper prompt audio, Turnstile, limits, CSP,
   deletion, and disk guard.
6. **Later.** Face blur before save, using the `blur` step's detectors, applied
   to the decrypted clip before `raw/` is written. Automatic bed calibration.

## Testing without a camera or Cloudflare

Consistent with the repository rule that no service needs hardware to test:

- `web`: pytest with FastAPI's TestClient and a stubbed Turnstile verifier.
  Covers creation, chunk ordering, caps, finalize, deletion, and that key
  material never lands in `web`'s filesystem.
- Crypto: a fixed test vector. Encrypt in Node's WebCrypto (the same code the
  page ships), decrypt in the worker's Python, and assert the round trip.
- `worker`: a synthetic clip from `ffmpeg -f lavfi -i testsrc=duration=6`,
  WebM-encoded and encrypted, run end to end with `predict` stubbed where
  model weights are absent.
- Phase acceptance on real hardware is manual and listed per phase above.

## Configuration

See `volunteer/.env.example`. The execution brief, with contracts and work
items, is `volunteer/HANDOFF.md`. `volunteer/.env` is gitignored by the root
`.env` rule.
