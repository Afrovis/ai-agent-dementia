# Volunteer site: execution brief

Read [PLAN.md](PLAN.md) for the design. This
file is what an implementer needs to build it without further context: the
fixed decisions, the rules, the exact contracts, and the work items with their
acceptance checks.

## 1. What this is, in three sentences

A public site at `upload.mathiasvissers.com`, served from the Mac mini through a
Cloudflare Tunnel. Healthy volunteers record a guided three-minute video with
their laptop camera, placed across the room. The site stores the recording
end-to-end encrypted on local disk, runs the existing `tools/video_eval`
pipeline on it, and gives the volunteer an encrypted download of the analysis
video.

## 2. Fixed decisions, do not re-litigate

| Decision | Value |
| --- | --- |
| Host | Mac mini, compose project `nc-volunteer` in `volunteer/docker-compose.yml`. |
| Exposure | Cloudflare Tunnel, remotely managed. No host ports published in production. |
| Access | Fully public. Cloudflare Turnstile on submission creation, managed mode with `appearance: interaction-only`: invisible for most visitors, a checkbox only when Cloudflare suspects an automated agent. |
| Storage | Local disk under `VOLUNTEER_DATA_DIR` (`../data-ai-agent-dementia/volunteer_clips`). No object storage. |
| Media | Video only, never audio. |
| Analysis | `prepare`, `predict --backend yolo --yolo-model yolo11s-pose.pt --variant letterbox640`, `visualize --mode pipeline`. Nothing else. |
| Encryption | AES-256-GCM in the browser; key wrapped with RSA-OAEP-SHA256 (3072-bit); only `worker` holds the private key. |
| Queue | SQLite in `VOLUNTEER_DATA_DIR/volunteer.db`, rollback journal, `busy_timeout=5000`. |
| Frontend | Plain HTML, CSS and JS served by `web`. No framework, no build step, no CDN. |
| Language | English. |
| Retention | Indefinite; delete on request. |
| Demo | Sample 02, unblurred, tag `yolo_yolo11s-pose-letterbox640-8a36a3e9-g`. |

## 3. Non-negotiable rules

1. **No plaintext video on anything Cloudflare can see.** Chunks are encrypted
   before `fetch`, and results are encrypted before they are served. If
   WebCrypto or `MediaRecorder` is unavailable, the page refuses to record and
   says why. There is no unencrypted fallback.
2. **`web` never holds key material.** Not the private key, the AES key, or
   the deletion token. The private key is mounted only into `worker`.
   `CLOUDFLARE_API_TOKEN` is never passed into any container; compose uses
   explicit `environment:` entries, never `env_file:`.
3. **Volunteer video never reaches an AI or cloud provider.** The worker runs
   `video_eval` subcommands through an allowlist (`prepare`, `predict`,
   `visualize`), and a test asserts that `sheets`, `label-codex` and
   `label-local` are rejected. `worker` runs with `network_mode: none`.
4. **The bedside stack comes first.** No shared network, bus, database or
   ports with the root compose project. `worker` is CPU- and memory-limited and
   processes one job at a time.
5. **The privacy copy must stay true.** Any change that touches storage,
   transport, analysis or third-party requests updates the copy in
   `volunteer/PLAN.md` and the page in the same commit.
6. **Nothing from `VOLUNTEER_DATA_DIR`, `VOLUNTEER_KEY_DIR` or `volunteer/.env`
   enters git.** Logs never contain URLs with fragments, tokens, wrapped keys,
   IP addresses or media bytes.

## 4. Layout and conventions

```
volunteer/
├── HANDOFF.md
├── docker-compose.yml
├── .env.example                  .env is gitignored by the root rule
├── common/                       installed into web and worker images
│   ├── pyproject.toml
│   ├── volunteer_common/
│   │   ├── db.py                 schema, migrations, connect(), status transitions
│   │   ├── crypto.py             wire format constants, AAD builders, Python decrypt/encrypt
│   │   ├── paths.py              layout under VOLUNTEER_DATA_DIR
│   │   └── log.py                log(service, message, **fields)
│   └── tests/
│       └── vectors/              committed test vectors produced by scripts/make_vector.mjs
├── web/
│   ├── Dockerfile                build context: volunteer/
│   ├── pyproject.toml
│   ├── volunteer_web/
│   │   ├── __main__.py, main.py, app.py (create_app), turnstile.py, ratelimit.py
│   │   └── static/
│   │       ├── index.html, record.html, status.html, style.css
│   │       ├── crypto.js, recorder.js, status.js, sample.js
│   │       └── script.json       guided script, the single source of prompt text
│   └── tests/
├── worker/
│   ├── Dockerfile                build context: repository root (needs tools/ and services/)
│   ├── pyproject.toml
│   ├── volunteer_worker/
│   │   ├── __main__.py, main.py  poll loop
│   │   ├── pipeline.py           allowlisted video_eval runner
│   │   ├── media.py              decrypt, remux
│   │   ├── purge.py              deletion
│   │   ├── make_keys.py, build_sample.py, delete.py
│   └── tests/
└── scripts/
    ├── setup_cloudflare.py       host-side, standard library only
    ├── render_prompts.py         runs inside the embodiment image
    └── make_vector.mjs           runs in node:22-slim
```

Match the existing services (see `services/dashboard`):

- Python 3.12, setuptools, ruff `target-version = "py312"`, `line-length = 100`,
  lint `E, F, I, UP`.
- Dev extras `pytest`, `httpx`, `ruff`.
- One JSON line per event on stdout with a `service` field (`volunteer-web` or
  `volunteer-worker`), like `dashboard/main.py::_log`.
- Tests need no camera, Cloudflare, network, model weights or Docker.
- Entry point `python -m volunteer_web` or `python -m volunteer_worker`.

`web` must not import `nc_shared` and must not have the repository in its
image. `worker` installs `tools/video_eval` and its perceive and capture
dependencies the way `services/perceive/Dockerfile` does with
`INSTALL_VIDEO_EVAL=true`. It also adds `ffmpeg`, the `cryptography` package,
pinned `mediapipe==0.10.21`, and a baked `yolo11s-pose.pt`. It takes the
`GIT_SHA` build argument, because the image has no `.git`.

## 5. Contracts

### 5.1 Compose

| Service | Network | Mounts | Environment |
| --- | --- | --- | --- |
| `cloudflared` | `edge` | none | `TUNNEL_TOKEN=${CLOUDFLARE_TUNNEL_TOKEN}` |
| `web` | `edge` (needs egress to `challenges.cloudflare.com`) | `${VOLUNTEER_DATA_DIR}:/data` | `PUBLIC_HOSTNAME`, `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY`, `MAX_RECORDING_SECONDS`, `MAX_UPLOAD_MB` |
| `worker` | `network_mode: none` | `${VOLUNTEER_DATA_DIR}:/data`, `${VOLUNTEER_KEY_DIR}:/keys:ro` | `WORKER_CPUS` applied as `cpus:`, `mem_limit: 6g` |

`cloudflared` runs `tunnel --no-autoupdate run`. `web` listens on `0.0.0.0:8000`
with no `ports:` entry. `docker-compose.dev.yml` may publish
`127.0.0.1:8000:8000` for local work and replaces Turnstile with its
always-pass test keys.

### 5.2 Files under `/data`

```
volunteer.db
public.pem                         written by make_keys, read by web
incoming/<id>/key.wrapped          raw RSA-OAEP ciphertext bytes
incoming/<id>/chunk_<n:06d>.enc
incoming/<id>/submission.json      consent version, mime type, markers, duration
raw/<id>.mkv
clips/<id>/…                       video_eval data root layout
analysis/<id>__pipeline.mp4
results/<id>.mp4.enc
sample/sample.mp4, sample/sample.json
prompts/<step_id>.wav
```

`<id>` is 22 characters of URL-safe base64 (16 random bytes), validated with
`^[A-Za-z0-9_-]{22}$` on every route before it touches a path.

### 5.3 Database

```sql
CREATE TABLE submissions (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,              -- ISO 8601 UTC
  updated_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN
    ('recording','finalized','processing','ready','failed','deleted')),
  consent_version TEXT NOT NULL,
  upload_token_sha256 TEXT NOT NULL,
  deletion_token_sha256 TEXT NOT NULL,
  mime_type TEXT,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  bytes INTEGER NOT NULL DEFAULT 0,
  duration_s REAL,
  last_chunk_at TEXT,
  pipeline_tag TEXT,
  error_public TEXT,                     -- safe to show the volunteer
  error_detail TEXT                      -- log-only, never served
);
CREATE TABLE schema_version (version INTEGER NOT NULL);
```

Allowed transitions:
- `recording → finalized` (web on finalize, or worker when stale for 15 minutes)
- `finalized → processing → ready | failed`, where only the worker moves out of
  `finalized`
- any state `→ deleted`

`db.py` enforces the transitions with a conditional `UPDATE … WHERE status = ?`,
and the claim step is atomic.

### 5.4 Crypto wire format

All base64 is URL-safe without padding.

- **Public key.** `GET /api/config` returns `public_key_spki` as base64 SPKI
  DER. The browser imports it with `RSA-OAEP`, `hash: SHA-256`.
- **Session secrets,** generated in the browser:
  - `k`: 32 random bytes, the AES-GCM key
  - `d`: 32 random bytes, the deletion token
- **Status link.** `/s/<id>#k=<b64 k>&d=<b64 d>`. Client code never sends
  `location.hash` or `location.href` anywhere, including in error reports.
- **Wrapped key.** `RSA-OAEP-SHA256(k)`, sent as `wrapped_key` at creation.
- **Chunk n.** The body is `iv (12 bytes) ‖ AES-GCM ciphertext including the
  16-byte tag`. The IV is random per chunk. AAD is the UTF-8 bytes of
  `"chunk:<id>:<n>"`.
- **Result.** A sequence of blocks, each
  `uint32 big-endian length L ‖ iv (12) ‖ ciphertext (L bytes, tag included)`.
  Plaintext blocks are 4 MiB except the last. The AAD of block i is
  `"result:<id>:<i>:<1 if last else 0>"`, so a truncated file fails to decrypt
  instead of producing a short video.
- **Deletion.** The browser sends `sha256(d)` as hex at creation, and `d`
  itself only when deleting. The server stores and compares hashes with
  `hmac.compare_digest`.

`common/tests/vectors/` holds one chunk vector and one two-block result vector
made by `scripts/make_vector.mjs` with WebCrypto. The Python tests decrypt them.
The JS tests (`node --test`) encrypt with `static/crypto.js` and check that the
Python encryptor's output decrypts.

### 5.5 HTTP API (`web`)

All JSON routes return `Cache-Control: no-store`. Chunk, finalize and result
routes require `Authorization: Bearer <upload_token>`, except the result and
status routes, which are public by unguessable id.

| Route | Request | Response |
| --- | --- | --- |
| `GET /` `/record` `/s/{id}` | — | Static HTML |
| `GET /api/config` | — | `{turnstile_site_key, public_key_spki, consent_version, max_recording_seconds, chunk_max_bytes}` |
| `POST /api/submissions` | `{turnstile_token, wrapped_key, deletion_token_sha256, consent_version, consents: {adult, understood, alone}}` | `201 {id, upload_token}`. `400` if a consent is false, `403` if Turnstile fails, `429` if rate-limited, `507` if the disk guard trips. |
| `PUT /api/submissions/{id}/chunks/{n}` | `application/octet-stream`, at most `chunk_max_bytes` (8 MiB) | `204`. `409` unless `recording`, `413` if too large or the total exceeds `MAX_UPLOAD_MB`. Re-putting the same `n` overwrites it. |
| `POST /api/submissions/{id}/finalize` | `{chunk_count, duration_s, mime_type, markers: [{step_id, expected_state, t_start_s, t_end_s, skipped}]}` | `204`. `422` if chunks `0..count-1` are not all present. |
| `GET /api/submissions/{id}` | — | `{status, created_at, queue_position, error_public}` |
| `GET /api/submissions/{id}/result` | — | Encrypted result stream, `404` until `ready` |
| `POST /api/submissions/{id}/delete` | `{deletion_token}` | `204`, `403` on mismatch |
| `GET /sample/sample.mp4` `/sample/sample.json` | — | Must honour `Range`; Safari will not play video without it. |
| `GET /prompts/{step_id}.wav` | — | Rendered prompt audio |
| `GET /healthz` | — | `{ok: true}` |

Rate limit: at most 5 submission creations per hour per `CF-Connecting-IP`,
held in memory only. The header is trusted because only `cloudflared` can reach
`web`. Turnstile siteverify is called without `remoteip`.

Turnstile widget on `/record`: render explicitly with `appearance: "interaction-only"`,
`execution: "render"` and `action: "submit"`. Keep its container hidden until
`before-interactive-callback` fires, then show the checkbox next to the consent
button. The consent button stays disabled until the token callback delivers a
token; `expired-callback` and `error-callback` reset the widget and disable the
button again, with a short retry message. Tokens are single-use and expire
after 300 s, so obtain the token when the consent form is shown, not earlier.

Response headers on HTML:
- `Content-Security-Policy: default-src 'self'; script-src 'self' https://challenges.cloudflare.com; frame-src https://challenges.cloudflare.com; media-src 'self' blob:; img-src 'self' data:; connect-src 'self'`
- `Permissions-Policy: camera=(self), microphone=()`
- `Referrer-Policy: no-referrer`
- `X-Content-Type-Options: nosniff`

### 5.6 Recorder

- `getUserMedia({video: {width: {ideal: 1280}, height: {ideal: 720}, frameRate: {ideal: 30}}, audio: false})`
- The MIME type is the first supported of `video/webm;codecs=vp9`,
  `video/webm;codecs=vp8` and `video/mp4`.
- `videoBitsPerSecond: 2_500_000`, `recorder.start(5000)`.
- Each `dataavailable` is encrypted and uploaded in order, with up to 3 retries
  and backoff. The UI shows chunks pending. Chunks only form a valid file when
  concatenated in order, so upload order is by `n`, not by completion.

### 5.7 Guided script (`static/script.json`)

`[{step_id, text, seconds, expected_state, optional, safety_note}]`. The first
version follows the table in `volunteer/PLAN.md`. Step ids:
`stand_still`, `walk_around`, `sit_edge`, `lie_down`, `sit_up`, `leave_view`,
`come_back`, `floor`, `walk_back`. `floor` has `optional: true` and a safety
note. Before relying on `expected_state` values for any rendering, check them
against the manual marker mapping in `tools/video_eval/video_eval/visualize.py`,
because unmapped names render as `upright`.

### 5.8 Worker job

1. Claim the oldest `finalized` row atomically and set `processing`.
2. Unwrap the key from `/keys/private.pem`. Decrypt chunks in order into
   `ffmpeg -f <webm|mp4> -i pipe:0 -map 0:v:0 -c copy /data/raw/<id>.mkv`. If
   that fails, re-encode with `-c:v libx264 -crf 23 -pix_fmt yuv420p`.
3. Run each step through `pipeline.run(subcommand, args)`, which rejects
   anything outside the allowlist, always passes `--data-root /data`, and
   captures stderr into `error_detail`:
   - `prepare --video /data/raw/<id>.mkv --clip <id> --variant letterbox640`
   - `predict --clip <id> --backend yolo --yolo-model /app/models/yolo11s-pose.pt --variant letterbox640`
   - read `pipeline_tag` from the newest `clips/<id>/predictions/*.meta.json`
   - `visualize --clip <id> --mode pipeline --pipeline-tag <tag> --force`
4. Encrypt `analysis/<id>__pipeline.mp4` into `results/<id>.mp4.enc.tmp`,
   `os.replace` it into place, and set `ready`.
5. On any exception, set `failed` with `error_public` "Processing failed. Your
   video is stored safely; Mathias will look at it." Log the detail.
6. Every loop also finalizes stale `recording` rows and runs pending purges.

## 6. Work items

Each item is independently delegable (Codex or the `coder` agent). The
coordinator runs the acceptance check and keeps the phase checks in
`volunteer/PLAN.md`.

| # | Item | Depends | Acceptance |
| --- | --- | --- | --- |
| V1 | Scaffold: compose file, `common`, `web` and `worker` packages with `/healthz`, logging, Dockerfiles, dev override. | — | `docker compose -f volunteer/docker-compose.yml config` is clean; `docker compose … up web` plus the dev override gives `curl 127.0.0.1:8000/healthz` → 200; `pytest` and `ruff` pass in both images. |
| V2 | `make_keys`: RSA-3072 into `/keys-rw/private.pem` (0600) and `/data/public.pem`; refuses to overwrite. | V1 | A second run exits non-zero without touching files; `docker compose exec web ls /keys` fails. |
| V3 | `scripts/setup_cloudflare.py` (section 7) and the `cloudflared` service. | V1 | Two consecutive runs; the second reports no changes. The strict `curl` check in section 8 returns 200 from a phone on mobile data. Script output contains no secrets. |
| V4 | `common/db.py` and `common/crypto.py`, `static/crypto.js`, `make_vector.mjs`, vectors and tests. | V1 | Python decrypts the committed WebCrypto vectors; `node --test` decrypts Python-made output; tampering with the AAD, IV or a truncated result fails. |
| V5 | `web` API from 5.5, including the Turnstile client (stubbed in tests), rate limit, caps, disk guard and headers. | V4 | Tests cover each status code in 5.5, transition guards, and that no request path writes anything except ciphertext, `key.wrapped` and `submission.json`. |
| V6 | Pages: landing with privacy copy, consent with the interaction-only Turnstile widget, setup checklist, recorder, status with decrypt-and-download and delete. | V5 | Manual: in a normal browser the Turnstile widget never appears and consent enables by itself; with the forced-interaction test sitekey the checkbox appears and consent enables only after ticking it; full recording in Chrome and Safari on macOS; closing the tab mid-recording leaves ordered chunks on disk; a refused camera permission shows a clear message; no request in devtools goes anywhere except self and Turnstile. |
| V7 | Worker job from 5.8. | V4, V5 | Container test: a `testsrc` WebM, encrypted with the JS vector code path, reaches `ready`; the decrypted result is a playable MP4; the allowlist test rejects `label-codex`. Record wall time for a real 3-minute clip. |
| V8 | Deletion: API, worker purge, `python -m volunteer_worker.delete <id>`. | V7 | After deletion, `find "$VOLUNTEER_DATA_DIR" -name "*<id>*"` prints nothing and the row reads `deleted`. |
| V9 | `build_sample` and the demo overlay (`sample.js`: canvas skeleton, bbox, state badge, timeline, gated indicator). | V1 | At five timestamps the overlay matches `analysis/2026-09-13_bedroom-sample-02__pipeline.mp4`; it plays and scrubs in Safari. |
| V10 | `script.json`, `scripts/render_prompts.py`, WebAudio beep, huge-text prompt view, skip buttons, markers in finalize. | V6 | Prompts are audible from 4 m at normal laptop volume; the stored markers match what the recorder saw within 1 s. |
| V11 | Hardening pass: CSP check in browser, log review for rule 6, CPU limit verified under load while the bedside stack runs, privacy copy re-read against the build. | V1–V10 | `docker stats` shows `worker` at or below `WORKER_CPUS` during a job, `perceive` keeps publishing, and grepping all logs for `#k=`, `Bearer` and `wrapped` finds nothing. |
| V12 | Later: face blur in the worker between decrypt and `raw/`; automatic `calibrate-bed` once confirmed offline. | V7 | To be written when scheduled. |

## 7. Cloudflare API, for V3

Standard library only (`urllib.request`). It reads `volunteer/.env`, rewrites
only the keys it owns, preserves comments, and never prints token values.
Check the endpoints against current Cloudflare docs before running.

1. Zone id: `GET /zones?name=$CLOUDFLARE_ZONE_NAME`.
2. Tunnel: `GET /accounts/$ACCOUNT/cfd_tunnel?name=nc-volunteer&is_deleted=false`,
   else `POST /accounts/$ACCOUNT/cfd_tunnel` with `{"name": "nc-volunteer", "config_src": "cloudflare"}`.
3. Token: `GET /accounts/$ACCOUNT/cfd_tunnel/<id>/token`, written as
   `CLOUDFLARE_TUNNEL_TOKEN`.
4. Ingress: `PUT /accounts/$ACCOUNT/cfd_tunnel/<id>/configurations` with
   `{"config": {"ingress": [{"hostname": "$PUBLIC_HOSTNAME", "service": "http://web:8000"}, {"service": "http_status:404"}]}}`.
5. DNS: find `CNAME $PUBLIC_HOSTNAME`, then create or update it with content
   `<id>.cfargotunnel.com` and `proxied: true`. If a non-CNAME record exists
   there, stop and report it rather than replacing it.
6. Turnstile: find a widget named `nc-volunteer` under
   `/accounts/$ACCOUNT/challenges/widgets`, else create it with
   `{"name": "nc-volunteer", "domains": ["$PUBLIC_HOSTNAME"], "mode": "managed"}`.
   The secret is returned only on create. If the widget exists and
   `TURNSTILE_SECRET_KEY` is empty, call `…/widgets/<sitekey>/rotate_secret`.

## 8. Local development

```sh
cd volunteer
cp .env.example .env            # already present on the Mac mini
docker compose build
docker compose run --rm --no-deps -v "$VOLUNTEER_KEY_DIR":/keys-rw worker python -m volunteer_worker.make_keys
python3 scripts/setup_cloudflare.py
docker compose up -d

# Tests, per package
docker compose run --rm --no-deps web sh -c "pip install -q -e '.[dev]' && pytest -q && ruff check . && ruff format --check ."
docker compose run --rm --no-deps worker sh -c "pip install -q -e '/app/volunteer/worker[dev]' && pytest -q /app/volunteer/worker /app/volunteer/common"
docker run --rm -v "$PWD":/v -w /v node:22-slim node --test "web/tests/js/*.test.cjs"

# Demo assets (V9)
docker compose run --rm --no-deps -v "$EVAL_DATA_DIR":/eval:ro worker \
  python -m volunteer_worker.build_sample --eval-root /eval \
  --clip 2026-09-13_bedroom-sample-02 --tag yolo_yolo11s-pose-letterbox640-8a36a3e9-g

# Prompt audio (V10), using the embodiment image from the root project
docker compose -f ../docker-compose.yml run --rm --no-deps \
  -v "$PWD/scripts:/scripts:ro" -v "$PWD/web/volunteer_web/static:/static:ro" \
  -v "$VOLUNTEER_DATA_DIR/prompts:/out" embodiment python /scripts/render_prompts.py /static/script.json /out
```

Strict production check, never with `-k`:

```sh
curl -s -o /dev/null -w "%{http_code}\n" https://upload.mathiasvissers.com/healthz
```

Gotchas:

- SQLite WAL uses shared memory, which is unreliable across containers on a
  macOS bind mount. Keep the rollback journal.
- Browser WebM has no duration header; always remux before `prepare`.
- Docker on macOS runs YOLO on CPU only. Measure, don't assume.
- The `predict` tag embeds the git SHA; always read it back from `.meta.json`.
- Starlette `FileResponse` range support depends on the version; test `Range`
  explicitly for `sample.mp4`.
- The Cloudflare free plan caps request bodies at 100 MB, which is why chunks
  are at most 8 MiB.
- Turnstile test keys: sitekey `1x00000000000000000000BB` always passes invisibly,
  `3x00000000000000000000FF` forces the interactive checkbox, and secret
  `1x0000000000000000000000000000000AA` always validates. Use the forced one to
  exercise the checkbox path, and confirm these against the Cloudflare docs.

## 9. Definition of done for any item

- The acceptance column is demonstrably met, and the evidence is in the commit
  or PR description.
- Unit tests pass with no camera, network, Cloudflare or model weights;
  `ruff check` and `ruff format --check` pass.
- New configuration is in `volunteer/.env.example` with a comment.
- Contract changes update section 5 of this file in the same commit.
- Rules in section 3 still hold. Say so explicitly for V5, V6, V7 and V11.
- Nothing from the data or key directories or from `.env` is committed.

## 10. Shortcuts that are not allowed

- An unencrypted upload path "for browsers without WebCrypto".
- `env_file: .env` on any service, which leaks the API token into containers.
- Publishing `web` on a host port in the production compose file.
- Loading htmx, fonts, analytics or anything else from a CDN.
- `speechSynthesis` for prompts.
- Running `label-local`, `label-codex` or `sheets` "for better results", or
  giving `worker` network access to reach Ollama.
- Decrypting in `web`, or caching decrypted media outside `raw/`, `clips/` and
  `analysis/`.
- Replacing an existing DNS record in `setup_cloudflare.py` without being asked.
- Logging request URLs from client-side error handlers.

## 11. Open questions, and who decides

| Question | Decides |
| --- | --- |
| Contact address for deletion requests shown on the site | Mathias |
| Whether findings from volunteer clips will be published (ethics approval, consent wording) | Mathias |
| Default `MAX_RECORDING_SECONDS` (300 proposed) and `WORKER_CPUS` for this Mac mini | Mathias, after V7 timing |
| Where to back up `VOLUNTEER_KEY_DIR` | Mathias |

## 12. Fixes

Running list of problems found during implementation, with date and item.

- 2026-09-16, V1: `mediapipe==0.10.21`, pinned in section 4 for `worker`'s
  Dockerfile, has no `linux/arm64` wheel on PyPI (only up to 0.10.18 there).
  Dropped the explicit pin; `worker` now takes whatever
  `services/perceive[mediapipe]`'s `mediapipe>=0.10,<1.0` resolves to on the
  build platform, same as `services/perceive/Dockerfile` itself does.
- 2026-09-16, V3: a Cloudflare Tunnel connector token (`cloudflared tunnel run
  --token`) is not a scoped API token and cannot authenticate
  `scripts/setup_cloudflare.py`'s REST calls. Mathias prefers his existing
  manual tunnel workflow, so the supported path is now either
  `scripts/setup_cloudflare.py` with a real scoped API token (section 7), or
  manual setup: create the tunnel and its `upload.mathiasvissers.com` public
  hostname in the Zero Trust dashboard (this also creates the DNS CNAME), and
  a Turnstile widget in the dashboard, then paste the resulting
  `CLOUDFLARE_TUNNEL_TOKEN`, `TURNSTILE_SITE_KEY` and `TURNSTILE_SECRET_KEY`
  into `.env` directly. `CLOUDFLARE_API_TOKEN` stays blank on the manual path.
- 2026-09-16, V4: `node --test web/tests/js` (a bare directory) throws
  `MODULE_NOT_FOUND` on Node 22 rather than discovering test files in it.
  Needs an explicit glob: `node --test "web/tests/js/*.test.cjs"`. Test
  files are named `*.test.cjs` (CommonJS) so `static/crypto.js` can stay a
  plain script with no bundler and still be `require()`-able from the test.
- 2026-09-16, V6: built without a real camera or a cached headless browser
  on this machine, so this is unverified against V6's own acceptance
  criteria: Turnstile's interaction-only behavior, an actual recording in
  Chrome and Safari, the refused-camera-permission message, and a devtools
  network check that nothing leaves the page except self and Turnstile.
  script.json's `walking`/`standing` markers fall through to `upright` in
  `tools/video_eval/video_eval/visualize.py`'s `_manual_state` (only
  `in_bed`/`sitting_up`/`on_floor`/`absent` are mapped there) -- harmless for
  now since markers are stored metadata, not live-rendered, but worth fixing
  in `_manual_state` before any future `clip.yaml` actually uses them.
- 2026-09-16, V7: `worker/Dockerfile` was missing `COPY shared/` and
  `pip install -e /app/shared`. `video_eval.predict` imports
  `capture.main`, which imports `nc_shared.bus` even though `worker` never
  opens a Redis connection -- `python -m video_eval predict` failed with
  `ModuleNotFoundError: No module named 'nc_shared'` until this was added.
  `--data-root` is also a top-level `video_eval` flag and must precede the
  subcommand (`--data-root X predict ...`, not `predict --data-root X`);
  `pipeline.run` initially built the command the other way round. Found by
  actually running a synthetic clip through the built image end to end
  (V7's own acceptance test), not by reading `video_eval`'s source.
- 2026-09-16, V7: installing `torch` alone from the CPU wheel index, then
  letting `ultralytics` (a `perceive[yolo]` dependency) pull in `torchvision`
  from default PyPI separately, gives a `torchvision` built against a
  different `torch` ABI: `RuntimeError: operator torchvision::nms does not
  exist` on the first real YOLO inference. Fixed by installing
  `torch torchvision` together from the same
  `https://download.pytorch.org/whl/cpu` index in one command.
  `services/perceive/Dockerfile` has the same latent bug (installs `torch`
  alone before `perceive[yolo]`) -- worth a matching fix there, since this
  was never exercised end to end until `worker`'s build did.
- 2026-09-16, V7: `tools/video_eval/video_eval/visualize.py`'s
  `_render_frame` hardcoded `item["bridge_letterbox_path"]`, but
  `prepare_video` only ever writes the manifest key for whichever
  `--variant` was actually prepared (`bridge_path` for `squash`,
  `bridge_letterbox_640_path` for `letterbox640`) -- so `visualize --mode
  pipeline` raised `KeyError` for every variant except `letterbox` itself,
  including the `letterbox640` variant this project uses throughout.
  Fixed in `visualize.py` to look up whichever bridge key the frame record
  actually has, via `video_eval.paths.BRIDGE_VARIANTS`. This is a bug in
  shared tooling, not `volunteer/`, so it may also affect other callers of
  `visualize --mode pipeline` with a non-`letterbox` variant. Confirmed
  fixed with a full end-to-end run: a synthetic `ffmpeg testsrc` clip,
  encrypted into chunks and pushed through `run_once` in the real built
  `worker` image, reached `ready` and decrypted into a valid, ffprobe-
  playable MP4 -- V7's stated acceptance test.
- 2026-09-16, V9: `build_sample.py`'s `encode_sample_video` wrote ffmpeg's
  output to a `.tmp`-suffixed path for the atomic rename (`os.replace`
  pattern used elsewhere), but ffmpeg infers the container format from the
  output filename's extension -- `sample.mp4.tmp` has none it recognizes,
  so muxer setup failed with "Unable to choose an output format". Fixed by
  passing `-f mp4` explicitly. Found by actually running `build_sample.py`
  against the real `2026-09-13_bedroom-sample-02` clip with local ffmpeg
  (available on this machine, so V9's own acceptance test could run
  end to end rather than only by inspection): rendered the reference
  `visualize --mode pipeline` video and confirmed the canvas overlay's
  skeleton, bbox and state label match it at five timestamps.
- 2026-09-16, V10: `scripts/render_prompts.py` written and confirmed
  end to end -- a local `piper-tts` install (this machine has one under
  `.venv-dialogue`, separate from the `embodiment` image) plus a fresh
  `python -m piper.download_voices en_US-lessac-medium` rendered all nine
  `script.json` steps to valid mono 22.05kHz WAVs, re-running without
  `--force` skipped all nine, and `--force` re-rendered them; a spot-check
  with `ffmpeg -af volumedetect` showed normal speech loudness (-15.7dB
  mean, 0dB peak), not an anemic whisper. Rendered at `--speed 1.0`
  (normal pace) rather than the bedside voice's 0.85x slow-down, since
  these are one-time instructional prompts to a volunteer at their laptop,
  not a nighttime companion's speech.
- 2026-09-16, V11: `volunteer_web.main`'s `uvicorn.run(...)` used the
  default access log, which prints the client address on every request
  line (`172.23.0.1:55980 - "GET /api/... HTTP/1.1" 200 OK`) -- a direct
  violation of rule 6 ("logs never contain ... IP addresses"). In
  production the address uvicorn sees is `cloudflared`'s internal Docker
  address rather than a volunteer's real IP, since Cloudflare Tunnel
  proxies the connection, but rule 6 doesn't carve out an exception for
  that and a stray access log is still the wrong default. Fixed with
  `access_log=False`; the app's own structured `log()` calls already cover
  what's worth recording. Found by actually building both images, running
  `web` and `worker` together with the real `docker-compose.yml` (isolated
  `.env` pointed at scratch data/key directories, not
  `VOLUNTEER_DATA_DIR`/`VOLUNTEER_KEY_DIR`), and pushing a synthetic
  `ffmpeg testsrc` clip through the real HTTP API end to end (create,
  encrypt and upload chunks with `volunteer_common.crypto`, finalize,
  poll) to `ready` -- not by reading the source. Also confirmed while at
  it: `docker inspect`'s `NanoCpus` on the `worker` container matched
  `WORKER_CPUS` exactly (the compose `cpus:` field is applied as a real
  cgroup limit, not just documentation), and grepping the rebuilt stack's
  logs for `#k=`, `Bearer`, `wrapped` and IP-shaped strings after the fix
  found nothing but the harmless `0.0.0.0` bind address. Not verified:
  `perceive` continuing to publish while `worker` runs a job, since the
  root bedside stack was not running on this machine at the time (no
  camera/mic attached) -- that half of V11's acceptance needs a real
  concurrent run on the Mac mini deployment. Privacy copy (`PLAN.md`,
  `index.html`) re-read against V9-V11's changes: still accurate, since
  none of the three touched storage, transport, analysis or third-party
  requests.
