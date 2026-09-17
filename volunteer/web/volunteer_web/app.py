"""FastAPI app factory for `web` (HANDOFF.md section 5.5)."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from volunteer_common import crypto

from volunteer_web import submissions
from volunteer_web.config import WebConfig
from volunteer_web.ratelimit import RateLimiter
from volunteer_web.submissions import SubmissionsService
from volunteer_web.turnstile import TurnstileVerifier, TurnstileVerifierProtocol

STATIC_DIR = Path(__file__).parent / "static"

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; "
        "frame-src https://challenges.cloudflare.com; media-src 'self' blob:; "
        "img-src 'self' data:; connect-src 'self'"
    ),
    "Permissions-Policy": "camera=(self), microphone=()",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    return header[len("Bearer ") :]


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("CF-Connecting-IP")
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


class ConsentFields(BaseModel):
    adult: bool
    understood: bool
    alone: bool


class CreateSubmissionRequest(BaseModel):
    turnstile_token: str
    wrapped_key: str
    deletion_token_sha256: str
    consent_version: str
    consents: ConsentFields
    mime_type: str | None = None


class MarkerModel(BaseModel):
    step_id: str
    expected_state: str
    t_start_s: float
    t_end_s: float
    skipped: bool = False


class FinalizeRequest(BaseModel):
    chunk_count: int
    duration_s: float
    mime_type: str | None = None
    markers: list[MarkerModel] = []


class DeleteRequest(BaseModel):
    deletion_token: str


def create_app(
    config: WebConfig | None = None,
    *,
    turnstile_verifier: TurnstileVerifierProtocol | None = None,
) -> FastAPI:
    config = config or WebConfig.from_env()
    data_root = Path(config.data_dir)
    app = FastAPI(title="volunteer-web")
    app.state.config = config
    app.state.submissions = SubmissionsService(
        data_root,
        min_free_disk_mb=config.min_free_disk_mb,
        consent_version=config.consent_version,
    )
    app.state.turnstile_verifier = turnstile_verifier or TurnstileVerifier(
        config.turnstile_secret_key
    )
    app.state.rate_limiter = RateLimiter(config.submissions_per_ip_per_hour)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        for key, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(key, value)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/config")
    def get_config() -> dict:
        public_key_path = data_root / "public.pem"
        spki_b64 = ""
        if public_key_path.exists():
            der = crypto.load_public_key_spki_der(public_key_path.read_bytes())
            spki_b64 = b64url_encode(der)
        return {
            "turnstile_site_key": config.turnstile_site_key,
            "public_key_spki": spki_b64,
            "consent_version": config.consent_version,
            "max_recording_seconds": config.max_recording_seconds,
            "chunk_max_bytes": config.chunk_max_bytes,
        }

    @app.post("/api/submissions", status_code=201)
    async def create_submission(payload: CreateSubmissionRequest, request: Request) -> dict:
        if not app.state.rate_limiter.allow(_client_ip(request)):
            raise HTTPException(429, "too many submissions from this address, try again later")
        if not await app.state.turnstile_verifier.verify(payload.turnstile_token):
            raise HTTPException(403, "verification failed")
        try:
            created = app.state.submissions.create(
                consents=payload.consents.model_dump(),
                wrapped_key=b64url_decode(payload.wrapped_key),
                deletion_token_sha256=payload.deletion_token_sha256,
                mime_type=payload.mime_type,
            )
        except submissions.ConsentRejected as exc:
            raise HTTPException(400, str(exc)) from exc
        except submissions.DiskLow as exc:
            raise HTTPException(507, str(exc)) from exc
        return {"id": created.submission_id, "upload_token": created.upload_token}

    @app.put("/api/submissions/{submission_id}/chunks/{n}", status_code=204)
    async def put_chunk(submission_id: str, n: int, request: Request) -> Response:
        body = await request.body()
        try:
            app.state.submissions.write_chunk(
                submission_id,
                n,
                body,
                upload_token=_bearer_token(request),
                chunk_max_bytes=config.chunk_max_bytes,
                max_upload_mb=config.max_upload_mb,
            )
        except submissions.NotFound as exc:
            raise HTTPException(404) from exc
        except submissions.Unauthorized as exc:
            raise HTTPException(401, str(exc)) from exc
        except submissions.Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except submissions.TooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        return Response(status_code=204)

    @app.post("/api/submissions/{submission_id}/finalize", status_code=204)
    async def finalize(submission_id: str, payload: FinalizeRequest, request: Request) -> Response:
        try:
            app.state.submissions.finalize(
                submission_id,
                upload_token=_bearer_token(request),
                chunk_count=payload.chunk_count,
                duration_s=payload.duration_s,
                mime_type=payload.mime_type,
                markers=[marker.model_dump() for marker in payload.markers],
            )
        except submissions.NotFound as exc:
            raise HTTPException(404) from exc
        except submissions.Unauthorized as exc:
            raise HTTPException(401, str(exc)) from exc
        except submissions.IncompleteUpload as exc:
            raise HTTPException(422, str(exc)) from exc
        except submissions.Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    @app.get("/api/submissions/{submission_id}")
    def get_status(submission_id: str) -> dict:
        try:
            return app.state.submissions.get_status(submission_id)
        except submissions.NotFound as exc:
            raise HTTPException(404) from exc

    @app.get("/api/submissions/{submission_id}/result")
    def get_result(submission_id: str) -> FileResponse:
        try:
            path = app.state.submissions.get_result_path(submission_id)
        except submissions.NotFound as exc:
            raise HTTPException(404) from exc
        return FileResponse(path, media_type="application/octet-stream")

    @app.post("/api/submissions/{submission_id}/delete", status_code=204)
    def delete_submission(submission_id: str, payload: DeleteRequest) -> Response:
        try:
            app.state.submissions.delete(submission_id, deletion_token=payload.deletion_token)
        except submissions.NotFound as exc:
            raise HTTPException(404) from exc
        except submissions.Unauthorized as exc:
            raise HTTPException(403, str(exc)) from exc
        return Response(status_code=204)

    @app.get("/record")
    def record_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "record.html")

    @app.get("/s/{submission_id}")
    def status_page(submission_id: str) -> FileResponse:
        return FileResponse(STATIC_DIR / "status.html")

    # StaticFiles re-checks the directory exists on every request even with
    # check_dir=False (that flag only skips the constructor-time check), so
    # these must exist before the first request rather than only at mount time.
    (data_root / "sample").mkdir(parents=True, exist_ok=True)
    (data_root / "prompts").mkdir(parents=True, exist_ok=True)
    app.mount("/sample", StaticFiles(directory=data_root / "sample"), name="sample")
    app.mount("/prompts", StaticFiles(directory=data_root / "prompts"), name="prompts")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app
