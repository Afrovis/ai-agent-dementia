"""Env-driven configuration for `web` (HANDOFF.md section 5.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebConfig:
    port: int = 8000
    data_dir: str = "/data"
    public_hostname: str = "localhost"
    turnstile_site_key: str = ""
    turnstile_secret_key: str = ""
    max_recording_seconds: int = 300
    max_upload_mb: int = 500
    chunk_max_bytes: int = 8 * 1024 * 1024
    consent_version: str = "v1"
    submissions_per_ip_per_hour: int = 5
    min_free_disk_mb: int = 2048

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> WebConfig:
        import os

        source = env if env is not None else os.environ
        return cls(
            port=int(source.get("PORT", cls.port)),
            data_dir=source.get("VOLUNTEER_DATA_DIR_MOUNT", "/data"),
            public_hostname=source.get("PUBLIC_HOSTNAME", cls.public_hostname),
            turnstile_site_key=source.get("TURNSTILE_SITE_KEY", ""),
            turnstile_secret_key=source.get("TURNSTILE_SECRET_KEY", ""),
            max_recording_seconds=int(
                source.get("MAX_RECORDING_SECONDS", cls.max_recording_seconds)
            ),
            max_upload_mb=int(source.get("MAX_UPLOAD_MB", cls.max_upload_mb)),
        )
