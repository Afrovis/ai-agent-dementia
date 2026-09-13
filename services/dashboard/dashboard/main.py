"""Entry point for the `dashboard` service (issues #10 and #22).

Caregiver web dashboard (HANDOFF.md section 4). Tonight and History join the
Zones editor in issue #22; profile, strategies, and system settings follow.

`DashboardConfig.from_env` reads `DASHBOARD_PASSWORD` and `ZONES_PATH`.
`DASHBOARD_PASSWORD` is required for the app to serve anything but 503s
(`dashboard.app.check_auth`, PLAN.md section 9: "protected by a single
password") -- there is deliberately no default, so a fresh install does
not accidentally expose a live camera frame with no password set.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import redis
import uvicorn
from nc_shared.bus import Bus

from dashboard.app import create_app

SERVICE_NAME = "dashboard"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVICE_NAME)


def _log(message: str, **fields: object) -> None:
    """Log one structured JSON line to stdout (HANDOFF.md section 4)."""
    logger.info(json.dumps({"service": SERVICE_NAME, "message": message, **fields}))


@dataclass(frozen=True)
class DashboardConfig:
    """`dashboard`'s env-driven configuration (HANDOFF.md section 4: env,
    then yaml -- `zones.yaml` is the yaml here -- then defaults in code)."""

    port: int = 8444
    password: str | None = None
    zones_path: str | None = None
    db_path: str = "/app/data/night.db"
    timezone: str = "UTC"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> DashboardConfig:
        """Build a `DashboardConfig` from environment variables, defaults otherwise."""
        env = os.environ if env is None else env
        return cls(
            port=int(env.get("DASHBOARD_PORT", "8444")),
            # `or None`, not `.get(..., None)`: compose passes an unset .env
            # variable through as an empty string, and an empty string must
            # fail closed the same way a genuinely unset variable does.
            password=env.get("DASHBOARD_PASSWORD") or None,
            zones_path=env.get("ZONES_PATH") or None,
            db_path=env.get("DB_PATH") or "/app/data/night.db",
            timezone=env.get("TZ") or "UTC",
        )


def run() -> None:
    """Start the dashboard FastAPI app on `DASHBOARD_PORT`."""
    config = DashboardConfig.from_env()
    redis_url = os.environ.get("REDIS_URL", "redis://bus:6379")

    if config.password is None:
        _log(
            "DASHBOARD_PASSWORD is not set; every route will answer 503 until it is configured",
            port=config.port,
        )
    else:
        _log("dashboard starting", port=config.port)

    bus = Bus(redis.Redis.from_url(redis_url))
    try:
        app = create_app(
            bus,
            password=config.password,
            zones_path=config.zones_path,
            db_path=config.db_path,
            timezone=config.timezone,
        )
    except ValueError as exc:
        # A password HTTP Basic cannot carry. Fail loudly with the reason
        # rather than starting a dashboard nobody can ever log in to.
        _log("refusing to start", error=str(exc))
        raise SystemExit(1) from exc

    uvicorn.run(app, host="0.0.0.0", port=config.port)  # noqa: S104 - LAN-only device


if __name__ == "__main__":
    run()
