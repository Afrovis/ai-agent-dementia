"""Tests for `dashboard.main.DashboardConfig.from_env` (no Redis, no camera,
no browser)."""

from __future__ import annotations

from dashboard.main import DashboardConfig


def test_from_env_uses_defaults_when_nothing_is_set():
    config = DashboardConfig.from_env(env={})

    assert config.port == 8444
    assert config.password is None
    assert config.zones_path is None
    assert config.db_path == "/app/data/night.db"
    assert config.timezone == "UTC"


def test_from_env_reads_configured_values():
    config = DashboardConfig.from_env(
        env={
            "DASHBOARD_PORT": "9000",
            "DASHBOARD_PASSWORD": "hunter2",
            "ZONES_PATH": "/app/config/zones.yaml",
            "DB_PATH": "/app/data/custom.db",
            "TZ": "America/New_York",
        }
    )

    assert config.port == 9000
    assert config.password == "hunter2"
    assert config.zones_path == "/app/config/zones.yaml"
    assert config.db_path == "/app/data/custom.db"
    assert config.timezone == "America/New_York"


def test_from_env_treats_an_empty_string_password_as_unset():
    # docker-compose passes a variable missing from `.env` through as an
    # empty string; that must fail closed the same way a genuinely unset
    # DASHBOARD_PASSWORD does (dashboard.app.check_auth), not silently
    # accept an empty password as valid.
    config = DashboardConfig.from_env(env={"DASHBOARD_PASSWORD": ""})

    assert config.password is None
