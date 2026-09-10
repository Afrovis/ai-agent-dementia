"""Tests for `agent.config.AgentConfig`, notably the night-window wrap
across midnight."""

from datetime import datetime, time

from agent.config import AgentConfig


def test_from_env_defaults():
    config = AgentConfig.from_env({})
    assert config.night_start == time(21, 0)
    assert config.night_end == time(7, 0)
    assert config.observe_seconds == 20.0
    assert config.cooldown_seconds == 300.0
    assert config.in_bed_stable_seconds == 120.0
    assert config.floor_limit_seconds == 0.0
    assert config.absent_limit_seconds == 600.0
    assert config.restroom_timeout_seconds == 900.0
    assert config.zone_confirm_readings == 3
    assert config.strategies_path is None
    assert config.say_min_gap_seconds == 8.0


def test_from_env_reads_every_key():
    env = {
        "AGENT_NIGHT_START": "22:30",
        "AGENT_NIGHT_END": "06:15",
        "AGENT_OBSERVE_SECONDS": "15",
        "AGENT_COOLDOWN_SECONDS": "120",
        "AGENT_IN_BED_STABLE_SECONDS": "90",
        "AGENT_FLOOR_LIMIT_SECONDS": "5",
        "AGENT_ABSENT_LIMIT_SECONDS": "300",
        "AGENT_RESTROOM_TIMEOUT_SECONDS": "600",
        "AGENT_ZONE_CONFIRM_READINGS": "5",
        "STRATEGIES_PATH": "/tmp/strategies.yaml",
        "AGENT_SAY_MIN_GAP_SECONDS": "10",
    }
    config = AgentConfig.from_env(env)
    assert config.night_start == time(22, 30)
    assert config.night_end == time(6, 15)
    assert config.observe_seconds == 15.0
    assert config.cooldown_seconds == 120.0
    assert config.in_bed_stable_seconds == 90.0
    assert config.floor_limit_seconds == 5.0
    assert config.absent_limit_seconds == 300.0
    assert config.restroom_timeout_seconds == 600.0
    assert config.zone_confirm_readings == 5
    assert config.strategies_path == "/tmp/strategies.yaml"
    assert config.say_min_gap_seconds == 10.0


def test_night_window_wraps_midnight_default():
    config = AgentConfig()  # 21:00 - 07:00
    assert config.in_night_window(datetime(2026, 1, 1, 23, 0)) is True
    assert config.in_night_window(datetime(2026, 1, 2, 2, 0)) is True
    assert config.in_night_window(datetime(2026, 1, 1, 21, 0)) is True  # inclusive start
    assert config.in_night_window(datetime(2026, 1, 2, 6, 59)) is True
    assert config.in_night_window(datetime(2026, 1, 2, 7, 0)) is False  # exclusive end
    assert config.in_night_window(datetime(2026, 1, 1, 20, 59)) is False
    assert config.in_night_window(datetime(2026, 1, 1, 12, 0)) is False


def test_night_window_zero_width_is_always_on():
    # A caregiver setting AGENT_NIGHT_START == AGENT_NIGHT_END is not
    # deliberately disabling the system forever; treat it as no
    # restriction rather than the empty range a literal `start <= t < end`
    # would produce.
    config = AgentConfig(night_start=time(9, 0), night_end=time(9, 0))
    assert config.in_night_window(datetime(2026, 1, 1, 9, 0)) is True
    assert config.in_night_window(datetime(2026, 1, 1, 0, 0)) is True
    assert config.in_night_window(datetime(2026, 1, 1, 23, 59)) is True


def test_night_window_non_wrapping_range():
    config = AgentConfig(night_start=time(1, 0), night_end=time(5, 0))
    assert config.in_night_window(datetime(2026, 1, 1, 1, 0)) is True
    assert config.in_night_window(datetime(2026, 1, 1, 3, 0)) is True
    assert config.in_night_window(datetime(2026, 1, 1, 4, 59)) is True
    assert config.in_night_window(datetime(2026, 1, 1, 5, 0)) is False
    assert config.in_night_window(datetime(2026, 1, 1, 0, 59)) is False
    assert config.in_night_window(datetime(2026, 1, 1, 22, 0)) is False
