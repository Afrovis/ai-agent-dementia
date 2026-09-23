"""Tests for `embodiment.main` helpers (no real model, certs, or server)."""

from pathlib import Path

import pytest

from embodiment.main import _clock_time, ssl_kwargs_for
from embodiment.tts import PiperSpeech


def test_ssl_kwargs_for_returns_empty_dict_when_certs_missing(tmp_path: Path):
    missing_cert = tmp_path / "lan.pem"
    missing_key = tmp_path / "lan-key.pem"

    assert ssl_kwargs_for(str(missing_cert), str(missing_key)) == {}


def test_ssl_kwargs_for_returns_paths_when_both_certs_exist(tmp_path: Path):
    cert_file = tmp_path / "lan.pem"
    cert_key = tmp_path / "lan-key.pem"
    cert_file.write_text("cert")
    cert_key.write_text("key")

    result = ssl_kwargs_for(str(cert_file), str(cert_key))

    assert result == {"ssl_certfile": str(cert_file), "ssl_keyfile": str(cert_key)}


def test_ssl_kwargs_for_returns_empty_dict_when_only_cert_exists(tmp_path: Path):
    cert_file = tmp_path / "lan.pem"
    cert_key = tmp_path / "lan-key.pem"
    cert_file.write_text("cert")

    assert ssl_kwargs_for(str(cert_file), str(cert_key)) == {}


def test_piper_from_model_rejects_invalid_speed_before_loading_a_model(tmp_path: Path):
    missing_model = tmp_path / "voice.onnx"

    try:
        PiperSpeech.from_model(missing_model, tmp_path / "cache", speed=0)
    except ValueError as exc:
        assert "PIPER_SPEED" in str(exc)
    else:
        raise AssertionError("invalid Piper speed was accepted")


@pytest.mark.parametrize("value", ["24:00", "12:60", "7:00", "07:0", "noon"])
def test_clock_time_rejects_invalid_hhmm(value):
    with pytest.raises(ValueError, match="EMBODIMENT_NIGHT_START"):
        _clock_time(value, "EMBODIMENT_NIGHT_START")


def test_clock_time_accepts_day_boundaries():
    assert _clock_time("00:00", "EMBODIMENT_NIGHT_START") == "00:00"
    assert _clock_time("23:59", "EMBODIMENT_NIGHT_END") == "23:59"
