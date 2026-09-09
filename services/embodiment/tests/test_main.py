"""Tests for `embodiment.main`'s HTTPS-vs-HTTP fallback logic (no real certs, no server)."""

from pathlib import Path

from embodiment.main import ssl_kwargs_for


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
