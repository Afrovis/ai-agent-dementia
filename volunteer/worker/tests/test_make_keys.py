from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization

from volunteer_worker.make_keys import generate_keypair


def test_generates_private_and_public_key(tmp_path: Path) -> None:
    private_path = tmp_path / "keys" / "private.pem"
    public_path = tmp_path / "data" / "public.pem"
    generate_keypair(private_path, public_path)

    assert oct(private_path.stat().st_mode)[-3:] == "600"
    private_key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    assert private_key.key_size == 3072
    public_key = serialization.load_pem_public_key(public_path.read_bytes())
    assert public_key.public_numbers() == private_key.public_key().public_numbers()


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    private_path = tmp_path / "keys" / "private.pem"
    public_path = tmp_path / "data" / "public.pem"
    generate_keypair(private_path, public_path)
    original = private_path.read_bytes()

    with pytest.raises(FileExistsError):
        generate_keypair(private_path, public_path)
    assert private_path.read_bytes() == original
