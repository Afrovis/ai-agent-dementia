from __future__ import annotations

import os

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from volunteer_common import crypto


@pytest.fixture(scope="module")
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    return private_key, private_key.public_key()


def test_wrap_unwrap_round_trip(keypair) -> None:
    private_key, public_key = keypair
    key = os.urandom(crypto.AES_KEY_BYTES)
    wrapped = crypto.wrap_key(public_key, key)
    assert crypto.unwrap_key(private_key, wrapped) == key


def test_unwrap_rejects_wrong_length_key(keypair) -> None:
    private_key, public_key = keypair
    wrapped = public_key.encrypt(
        b"too-short",
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    with pytest.raises(ValueError):
        crypto.unwrap_key(private_key, wrapped)


def test_chunk_round_trip() -> None:
    key = os.urandom(crypto.AES_KEY_BYTES)
    plaintext = b"some chunk of video bytes"
    encrypted = crypto.encrypt_chunk(key, "sub1", 3, plaintext, iv=os.urandom(crypto.GCM_IV_BYTES))
    assert crypto.decrypt_chunk(key, "sub1", 3, encrypted) == plaintext


def test_chunk_rejects_wrong_index_as_aad_tamper() -> None:
    key = os.urandom(crypto.AES_KEY_BYTES)
    encrypted = crypto.encrypt_chunk(key, "sub1", 3, b"payload", iv=os.urandom(crypto.GCM_IV_BYTES))
    with pytest.raises(Exception):
        crypto.decrypt_chunk(key, "sub1", 4, encrypted)


def test_chunk_rejects_tampered_ciphertext() -> None:
    key = os.urandom(crypto.AES_KEY_BYTES)
    encrypted = bytearray(
        crypto.encrypt_chunk(key, "sub1", 0, b"payload", iv=os.urandom(crypto.GCM_IV_BYTES))
    )
    encrypted[-1] ^= 0xFF
    with pytest.raises(Exception):
        crypto.decrypt_chunk(key, "sub1", 0, bytes(encrypted))


def test_result_round_trip_multi_block() -> None:
    key = os.urandom(crypto.AES_KEY_BYTES)
    plaintext = os.urandom(int(crypto.RESULT_BLOCK_BYTES * 2.5))
    stream = crypto.encrypt_result(key, "sub1", plaintext, iv_source=lambda: os.urandom(12))
    assert crypto.decrypt_result(key, "sub1", stream) == plaintext


def test_result_empty_plaintext_is_single_block() -> None:
    key = os.urandom(crypto.AES_KEY_BYTES)
    stream = crypto.encrypt_result(key, "sub1", b"", iv_source=lambda: os.urandom(12))
    blocks = crypto.iter_result_blocks_from_stream(stream)
    assert len(blocks) == 1
    assert blocks[0].is_last is True
    assert crypto.decrypt_result(key, "sub1", stream) == b""


def test_result_truncated_stream_fails_to_decrypt() -> None:
    key = os.urandom(crypto.AES_KEY_BYTES)
    plaintext = os.urandom(crypto.RESULT_BLOCK_BYTES + 100)
    stream = crypto.encrypt_result(key, "sub1", plaintext, iv_source=lambda: os.urandom(12))
    truncated = stream[: len(stream) - 10]
    with pytest.raises(Exception):
        crypto.decrypt_result(key, "sub1", truncated)


def test_result_wrong_submission_id_fails_aad_check() -> None:
    key = os.urandom(crypto.AES_KEY_BYTES)
    stream = crypto.encrypt_result(key, "sub1", b"hello world", iv_source=lambda: os.urandom(12))
    with pytest.raises(Exception):
        crypto.decrypt_result(key, "sub2", stream)


def test_spki_der_round_trips_through_pem(keypair) -> None:
    _, public_key = keypair
    from cryptography.hazmat.primitives import serialization

    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    der = crypto.load_public_key_spki_der(pem)
    assert der == public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
