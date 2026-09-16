"""Wire format for volunteer clip encryption (HANDOFF.md section 5.4).

Everything here is symmetric with `web/volunteer_web/static/crypto.js`: the
browser encrypts chunks and decrypts results with WebCrypto using the same
constants, and `common/tests/vectors/` pins both directions against fixed
vectors made by `scripts/make_vector.mjs`.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AES_KEY_BYTES = 32
GCM_IV_BYTES = 12
GCM_TAG_BYTES = 16
RESULT_BLOCK_BYTES = 4 * 1024 * 1024
LENGTH_PREFIX_BYTES = 4


def chunk_aad(submission_id: str, n: int) -> bytes:
    return f"chunk:{submission_id}:{n}".encode()


def result_block_aad(submission_id: str, index: int, *, is_last: bool) -> bytes:
    return f"result:{submission_id}:{index}:{1 if is_last else 0}".encode()


def unwrap_key(private_key: rsa.RSAPrivateKey, wrapped_key: bytes) -> bytes:
    """RSA-OAEP-SHA256 unwrap of the browser-generated AES key."""
    key = private_key.decrypt(
        wrapped_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    if len(key) != AES_KEY_BYTES:
        raise ValueError(f"unwrapped key is {len(key)} bytes, expected {AES_KEY_BYTES}")
    return key


def wrap_key(public_key: rsa.RSAPublicKey, key: bytes) -> bytes:
    """Inverse of `unwrap_key`, used by tests to build fixtures without a browser."""
    return public_key.encrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def load_public_key_spki_der(pem_bytes: bytes) -> bytes:
    """Return the SPKI DER form served by `GET /api/config` (`public_key_spki`)."""
    public_key = serialization.load_pem_public_key(pem_bytes)
    return public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def decrypt_chunk(key: bytes, submission_id: str, n: int, ciphertext_with_iv: bytes) -> bytes:
    """`iv (12 bytes) || AES-GCM ciphertext including the 16-byte tag`."""
    if len(ciphertext_with_iv) < GCM_IV_BYTES + GCM_TAG_BYTES:
        raise ValueError("chunk too short to contain an IV and tag")
    iv, ciphertext = ciphertext_with_iv[:GCM_IV_BYTES], ciphertext_with_iv[GCM_IV_BYTES:]
    return AESGCM(key).decrypt(iv, ciphertext, chunk_aad(submission_id, n))


def encrypt_chunk(key: bytes, submission_id: str, n: int, plaintext: bytes, *, iv: bytes) -> bytes:
    """Inverse of `decrypt_chunk`, used by tests and by `make_vector`-equivalent fixtures."""
    if len(iv) != GCM_IV_BYTES:
        raise ValueError(f"iv must be {GCM_IV_BYTES} bytes")
    ciphertext = AESGCM(key).encrypt(iv, plaintext, chunk_aad(submission_id, n))
    return iv + ciphertext


@dataclass(frozen=True)
class ResultBlock:
    index: int
    is_last: bool
    ciphertext_with_iv: bytes


def iter_result_blocks_from_stream(data: bytes) -> list[ResultBlock]:
    """Parse the length-prefixed block stream described in HANDOFF.md 5.4."""
    blocks: list[ResultBlock] = []
    offset = 0
    index = 0
    while offset < len(data):
        if offset + LENGTH_PREFIX_BYTES > len(data):
            raise ValueError("truncated result stream: missing length prefix")
        length = int.from_bytes(data[offset : offset + LENGTH_PREFIX_BYTES], "big")
        offset += LENGTH_PREFIX_BYTES
        body_len = GCM_IV_BYTES + length
        if offset + body_len > len(data):
            raise ValueError("truncated result stream: missing block body")
        body = data[offset : offset + body_len]
        offset += body_len
        is_last = offset >= len(data)
        blocks.append(ResultBlock(index=index, is_last=is_last, ciphertext_with_iv=body))
        index += 1
    return blocks


def decrypt_result(key: bytes, submission_id: str, data: bytes) -> bytes:
    """Decrypt the full length-prefixed result stream into plaintext MP4 bytes."""
    plaintext = bytearray()
    blocks = iter_result_blocks_from_stream(data)
    for block in blocks:
        iv = block.ciphertext_with_iv[:GCM_IV_BYTES]
        ciphertext = block.ciphertext_with_iv[GCM_IV_BYTES:]
        aad = result_block_aad(submission_id, block.index, is_last=block.is_last)
        plaintext += AESGCM(key).decrypt(iv, ciphertext, aad)
    return bytes(plaintext)


def encrypt_result(
    key: bytes,
    submission_id: str,
    plaintext: bytes,
    *,
    iv_source,
    block_size: int = RESULT_BLOCK_BYTES,
) -> bytes:
    """Encrypt plaintext into the length-prefixed block stream `worker` serves.

    `iv_source` is a zero-argument callable returning `GCM_IV_BYTES` random
    bytes, injected so tests can supply deterministic IVs. `block_size`
    defaults to the real 4 MiB plaintext block size; tests and the committed
    two-block vector (`scripts/make_vector.mjs`) pass a tiny one so a
    multi-block fixture does not require multi-megabyte plaintext.
    """
    out = bytearray()
    total = len(plaintext)
    block_count = max(1, (total + block_size - 1) // block_size)
    for index in range(block_count):
        start = index * block_size
        end = min(start + block_size, total)
        block_plaintext = plaintext[start:end]
        is_last = index == block_count - 1
        iv = iv_source()
        aad = result_block_aad(submission_id, index, is_last=is_last)
        ciphertext = AESGCM(key).encrypt(iv, block_plaintext, aad)
        body = iv + ciphertext
        out += len(ciphertext).to_bytes(LENGTH_PREFIX_BYTES, "big")
        out += body
    return bytes(out)
