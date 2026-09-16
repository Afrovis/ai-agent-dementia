"""Decrypts the committed WebCrypto vectors (HANDOFF.md section 5.4):
`scripts/make_vector.mjs` makes them with the browser's crypto code
(`static/crypto.js`); this asserts the Python side agrees on every byte of
the wire format -- AAD strings, IV placement, and the result stream's
length-prefixed blocks.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from volunteer_common import crypto

VECTORS_DIR = Path(__file__).parent / "vectors"


def _b64(value: str) -> bytes:
    return base64.b64decode(value)


def test_chunk_vector_decrypts() -> None:
    vector = json.loads((VECTORS_DIR / "chunk_vector.json").read_text())
    key = _b64(vector["key"])
    plaintext = crypto.decrypt_chunk(
        key,
        vector["submission_id"],
        vector["n"],
        _b64(vector["ciphertext_with_iv"]),
    )
    assert plaintext == _b64(vector["plaintext"])


def test_result_vector_decrypts() -> None:
    vector = json.loads((VECTORS_DIR / "result_vector.json").read_text())
    key = _b64(vector["key"])
    plaintext = crypto.decrypt_result(key, vector["submission_id"], _b64(vector["stream"]))
    assert plaintext == _b64(vector["plaintext"])

    blocks = crypto.iter_result_blocks_from_stream(_b64(vector["stream"]))
    assert len(blocks) == 2
    assert blocks[-1].is_last is True
