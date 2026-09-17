"""Decrypt a submission's chunks and remux them into a real container
(HANDOFF.md section 5.8, steps 1-2).

Browser WebM has no duration header, so this always remuxes before
`video_eval prepare` even touches the file (CLAUDE.md gotcha).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import rsa
from volunteer_common import crypto, paths


def decrypt_chunks(
    data_root: Path, submission_id: str, chunk_count: int, private_key: rsa.RSAPrivateKey
) -> bytes:
    wrapped_key = paths.key_wrapped_path(data_root, submission_id).read_bytes()
    key = crypto.unwrap_key(private_key, wrapped_key)
    plaintext = bytearray()
    for n in range(chunk_count):
        ciphertext = paths.chunk_path(data_root, submission_id, n).read_bytes()
        plaintext += crypto.decrypt_chunk(key, submission_id, n, ciphertext)
    return bytes(plaintext)


def remux(decrypted: bytes, raw_path: Path) -> None:
    """`ffmpeg -i pipe:0 -map 0:v:0 -c copy <raw_path>`, falling back to a
    re-encode when the container can't be copied as-is (fragmented Safari
    MP4)."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    copy_result = subprocess.run(
        ["ffmpeg", "-y", "-i", "pipe:0", "-map", "0:v:0", "-c", "copy", str(raw_path)],
        input=decrypted,
        capture_output=True,
    )
    if copy_result.returncode == 0:
        return

    reencode_result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(raw_path),
        ],
        input=decrypted,
        capture_output=True,
    )
    if reencode_result.returncode != 0:
        stderr = reencode_result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg could not remux the recording: {stderr[-2000:]}")


def decrypt_submission_to_raw(
    data_root: Path, submission_id: str, chunk_count: int, private_key: rsa.RSAPrivateKey
) -> Path:
    decrypted = decrypt_chunks(data_root, submission_id, chunk_count, private_key)
    raw_path = paths.raw_path(data_root, submission_id)
    remux(decrypted, raw_path)
    return raw_path
