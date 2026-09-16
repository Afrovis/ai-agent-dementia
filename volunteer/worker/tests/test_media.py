from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from volunteer_common import crypto, paths

from volunteer_worker import media

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _make_test_video(tmp_path: Path) -> bytes:
    video_path = tmp_path / "source.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=64x64:rate=5",
            "-c:v",
            "libvpx",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    return video_path.read_bytes()


@pytest.fixture(scope="module")
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
def test_decrypt_chunks_and_remux_round_trip(tmp_path: Path, keypair) -> None:
    private_key, public_key = keypair
    video_bytes = _make_test_video(tmp_path)

    submission_id = "sub-media-test-0000001"
    aes_key = b"\x01" * crypto.AES_KEY_BYTES
    wrapped_key = crypto.wrap_key(public_key, aes_key)
    paths.key_wrapped_path(tmp_path, submission_id).parent.mkdir(parents=True, exist_ok=True)
    paths.key_wrapped_path(tmp_path, submission_id).write_bytes(wrapped_key)

    chunk_size = 4096
    chunks = [video_bytes[i : i + chunk_size] for i in range(0, len(video_bytes), chunk_size)]
    for n, chunk_plaintext in enumerate(chunks):
        iv = os.urandom(crypto.GCM_IV_BYTES)
        encrypted = crypto.encrypt_chunk(aes_key, submission_id, n, chunk_plaintext, iv=iv)
        paths.chunk_path(tmp_path, submission_id, n).write_bytes(encrypted)

    raw_path = media.decrypt_submission_to_raw(tmp_path, submission_id, len(chunks), private_key)

    assert raw_path.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", str(raw_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip()


def test_remux_raises_on_garbage_input(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        media.remux(b"not a real video container", tmp_path / "raw.mkv")
