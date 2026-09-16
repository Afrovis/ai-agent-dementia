"""Generate the RSA-3072 key pair (HANDOFF.md section 5, V2).

Refuses to overwrite an existing private key: losing it makes every stored
clip unreadable, so a re-run must never silently replace it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

KEY_SIZE = 3072


def generate_keypair(private_key_path: Path, public_key_path: Path) -> None:
    if private_key_path.exists():
        raise FileExistsError(f"{private_key_path} already exists; refusing to overwrite it")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=KEY_SIZE)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(private_key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, private_pem)
    finally:
        os.close(fd)
    public_key_path.write_bytes(public_pem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-dir", default=os.environ.get("VOLUNTEER_KEY_DIR", "/keys-rw"))
    parser.add_argument("--data-dir", default=os.environ.get("VOLUNTEER_DATA_DIR_MOUNT", "/data"))
    args = parser.parse_args(argv)

    private_key_path = Path(args.key_dir) / "private.pem"
    public_key_path = Path(args.data_dir) / "public.pem"
    try:
        generate_keypair(private_key_path, public_key_path)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {private_key_path} and {public_key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
