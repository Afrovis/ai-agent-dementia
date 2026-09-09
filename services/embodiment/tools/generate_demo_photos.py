"""Regenerate the bundled demo photos in `embodiment/demo_photos/`.

A pure-stdlib PNG writer, so the repo carries small, reproducible
placeholder images without adding an image library to the service's runtime
dependencies. Output is deterministic: running this twice produces
byte-identical files.

The images are deliberately dark and low contrast. They render behind the
face at 0.6 opacity, under the night brightness overlay, so anything busy or
bright would fight the text the page exists to show.

Usage:

    python services/embodiment/tools/generate_demo_photos.py \
        services/embodiment/embodiment/demo_photos
"""

from __future__ import annotations

import struct
import sys
import zlib
from collections.abc import Callable
from pathlib import Path

WIDTH = 960
HEIGHT = 640

Pixel = tuple[float, float, float]
PixelFn = Callable[[int, int, int, int], Pixel]


def _chunk(tag: bytes, data: bytes) -> bytes:
    """Return one PNG chunk: length, tag, payload, CRC."""
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_png(path: Path, width: int, height: int, pixel_fn: PixelFn) -> int:
    """Write an 8-bit RGB PNG at `path`, colouring each pixel via `pixel_fn`.

    Returns the number of bytes written. Every scanline uses filter type 0
    (None), which keeps the writer trivial at a small cost in size; these
    images are flat gradients, so zlib compresses them well regardless.
    """
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            for value in pixel_fn(x, y, width, height):
                raw.append(max(0, min(255, int(value))))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return len(png)


def glow(x: int, y: int, w: int, h: int, cx: float, cy: float, radius: float) -> float:
    """Soft radial falloff in [0, 1], centred on (`cx`, `cy`) as fractions of w/h."""
    dx = (x - cx * w) / (radius * w)
    dy = (y - cy * h) / (radius * h)
    distance = (dx * dx + dy * dy) ** 0.5
    return max(0.0, 1.0 - distance) ** 2


def room(x: int, y: int, w: int, h: int) -> Pixel:
    """Warm, dim wash with a lamp glow up and to the left (`orient_time_place`)."""
    t = y / h
    intensity = glow(x, y, w, h, 0.28, 0.34, 0.55)
    return (
        58 - 40 * t + 70 * intensity,
        41 - 29 * t + 52 * intensity,
        27 - 19 * t + 30 * intensity,
    )


def family(x: int, y: int, w: int, h: int) -> Pixel:
    """Cooler, softer wash with a centred glow (`familiar_voice`)."""
    t = y / h
    intensity = glow(x, y, w, h, 0.5, 0.42, 0.62)
    return (
        36 - 24 * t + 46 * intensity,
        42 - 28 * t + 50 * intensity,
        55 - 36 * t + 58 * intensity,
    )


IMAGES: dict[str, PixelFn] = {"demo_room": room, "demo_family": family}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <output-directory>", file=sys.stderr)
        return 2

    out = Path(argv[1])
    if not out.is_dir():
        print(f"not a directory: {out}", file=sys.stderr)
        return 1

    for name, pixel_fn in IMAGES.items():
        size = write_png(out / f"{name}.png", WIDTH, HEIGHT, pixel_fn)
        print(f"{name}.png {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
