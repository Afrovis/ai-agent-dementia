"""Pure, no-I/O helpers for comparing two JPEG frames for motion.

Kept separate from `gate.py` (which owns the *stateful* admit-or-drop
decision and the clock) so the actual pixel comparison is trivial to unit
test with tiny in-process Pillow images and no camera, no bus, and no time
mocking (HANDOFF.md: "every service must be testable with no camera, mic,
Ollama, or Redis").
"""

from __future__ import annotations

import io

from PIL import Image

SIGNATURE_SIZE = 16
"""Side length, in pixels, of the downscaled grayscale thumbnail used as a
frame's motion signature. Small on purpose: 16x16 = 256 values is plenty to
tell "the room changed" from "the room did not", and cheap enough to compute
on every admitted frame without a GPU."""


def frame_signature(jpeg: bytes) -> tuple[int, ...]:
    """Decode `jpeg` and return a coarse grayscale signature for motion comparison.

    The image is converted to grayscale ("L" mode) and resized to
    `SIGNATURE_SIZE` x `SIGNATURE_SIZE`, then flattened to a tuple of
    `SIGNATURE_SIZE ** 2` luma values (0 to 255). Resizing this small
    deliberately throws away compression noise and fine detail, which
    matters for a night camera: JPEG artifacts and IR grain must not look
    like motion. Returns an empty tuple if `jpeg` cannot be decoded, so a
    corrupt or truncated frame degrades to "signature unknown" rather than
    raising, which `motion_score` treats as "assume motion" (fail loud, not
    quiet -- HANDOFF.md rule 4).
    """
    try:
        with Image.open(io.BytesIO(jpeg)) as image:
            grayscale = image.convert("L").resize((SIGNATURE_SIZE, SIGNATURE_SIZE))
            # `tobytes()` rather than the deprecated `getdata()`: in "L"
            # mode one byte is one luma value, so this is the same 256 ints.
            return tuple(grayscale.tobytes())
    except Exception:  # noqa: BLE001 - any decode failure means "no signature"
        return ()


def motion_score(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Return how different two frame signatures are, normalised to 0.0-1.0.

    Computed as the mean absolute difference of the two signatures' luma
    values, divided by 255 (the maximum possible per-pixel difference). An
    empty signature or a length mismatch (e.g. one frame failed to decode)
    is treated as maximum motion (`1.0`): the gate then admits the frame
    rather than silently sitting on a comparison it cannot trust.
    """
    if not a or not b or len(a) != len(b):
        return 1.0
    total_difference = sum(abs(x - y) for x, y in zip(a, b, strict=True))
    return (total_difference / len(a)) / 255.0
