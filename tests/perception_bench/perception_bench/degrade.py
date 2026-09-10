"""A night-degradation image transform: approximates infrared from lit RGB.

**This is a de-risking tool, not evidence.** It can reveal a pose backend
that collapses the moment colour disappears and contrast drops -- a real
and useful thing to know before anyone records real infrared -- but it
cannot prove a backend works on infrared. Only tier 3
(`perception_bench.infrared`), scored against real IR clips from the actual
room, can do that. Do not report numbers produced with this transform as
"infrared accuracy" anywhere, including in this bench's own output: they
are labelled "daylight, night-degraded" throughout `report.py` specifically
to keep that distinction visible.

Four steps, applied in order, each approximating one real property of an
IR-lit room at night that a daylight RGB frame does not have:

1. **Greyscale** -- IR sensors have no colour information at all.
2. **Reduced contrast** -- a single IR illuminator lights a room far less
   evenly than daylight through a window; the dynamic range of what comes
   back is narrower.
3. **Sensor noise** -- IR sensors are noisier than daylight RGB sensors,
   especially at the gain levels needed to see anything at night. Modelled
   as additive Gaussian noise on the pixel values.
4. **Vignette falloff** -- a single IR illuminator near the camera lights
   the centre of the frame far better than the corners, unlike daylight
   through a window. Modelled as a radial multiplicative falloff from the
   frame centre.

None of these constants were measured against a real IR camera in this
room; they are a reasonable starting guess for "visibly night-like",
documented as such rather than presented as calibrated.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance


def apply_night_degradation(
    image: Image.Image,
    *,
    contrast_factor: float = 0.55,
    noise_std: float = 18.0,
    vignette_strength: float = 0.6,
    rng: random.Random | None = None,
) -> Image.Image:
    """Return a new image approximating how `image` might look under IR at
    night. `image` is unmodified.

    `contrast_factor` is passed straight to `PIL.ImageEnhance.Contrast`
    (1.0 is unchanged, lower is flatter). `noise_std` is the standard
    deviation of additive Gaussian noise, in 0-255 pixel units.
    `vignette_strength` is how dark the frame corners get relative to the
    centre, 0.0 (no vignette) to 1.0 (corners driven to black). `rng`
    seeds the noise for reproducible test fixtures; a fresh
    `random.Random()` is used if not given, so repeated calls in
    production code (tier 2 scoring) see different noise per frame, the
    same as a real sensor would.
    """
    rng = rng if rng is not None else random.Random()

    grey = image.convert("L").convert("RGB")
    low_contrast = ImageEnhance.Contrast(grey).enhance(contrast_factor)

    array = np.asarray(low_contrast).astype(np.float32)
    noise = _gaussian_noise(array.shape, noise_std, rng)
    noisy = np.clip(array + noise, 0, 255)

    vignette = _vignette_mask(array.shape[1], array.shape[0], vignette_strength)
    vignetted = np.clip(noisy * vignette[:, :, None], 0, 255).astype(np.uint8)

    return Image.fromarray(vignetted, mode="RGB")


def _gaussian_noise(shape: tuple[int, ...], std: float, rng: random.Random) -> np.ndarray:
    """Additive Gaussian noise of `shape`, seeded from `rng` so a given
    `random.Random` seed reproduces exactly, without depending on numpy's
    own global random state (which callers may be using for something
    else entirely, e.g. building synthetic fixtures)."""
    seed = rng.randint(0, 2**32 - 1)
    generator = np.random.default_rng(seed)
    return generator.normal(loc=0.0, scale=std, size=shape).astype(np.float32)


def _vignette_mask(width: int, height: int, strength: float) -> np.ndarray:
    """A `(height, width)` multiplicative mask, 1.0 at the centre falling
    to `1.0 - strength` at the frame corners, radially."""
    ys, xs = np.mgrid[0:height, 0:width]
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    max_radius = np.hypot(center_x, center_y)
    radius = np.hypot(xs - center_x, ys - center_y) / max_radius
    return 1.0 - strength * np.clip(radius, 0.0, 1.0)
