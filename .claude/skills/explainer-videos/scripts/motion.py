"""Shared drawing kit for the Night Companion explainer videos (1080p, 30 fps).

Extracted from the first pipeline-hub render (2026-09-18) so every explainer uses
the same palette, fonts and primitives. Fonts live in WORK/fonts (OFL, Google Fonts). Everything is drawn at 2x (S) and downsampled.
"""

from __future__ import annotations

import math
import os
import subprocess
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def _find_data() -> Path:
    """NC_DATA, else the first data-ai-agent-dementia/ beside an ancestor of this file.

    Walking up works from the main checkout and from .claude/worktrees/* alike.
    """
    if os.environ.get("NC_DATA"):
        return Path(os.environ["NC_DATA"])
    for p in Path(__file__).resolve().parents:
        if (p / "data-ai-agent-dementia").is_dir():
            return p / "data-ai-agent-dementia"
    raise SystemExit("data-ai-agent-dementia/ not found next to the repo; set NC_DATA")


DATA = _find_data()
WORK = Path(os.environ.get("DEMO_WORK", DATA / "analysis" / "demo-videos"))
FONTS = WORK / "fonts"
W, H, S, FPS = 1920, 1080, 2, 30

GROUND = (245, 244, 240)
PAPER = (251, 250, 247)
LINE = (201, 198, 190)
LINE_SOFT = (220, 218, 211)
INK = (43, 46, 51)
GRAY = (98, 102, 109)
FAINT = (154, 157, 163)
WARM = (217, 134, 28)
WARM_BG = (251, 238, 219)
WARM_INK = (138, 82, 12)
COOL = (47, 143, 196)
COOL_BG = (227, 240, 247)
ALERT = (196, 85, 58)
ALERT_BG = (248, 228, 222)
ALERT_PILL = (241, 207, 197)
ALERT_INK = (142, 53, 32)


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=None)
def font(face: str, size: float, var: str) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONTS / f"{face}.ttf"), int(round(size * S)))
    f.set_variation_by_name(var)
    return f


def mono(size, var="Regular"):
    return font("JetBrainsMono", size, var)


def sans(size, var="Regular"):
    return font("InstrumentSans", size, var)


def serif(size, var="Regular"):
    return font("Newsreader", size, var)


def serif_i(size, var="Italic"):
    return font("Newsreader-Italic", size, var)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, a, b):
    return clamp((t - a) / (b - a)) if b > a else float(t >= a)


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def mix(c, bg, a):
    a = clamp(a)
    return tuple(int(round(bg[i] + (c[i] - bg[i]) * a)) for i in range(3))


def sc(v):
    return int(round(v * S))


def wrap(text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if f.getlength(trial) <= width * S or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (W * S, H * S), GROUND)
        self.d = ImageDraw.Draw(self.im)

    def rrect(self, x, y, w, h, r, fill=None, outline=None, width=1.5, dash=False):
        box = [sc(x), sc(y), sc(x + w), sc(y + h)]
        if dash and outline:
            if fill:
                self.d.rounded_rectangle(box, sc(r), fill=fill)
            self.dashed_rect(x, y, w, h, outline, width)
            return
        self.d.rounded_rectangle(box, sc(r), fill=fill, outline=outline, width=sc(width) if outline else 0)

    def dashed_rect(self, x, y, w, h, col, width, on=6, off=5):
        for (x0, y0, x1, y1) in ((x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)):
            self.polyline([(x0, y0), (x1, y1)], col, width, dash=(on, off))

    def text(self, x, y, s, f, col, anchor="la"):
        self.d.text((sc(x), sc(y)), s, font=f, fill=col, anchor=anchor)

    def lines(self, x, y, lines, f, col, lh, chars=None):
        """Draw wrapped lines; `chars` reveals only the first N characters (typing)."""
        left = chars
        for i, ln in enumerate(lines):
            if left is not None:
                if left <= 0:
                    break
                ln = ln[: int(left)]
                left -= len(ln) + 1
            self.text(x, y + i * lh, ln, f, col)

    def polyline(self, pts, col, width, dash=None, frac=1.0):
        pts = _trim(pts, frac)
        if len(pts) < 2:
            return
        spts = [(sc(px), sc(py)) for px, py in pts]
        if not dash:
            self.d.line(spts, fill=col, width=sc(width), joint="curve")
            return
        on, off = dash
        carry, drawing = 0.0, True
        for (x0, y0), (x1, y1) in zip(spts, spts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            pos = 0.0
            while pos < seg:
                span = (on if drawing else off) * S - carry
                end = min(seg, pos + span)
                if drawing:
                    a, b = pos / seg, end / seg
                    self.d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                                fill=col, width=sc(width))
                if end - pos < span:
                    carry += end - pos
                    break
                carry, drawing, pos = 0.0, not drawing, end
            else:
                continue

    def arrow_head(self, tip, prev, col, size=11):
        ang = math.atan2(tip[1] - prev[1], tip[0] - prev[0])
        a1, a2 = ang + 2.65, ang - 2.65
        pts = [tip, (tip[0] + size * math.cos(a1), tip[1] + size * math.sin(a1)),
               (tip[0] + size * math.cos(a2), tip[1] + size * math.sin(a2))]
        self.d.polygon([(sc(px), sc(py)) for px, py in pts], fill=col)

    def wire(self, pts, col, width=2.5, frac=1.0, dash=None, head=True):
        if frac <= 0:
            return
        self.polyline(pts, col, width, dash=dash, frac=frac)
        if head and frac >= 0.999:
            self.arrow_head(pts[-1], pts[-3] if len(pts) > 3 else pts[0], col)

    def paste(self, img, x, y, w, h, r, alpha=1.0):
        tile = img.resize((sc(w), sc(h)), Image.LANCZOS)
        mask = Image.new("L", tile.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, tile.size[0] - 1, tile.size[1] - 1], sc(r),
                                               fill=int(255 * clamp(alpha)))
        self.im.paste(tile, (sc(x), sc(y)), mask)

    def final(self):
        return self.im.resize((W, H), Image.LANCZOS)


def _trim(pts, frac):
    if frac >= 1:
        return pts
    lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    target, out = sum(lens) * clamp(frac), [pts[0]]
    for (a, b), L in zip(zip(pts, pts[1:]), lens):
        if target <= L:
            k = target / L if L else 0
            out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
            return out
        target -= L
        out.append(b)
    return out


def bezier(p0, p1, p2, p3, n=40):
    return [(((1 - t) ** 3) * p0[0] + 3 * ((1 - t) ** 2) * t * p1[0] + 3 * (1 - t) * t * t * p2[0] + t ** 3 * p3[0],
             ((1 - t) ** 3) * p0[1] + 3 * ((1 - t) ** 2) * t * p1[1] + 3 * (1 - t) * t * t * p2[1] + t ** 3 * p3[1])
            for t in (i / n for i in range(n + 1))]



# ---------------------------------------------------------------- extra primitives
def chip(cv, x, y, label, a=1.0, fg=None, bg=None, size=13, anchor="mm"):
    """An event-name pill (e.g. "PersonState") centred on (x, y); returns its width."""
    fg = fg or INK
    bg = bg or PAPER
    f = mono(size, "Medium")
    tw = f.getlength(label) / S
    w, h = tw + 22, size + 15
    x0 = x - w / 2 if anchor == "mm" else x
    cv.rrect(x0, y - h / 2, w, h, h / 2, fill=mix(bg, GROUND, a), outline=mix(fg, GROUND, a * 0.55), width=1.2)
    cv.text(x0 + 11, y - h / 2 + 7, label, f, mix(fg, bg, a))
    return w


def point_at(pts, frac):
    """Point a fraction `frac` of the way along polyline `pts`."""
    trimmed = _trim(pts, clamp(frac))
    return trimmed[-1]


def fade_all(im, real_t, dur, ground=None):
    f = ease(prog(real_t, 0, 0.4)) * (1 - ease(prog(real_t, dur - 0.6, dur)))
    if f < 1:
        im = Image.blend(Image.new("RGB", im.size, ground or GROUND), im, f)
    return im


def encode(render_one, n_frames, out, init=None, initargs=(), workers=8):
    """Render frames in a pool and pipe them to ffmpeg (H.264, CRF 16, yuv420p, faststart)."""
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    with Pool(workers, initializer=init, initargs=initargs) as pool:
        for k, buf in enumerate(pool.imap(render_one, range(n_frames), chunksize=4)):
            ff.stdin.write(buf)
            if k % 300 == 0:
                print(f"frame {k}/{n_frames}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)
