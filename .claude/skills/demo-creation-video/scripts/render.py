"""Render a Direction A demo clip from a snippet's 30 fps pose JSONL over the source video.

    python render.py <clip_id> <snippet_name> [--pred FILE] [--label "Bedroom sample 02"]

Everything is drawn on a 4K canvas and downsampled to 1080p, which gives anti-aliased
lines without a vector library. Joints are smoothed with a One Euro filter. The state
label, timeline and floor chip come from the clip's real 2 fps pipeline predictions;
the TRUTH lane from labels/reference.yaml; both are skipped when missing.
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

def _find_data():
    """NC_DATA, else the first data-ai-agent-dementia/ found beside an ancestor of this file.

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
BW, BH = 3840, 2160  # drawing canvas
S = 2.0  # design units are 1080p px
OUT_W, OUT_H = 1920, 1080
FPS = 30
DUR = 1.0  # set per clip from clip.yaml
FIT = (0.0, 0.0, 1920.0, 1080.0)  # source placement in 1080p units (x, y, w, h); non-16:9 sources are fitted, not stretched


def to_px(x, y):
    return FIT[0] + x * FIT[2], FIT[1] + y * FIT[3]

INK = (14, 16, 18)
IVORY = (243, 239, 230)
MUTED = (169, 165, 156)
WARM = (242, 191, 134)
COOL = (147, 207, 227)
STATE = {
    "in_bed": ((143, 194, 157), "In bed"),
    "sitting_up": ((220, 182, 108), "Sitting up"),
    "standing": ((141, 180, 219), "Standing"),
    "walking": ((127, 208, 199), "Walking"),
    "on_floor": ((236, 122, 95), "On the floor"),
    "absent": ((111, 115, 122), "Not in view"),
    None: ((74, 78, 85), "Warming up"),
}
L_LIMBS = [(11, 13, 7.0), (13, 15, 5.5), (5, 7, 6.0), (7, 9, 4.5)]
R_LIMBS = [(12, 14, 7.0), (14, 16, 5.5), (6, 8, 6.0), (8, 10, 4.5)]
TRAIL_JOINTS = (9, 10, 15, 16)
TRAIL_S = 0.5


def font(name, size, weight=None, opsz=None):
    f = ImageFont.truetype(str(FONTS / name), int(size * S))
    axes = []
    try:
        for ax in f.get_variation_axes():
            n = ax["name"] if isinstance(ax["name"], str) else ax["name"].decode()
            if n.lower().startswith("weight"):
                axes.append(weight or 400)
            elif n.lower().startswith("optical"):
                axes.append(opsz or min(max(size, ax["minimum"]), ax["maximum"]))
            elif n.lower().startswith("width"):
                axes.append(100)
            else:
                axes.append(ax["default"])
        f.set_variation_by_axes(axes)
    except OSError:
        pass
    return f


F_TITLE = font("InstrumentSans.ttf", 22, 600)
F_MONO_S = font("JetBrainsMono.ttf", 13, 500)
F_MONO_M = font("JetBrainsMono.ttf", 15, 400)
F_MONO_L = font("JetBrainsMono.ttf", 18, 400)
F_STATE = font("Newsreader.ttf", 52, 400)
F_CHIP = font("InstrumentSans.ttf", 22, 500)


# ---------------------------------------------------------------- smoothing
class OneEuro:
    def __init__(self, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _a(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt=1 / FPS):
        if self.x is None:
            self.x, self.dx = x, 0.0
            return x
        dx = (x - self.x) / dt
        a_d = self._a(self.dc, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._a(self.mc + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


def smooth(records):
    """Returns per-frame list of 17 (x, y, c) in 1080p px, or None; plus a presence alpha."""
    fx = [OneEuro() for _ in range(17)]
    fy = [OneEuro() for _ in range(17)]
    conf = [0.0] * 17
    last = None
    presence = 0.0
    out = []
    for r in records:
        if r["kp"]:
            pts = []
            for j, (x, y, c) in enumerate(r["kp"]):
                px, py = to_px(x, y)
                if c < 0.2 and last is not None:
                    px, py = last[j][0], last[j][1]  # hold rather than chase a guess
                conf[j] = 0.65 * conf[j] + 0.35 * c
                pts.append((fx[j](px), fy[j](py), conf[j]))
            last = pts
            presence = min(1.0, presence + 1 / (0.25 * FPS))
        else:
            presence = max(0.0, presence - 1 / (0.3 * FPS))
        out.append((last, presence))
    return out


# ---------------------------------------------------------------- drawing
def P(x, y):
    return x * S, y * S


def stroke(base, pts, width, color, opacity, dots=False):
    """Round-capped polyline with true alpha (no double-blending at joints)."""
    if opacity <= 0.01 or len(pts) < 2:
        return
    w = width * S
    xs = [p[0] * S for p in pts]
    ys = [p[1] * S for p in pts]
    pad = w + 2
    x0, y0 = int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad))
    x1, y1 = int(min(BW, max(xs) + pad)), int(min(BH, max(ys) + pad))
    if x1 <= x0 or y1 <= y0:
        return
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    d = ImageDraw.Draw(mask)
    loc = [(x - x0, y - y0) for x, y in zip(xs, ys)]
    if dots:
        n = int(sum(math.dist(loc[i], loc[i + 1]) for i in range(len(loc) - 1)) / (10 * S))
        a, b = loc[0], loc[-1]
        for k in range(n + 1):
            u = k / max(1, n)
            cx, cy = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
            d.ellipse((cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2), fill=255)
    else:
        d.line(loc, fill=255, width=int(round(w)), joint="curve")
        for cx, cy in (loc[0], loc[-1]):
            d.ellipse((cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2), fill=255)
    if opacity < 1:
        mask = mask.point(lambda v: int(v * opacity))
    base.paste(Image.new("RGB", mask.size, color), (x0, y0), mask)


def fill_poly(base, pts, color, opacity):
    xs = [p[0] * S for p in pts]
    ys = [p[1] * S for p in pts]
    x0, y0, x1, y1 = int(max(0, min(xs))), int(max(0, min(ys))), int(min(BW, max(xs)) + 1), int(min(BH, max(ys)) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    mask = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in zip(xs, ys)], fill=int(255 * opacity))
    base.paste(Image.new("RGB", mask.size, color), (x0, y0), mask)


def text(base, xy, s, f, color, opacity=1.0, spacing=0.0, anchor="ls"):
    """Text with optional letter-spacing (design px) and alpha."""
    if opacity <= 0.01:
        return 0
    x, y = xy[0] * S, xy[1] * S
    if anchor.startswith("r"):
        x -= text_width(s, f, spacing) * S
    w = text_width(s, f, spacing) * S
    asc, desc = f.getmetrics()
    x0, y0 = int(x - 4), int(y - asc - 4)
    mask = Image.new("L", (int(w + 8), asc + desc + 8), 0)
    d = ImageDraw.Draw(mask)
    cx = 4.0
    for ch in s:
        d.text((cx, 4 + asc), ch, font=f, fill=int(255 * opacity), anchor="ls")
        cx += f.getlength(ch) + spacing * S
    base.paste(Image.new("RGB", mask.size, color), (x0, y0), mask)
    return w / S


def text_width(s, f, spacing=0.0):
    return (sum(f.getlength(ch) for ch in s) + spacing * S * max(0, len(s) - 1)) / S


def skeleton(base, pts, presence, trails):
    if pts is None or presence <= 0:
        return

    def ok(j, th=0.35):
        return pts[j][2] >= th

    def op(c):
        return presence * (0.25 + 0.75 * max(0.0, min(1.0, (c - 0.15) / 0.55)))

    # trails
    for j, hist in trails.items():
        col = WARM if j % 2 == 1 else COOL
        n = len(hist)
        for i in range(n - 1):
            if hist[i] is None or hist[i + 1] is None:
                continue
            u = (i + 1) / n
            stroke(base, [hist[i], hist[i + 1]], 1.2 + 2.3 * u, col, presence * (0.05 + 0.55 * u))
    # casing
    edges = [(a, b) for a, b, _ in L_LIMBS + R_LIMBS] + [(5, 6), (11, 12), (5, 11), (6, 12)]
    for a, b in edges:
        if ok(a) and ok(b):
            stroke(base, [pts[a][:2], pts[b][:2]], 11, INK, 0.5 * presence)
    # torso plate
    if all(ok(j, 0.3) for j in (5, 6, 11, 12)):
        quad = [pts[5][:2], pts[6][:2], pts[12][:2], pts[11][:2]]
        fill_poly(base, quad, IVORY, 0.08 * presence)
        stroke(base, quad + [quad[0]], 2, IVORY, 0.55 * presence)
    # limbs
    for limbs, col in ((L_LIMBS, WARM), (R_LIMBS, COOL)):
        for a, b, w in limbs:
            c = min(pts[a][2], pts[b][2])
            if c < 0.12:
                continue
            if c < 0.35:
                stroke(base, [pts[a][:2], pts[b][:2]], 3, col, op(c), dots=True)
            else:
                stroke(base, [pts[a][:2], pts[b][:2]], w, col, op(c))
    # neck + nose
    if ok(0, 0.3) and ok(5) and ok(6):
        mx, my = (pts[5][0] + pts[6][0]) / 2, (pts[5][1] + pts[6][1]) / 2
        ex, ey = mx + (pts[0][0] - mx) * 0.55, my + (pts[0][1] - my) * 0.55
        stroke(base, [(mx, my), (ex, ey)], 10, INK, 0.5 * presence)
        stroke(base, [(mx, my), (ex, ey)], 4, IVORY, 0.85 * presence)
    # joints
    d = ImageDraw.Draw(base)
    for j in range(5, 17):
        if not ok(j) or presence < 0.6:
            continue
        col = WARM if j % 2 == 1 else COOL
        x, y = P(*pts[j][:2])
        r = 5 * S
        d.ellipse((x - r, y - r, x + r, y + r), fill=INK, outline=col, width=int(2.5 * S))


def bed_zone(base, bed, active, color):
    pts = [to_px(x, y) for x, y in bed]
    if active:
        fill_poly(base, pts, color, 0.13)
        stroke(base, pts + [pts[0]], 2.5, color, 1.0)
    else:
        # dotted outline
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            n = max(1, int(math.dist(a, b) / 9))
            d = ImageDraw.Draw(base, "RGBA")
            for k in range(n):
                u = k / n
                x, y = P(a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
                d.ellipse((x - 1.6 * S, y - 1.6 * S, x + 1.6 * S, y + 1.6 * S), fill=IVORY + (50,))
    top = min(pts, key=lambda p: p[1])
    text(base, (top[0] - 4, top[1] - 14), "BED", F_MONO_S, color if active else MUTED, 1.0 if active else 0.6, spacing=1.6)


# ---------------------------------------------------------------- HUD
def segments_ref(ref):
    return [(s["from_s"], s["to_s"], s["state"]) for s in ref]


def segments_pred(rows):
    out, prev, start = [], None, 0.0
    for r in rows:
        if r["state"] != prev:
            if prev is not None or out:
                out.append((start, r["t_s"], prev))
            prev, start = r["state"], r["t_s"]
    out.append((start, DUR, prev))
    return out


def timeline(base, x, y, w, t, lanes, window):
    lx, lw = x + 84, w - 84
    d = ImageDraw.Draw(base, "RGBA")
    # snippet window bracket
    wx0, wx1 = lx + window[0] / DUR * lw, lx + window[1] / DUR * lw
    d.rounded_rectangle((*P(wx0 - 4, y - 14), *P(wx1 + 4, y + 50)), radius=6 * S, fill=IVORY + (18,))
    for i, (name, segs) in enumerate(lanes):
        yy = y + i * 26
        text(base, (x, yy + 9), name, F_MONO_S, MUTED, spacing=1.8)
        for a, b, st in segs:
            col = STATE[st][0]
            alpha = 255 if a <= t else 90
            x0, x1 = lx + a / DUR * lw + 1, lx + b / DUR * lw - 1
            if x1 - x0 < 1:
                continue
            d.rounded_rectangle((*P(x0, yy), *P(x1, yy + 10)), radius=5 * S, fill=col + (alpha,))
    px = lx + t / DUR * lw
    d.rounded_rectangle((*P(px - 1, y - 10), *P(px + 1, y + 46)), radius=1 * S, fill=IVORY + (255,))


def scrim(h_px):
    """Bottom fade so the HUD reads without hiding the floor."""
    a = np.zeros((BH, 1), np.float32)
    top = BH - int(h_px * S)
    ramp = np.linspace(0, 1, BH - top) ** 1.6
    a[top:, 0] = ramp * 0.78
    return a


def vignette():
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    nx, ny = (xx / BW - 0.5) * 2, (yy / BH - 0.5) * 2
    r = np.sqrt(nx**2 * 0.8 + ny**2)
    return np.clip(1 - 0.45 * np.clip(r - 0.55, 0, None) ** 1.5, 0, 1)


# ---------------------------------------------------------------- inputs
def pick_prediction(clip_dir):
    """Prefer the tuned replay run; else the newest yolo run that carries a state field."""
    pdir = clip_dir / "predictions"
    cands = sorted(pdir.glob("replay_*improved*.jsonl")) or sorted(pdir.glob("*yolo*-g.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        with open(p) as fh:
            if "state" in json.loads(fh.readline()):
                return p
    return None


def unletterbox(pts, src_w, src_h):
    """Bridge frames letterbox the source into 4:3; map bridge-normalised coords back to the source."""
    a = src_w / src_h
    if a >= 4 / 3:
        frac = (4 / 3) / a
        off = (1 - frac) / 2
        return [[x, (y - off) / frac] for x, y in pts]
    frac = a / (4 / 3)
    off = (1 - frac) / 2
    return [[(x - off) / frac, y] for x, y in pts]


def default_label(clip):
    tail = clip.split("_", 1)[-1].replace("-", " ")
    return tail[:1].upper() + tail[1:]


# ---------------------------------------------------------------- main
def render(clip, name, pred_path=None, label=None):
    global DUR, FIT
    clip_dir = DATA / "clips" / clip
    meta = yaml.safe_load(open(clip_dir / "clip.yaml"))
    DUR = float(meta["video"]["duration_s"])
    src_w, src_h = meta["video"]["width"], meta["video"]["height"]
    if meta["video"].get("rotation") in (90, 270, -90):
        src_w, src_h = src_h, src_w
    if src_w < src_h:
        raise SystemExit("portrait source: the HUD layout is landscape-only so far (see SKILL.md, Open problems)")
    label = label or default_label(clip)
    a = src_w / src_h
    fw, fh = (1920.0, 1920.0 / a) if a >= 16 / 9 else (1080.0 * a, 1080.0)
    FIT = ((1920.0 - fw) / 2, (1080.0 - fh) / 2, fw, fh)

    recs = [json.loads(l) for l in open(WORK / clip / f"{name}.pose.jsonl")]
    start, end = recs[0]["t"], recs[-1]["t"]
    sm = smooth(recs)

    pred_path = Path(pred_path) if pred_path else pick_prediction(clip_dir)
    pred_rows = [json.loads(l) for l in open(pred_path)] if pred_path else []
    ref_file = clip_dir / "labels" / "reference.yaml"
    ref = yaml.safe_load(open(ref_file))["timeline"] if ref_file.exists() else None
    zfile = DATA / "analysis" / "walking-bed-2026-09-16" / "zones" / f"{clip}.yaml"
    bed = unletterbox(yaml.safe_load(open(zfile))["bed"], src_w, src_h) if zfile.exists() else None
    lanes = ([("TRUTH", segments_ref(ref))] if ref else []) + ([("SEEN", segments_pred(pred_rows))] if pred_rows else [])
    print("pred:", pred_path, "| truth:", bool(ref), "| bed zone:", bool(bed))

    def pred_at(t):
        cur = pred_rows[0]
        for r in pred_rows:
            if r["t_s"] <= t:
                cur = r
            else:
                break
        return cur

    dim = (0.62 * vignette())[..., None]
    sc = scrim(260)[..., None]
    base_mul = dim * (1 - sc)

    cap = cv2.VideoCapture(str(DATA / "raw" / meta["source"]))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(src_fps / FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * src_fps))

    out_path = WORK / clip / f"{name}-30fps.mp4"
    n = len(recs)
    fade = int(0.4 * FPS)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-",
         "-vf", f"fade=in:0:{fade},fade=out:{n - fade}:{fade}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )
    hist = {j: [] for j in TRAIL_JOINTS}
    shown_state, prev_state, fade_t = None, None, 1.0
    floor_since = None
    for i, rec in enumerate(recs):
        ok, img = cap.read()
        for _ in range(step - 1):
            cap.grab()
        if not ok:
            break
        t = rec["t"]
        if img.shape[1] != BW or img.shape[0] != BH:
            fx0, fy0, fw_, fh_ = (round(v * S) for v in FIT)
            canvas = np.zeros((BH, BW, 3), np.uint8)
            canvas[fy0:fy0 + fh_, fx0:fx0 + fw_] = cv2.resize(img, (fw_, fh_), interpolation=cv2.INTER_CUBIC)
            img = canvas
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        base = Image.fromarray((rgb * base_mul).clip(0, 255).astype(np.uint8))

        pr = pred_at(t) if pred_rows else None
        st = pr["state"] if pr else None
        if st != shown_state:
            prev_state, shown_state, fade_t = shown_state, st, 0.0
        fade_t = min(1.0, fade_t + 1 / (0.3 * FPS))
        col = STATE[st][0]
        floor_since = (floor_since if floor_since is not None else t) if st == "on_floor" else None

        if bed:
            bed_zone(base, bed, bool(pr) and pr["zone"] == "bed", col)

        pts, presence = sm[i]
        for j in TRAIL_JOINTS:
            hist[j].append(pts[j][:2] if pts is not None and pts[j][2] > 0.6 else None)
            hist[j] = hist[j][-int(TRAIL_S * FPS):]
        skeleton(base, pts, presence, hist)

        # top bar
        text(base, (72, 78), label, F_TITLE, IVORY)
        m, s = divmod(t, 60)
        text(base, (72 + text_width(label, F_TITLE) + 22, 78), f"{int(m):02d}:{s:05.2f}", F_MONO_L, IVORY, 0.8)
        text(base, (OUT_W - 72, 76), "30 FPS · YOLO11X-POSE ON 4K CROP", F_MONO_S, IVORY, 0.8, spacing=1.2, anchor="rs")

        # floor chip: duration only, no claim about what the agent does
        if floor_since is not None:
            dur = t - floor_since
            cx, cy, cw = OUT_W - 72 - 330, 110, 330
            d = ImageDraw.Draw(base, "RGBA")
            d.rounded_rectangle((*P(cx, cy), *P(cx + cw, cy + 64)), radius=14 * S, fill=(20, 22, 25, 235), outline=col + (255,), width=int(1.5 * S))
            text(base, (cx + 22, cy + 26), "ON THE FLOOR", F_MONO_S, col, spacing=1.8)
            text(base, (cx + 22, cy + 52), f"for {int(dur)} s", F_CHIP, IVORY)

        # lower third
        if pr:
            text(base, (72, 950), "WHAT THE SYSTEM SEES", F_MONO_S, MUTED, spacing=2.0)
            conf = pr["state_confidence"] or 0.0
            zone = {"bed": "bed", "other": "room", None: "—"}.get(pr["zone"], pr["zone"])
            for s_, a_ in ((prev_state, 1 - fade_t), (shown_state, fade_t)):
                if a_ <= 0.01:
                    continue
                c_, word = STATE[s_]
                d = ImageDraw.Draw(base, "RGBA")
                d.ellipse((*P(72, 983), *P(86, 997)), fill=c_ + (int(255 * a_),))
                text(base, (104, 1004), word, F_STATE, IVORY, a_)
            text(base, (72, 1044), f"confidence {conf:.2f}   zone {zone}", F_MONO_M, IVORY, 0.85)
        if lanes:
            timeline(base, 820, 986 + (26 if len(lanes) == 1 else 0), OUT_W - 72 - 820, t, lanes, (start, end))

        frame = base.resize((OUT_W, OUT_H), Image.LANCZOS)
        ff.stdin.write(frame.tobytes())
        if i % 60 == 0:
            print(name, i, n, flush=True)
    ff.stdin.close()
    ff.wait()
    cap.release()
    print("wrote", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--pred", help="prediction JSONL for the state HUD (default: best available)")
    ap.add_argument("--label", help="title in the top bar (default: from the clip id)")
    a = ap.parse_args()
    for nm in a.names:
        render(a.clip, nm, a.pred, a.label)
