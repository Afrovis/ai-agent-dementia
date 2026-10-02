"""Convert an MP4 to an animated WebP by streaming frames from ffmpeg into Pillow.

The local ffmpeg has no WebP encoder; Pillow's has animation support.
Usage: to_webp.py in.mp4 out.webp WIDTH FPS QUALITY   (README copies: 1280 12 70-75)
"""
import subprocess, sys
from PIL import Image
src, out, w, fps, q = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
h = w * 9 // 16
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-vf", f"fps={fps},scale={w}:{h}:flags=lanczos",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
def frames():
    while (buf := p.stdout.read(w * h * 3)) and len(buf) == w * h * 3:
        yield Image.frombytes("RGB", (w, h), buf)
it = frames(); first = next(it)
first.save(out, save_all=True, append_images=it, duration=round(1000 / fps), loop=0,
           quality=q, method=4, minimize_size=False, allow_mixed=True)
