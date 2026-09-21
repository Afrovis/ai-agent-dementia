"""Run the real-time perceive pipeline over GMDCSA24 and score fall detection.

Raw videos stay where they are. Outputs (annotation copy, video links,
per-video predictions, report) go under ../data-ai-agent-dementia/datasets/gmdcsa24.

Run with the video-eval venv:
    .venv-video-eval/bin/python tools/gmdcsa24_eval/run_eval.py --backend yolo
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from capture.main import CaptureConfig, build_gate
from PIL import Image
from perceive.backends import build_backend
from perceive.classify import ground_zone_for_pose, zone_for_pose
from perceive.main import PerceiveConfig, build_tracker
from perceive.zones import ZoneMap

SRC = Path(
    "/Volumes/Mathias_SSD_2T/nighttime_ir_activity_data/extracted/gmdcsa24/"
    "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-5abac76"
)
OUT = Path("/Users/mathiasserver/Documents/data-ai-agent-dementia/datasets/gmdcsa24")
FPS = 2.0
CLASS_RE = re.compile(r"\s*([^\[;]+?)\s*\[\s*([\d.]+)\s*to\s*([\d.]+)\s*\]")


@dataclass
class Clip:
    subject: str
    kind: str  # ADL or Fall
    name: str
    length: float
    period: str
    classes: list[tuple[str, float, float]]

    @property
    def key(self) -> str:
        return f"{self.subject.replace(' ', '').lower()}_{self.kind.lower()}_{self.name[:-4]}"

    @property
    def video(self) -> Path:
        return SRC / self.subject / self.kind / self.name

    @property
    def fall_onset(self) -> float | None:
        starts = [s for c, s, _ in self.classes if c.lower().startswith("fall")]
        return min(starts) if starts else None


def load_clips() -> list[Clip]:
    clips = []
    for subject in sorted(p for p in SRC.iterdir() if p.is_dir()):
        for kind in ("ADL", "Fall"):
            with (subject / f"{kind}.csv").open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
                    name = row.get("File Name", "")
                    if not name:
                        continue
                    classes = [
                        (m[0], float(m[1]), float(m[2]))
                        for m in CLASS_RE.findall(row.get("Classes", ""))
                    ]
                    clips.append(
                        Clip(
                            subject.name,
                            kind,
                            name,
                            float(row.get("Length (seconds)") or 0),
                            row.get("Time of Recording", ""),
                            classes,
                        )
                    )
    return clips


def extract_frames(video: Path, tmp: Path) -> list[Path]:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-an", "-vf", f"fps={FPS:g}",
         "-q:v", "3", str(tmp / "f_%06d.jpg")],
        check=True,
    )
    return sorted(tmp.glob("f_*.jpg"))


def squash_jpeg(path: Path) -> bytes:
    image = Image.open(path).convert("RGB").resize((320, 240), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=60)
    return buf.getvalue()


def predict(clip: Clip, backend, config: PerceiveConfig) -> list[dict]:
    tracker = build_tracker(config)
    gate = build_gate(CaptureConfig.from_env())
    zones = ZoneMap()
    rows = []
    with tempfile.TemporaryDirectory() as tmp_name:
        frames = extract_frames(clip.video, Path(tmp_name))
        for index, path in enumerate(frames):
            t_s = index / FPS
            jpeg = squash_jpeg(path)
            gated = not gate.admit(jpeg, t_s)
            detected = None
            if not gated:
                pose = backend.detect(jpeg)
                detected = pose is not None
                zone = zone_for_pose(zones, pose) if pose is not None else "other"
                ground = ground_zone_for_pose(zones, pose) if pose is not None else None
                tracker.update(pose, zone, t_s, ground_zone=ground)
            snap = tracker.snapshot()
            rows.append({"t_s": t_s, "gated": gated, "detected": detected,
                         "state": snap[0] if snap else None})
    return rows


def first_time(rows: list[dict], state: str, after: float = 0.0) -> float | None:
    for row in rows:
        if row["state"] == state and row["t_s"] >= after:
            return row["t_s"]
    return None


def score(clips: list[Clip], preds: dict[str, list[dict]]) -> tuple[dict, str]:
    falls = [c for c in clips if c.kind == "Fall" and c.key in preds]
    adls = [c for c in clips if c.kind == "ADL" and c.key in preds]
    lines = []
    hit, lat = 0, []
    early = 0
    for c in falls:
        onset = c.fall_onset
        t = first_time(preds[c.key], "on_floor")
        if t is not None and onset is not None:
            if t < onset - 1.0:
                early += 1  # on_floor flagged well before the annotated fall
            else:
                hit += 1
                lat.append(max(0.0, t - onset))
    fa = [c for c in adls if first_time(preds[c.key], "on_floor") is not None]

    def rate(n, d):
        return round(n / d, 3) if d else None

    by_subject = {}
    for s in sorted({c.subject for c in clips}):
        f = [c for c in falls if c.subject == s]
        a = [c for c in adls if c.subject == s]
        fh = sum(1 for c in f if (t := first_time(preds[c.key], "on_floor")) is not None
                 and c.fall_onset is not None and t >= c.fall_onset - 1.0)
        fp = sum(1 for c in a if first_time(preds[c.key], "on_floor") is not None)
        by_subject[s] = {"falls": len(f), "fall_detected": fh, "adl": len(a), "adl_false_alarm": fp}
    lat.sort()
    summary = {
        "falls": len(falls),
        "fall_detected": hit,
        "fall_recall": rate(hit, len(falls)),
        "on_floor_before_annotated_onset": early,
        "latency_median_s": lat[len(lat) // 2] if lat else None,
        "latency_p90_s": lat[int(len(lat) * 0.9)] if lat else None,
        "adl": len(adls),
        "adl_false_alarms": len(fa),
        "adl_false_alarm_rate": rate(len(fa), len(adls)),
        "by_subject": by_subject,
    }
    lines += ["# GMDCSA24 vs real-time pipeline", "", "```json", json.dumps(summary, indent=2), "```", "",
              "## Missed falls", ""]
    for c in falls:
        t = first_time(preds[c.key], "on_floor")
        if t is None or (c.fall_onset is not None and t < c.fall_onset - 1.0):
            states = sorted({r["state"] for r in preds[c.key] if r["state"]})
            lines.append(f"- {c.key} ({c.period}) onset={c.fall_onset} first_on_floor={t} states={states}")
    lines += ["", "## ADL false alarms (on_floor reported)", ""]
    for c in fa:
        lines.append(f"- {c.key} ({c.period}) classes={c.classes} first_on_floor="
                     f"{first_time(preds[c.key], 'on_floor')}")
    return summary, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="yolo")
    parser.add_argument("--yolo-model", default="yolo11s-pose.pt")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    clips = load_clips()
    if args.limit:
        clips = clips[: args.limit]
    tag = args.backend
    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    (OUT / "predictions" / tag).mkdir(parents=True, exist_ok=True)
    for subject in SRC.iterdir():
        for f in subject.glob("*.csv") if subject.is_dir() else []:
            dest = OUT / "annotations" / f"{subject.name.replace(' ', '').lower()}_{f.name}"
            shutil.copyfile(f, dest)
    shutil.copyfile(SRC / "README.md", OUT / "annotations" / "README.upstream.md")
    with (OUT / "clips.jsonl").open("w") as index:
        for c in clips:
            link = OUT / "videos" / f"{c.key}.mp4"
            if not link.is_symlink():
                link.symlink_to(c.video)
            index.write(json.dumps({"key": c.key, "subject": c.subject, "kind": c.kind,
                                    "length_s": c.length, "period": c.period,
                                    "classes": c.classes, "video": str(link)}) + "\n")

    backend = build_backend(args.backend, model_path=args.yolo_model, static_image_mode=True)
    config = PerceiveConfig.from_env()
    preds = {}
    for i, c in enumerate(clips, 1):
        cache = OUT / "predictions" / tag / f"{c.key}.json"
        if cache.exists():
            preds[c.key] = json.loads(cache.read_text())
            continue
        preds[c.key] = predict(c, backend, config)
        cache.write_text(json.dumps(preds[c.key]))
        print(f"[{i}/{len(clips)}] {c.key}", flush=True)
    summary, report = score(clips, preds)
    (OUT / "reports").mkdir(exist_ok=True)
    (OUT / "reports" / f"{tag}.md").write_text(report)
    (OUT / "reports" / f"{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
