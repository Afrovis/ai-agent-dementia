"""Two-stage 30 fps pose for a demo snippet: yolo11m finds the person, yolo11x-pose runs on a 4K crop.

    python infer.py <clip_id> <snippet_name> <start_s> <end_s>

Writes <WORK>/<clip_id>/<snippet_name>.pose.jsonl (one record per output frame) so the
renderer can be re-run for styling changes without touching the models.
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

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
MODELS = DATA / "models"


def biggest(boxes):
    # largest box, not highest confidence: the bedroom mirror produces a confident reflection
    b = boxes.xyxy.cpu().numpy()
    return ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax()


def source_of(clip):
    meta = yaml.safe_load(open(DATA / "clips" / clip / "clip.yaml"))
    return DATA / "raw" / meta["source"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("name")
    ap.add_argument("start", type=float)
    ap.add_argument("end", type=float)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    det = YOLO(str(MODELS / "yolo11m-pose.pt"))
    pose = YOLO(str(MODELS / "yolo11x-pose.pt"))
    out_dir = WORK / a.clip
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source_of(a.clip)))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(src_fps / a.fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(a.start * src_fps))
    n_out = round((a.end - a.start) * a.fps)
    t0 = time.time()
    with open(out_dir / f"{a.name}.pose.jsonl", "w") as fh:
        for i in range(n_out):
            ok, img = cap.read()
            for _ in range(step - 1):
                cap.grab()
            if not ok:
                break
            H, W = img.shape[:2]
            rec = {"i": i, "t": a.start + i / a.fps, "kp": None}
            r = det(img, imgsz=1280, verbose=False, device=a.device)[0]
            if len(r.boxes):
                k = biggest(r.boxes)
                x0, y0, x1, y1 = r.boxes.xyxy[k].cpu().numpy()
                pw, ph = (x1 - x0) * 0.25, (y1 - y0) * 0.25
                x0, y0 = int(max(0, x0 - pw)), int(max(0, y0 - ph))
                x1, y1 = int(min(W, x1 + pw)), int(min(H, y1 + ph))
                r2 = pose(img[y0:y1, x0:x1], imgsz=960, verbose=False, device=a.device)[0]
                if len(r2.boxes):
                    j = biggest(r2.boxes)
                    kp = r2.keypoints.data[j].cpu().numpy()
                    rec["kp"] = [[round(float((p[0] + x0) / W), 5), round(float((p[1] + y0) / H), 5), round(float(p[2]), 3)] for p in kp]
                    rec["conf"] = round(float(r2.boxes.conf[j]), 3)
            fh.write(json.dumps(rec) + "\n")
            if i % 60 == 0:
                print(a.name, i, n_out, f"{(time.time() - t0) / (i + 1):.2f}s/frame", flush=True)
    cap.release()
    print("wrote", out_dir / f"{a.name}.pose.jsonl")


if __name__ == "__main__":
    main()
