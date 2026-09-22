FROM python:3.12-slim

ARG INSTALL_VIDEO_EVAL=false

WORKDIR /app

# MediaPipe's OpenCV dependency links against these at import time. Without
# them the Python package installs successfully in slim but the backend still
# crashes before its first frame with a missing `libGL.so.1`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /app/shared/
COPY services/perceive/ /app/services/perceive/

# `PERCEIVE_POSE_BACKEND` defaults to `yolo` on the bigger 640x480 bridge
# frame: YOLO11s-pose found far more floor falls there than MediaPipe, the
# nano model, or YOLOv8s-pose (which locks onto a static object on the bed
# instead of the fallen person on video 3) did (docs/FLOOR_DETECTION_HANDOFF.md),
# so the `yolo` extra is installed here, unlike `capture`'s optional `camera`
# extra which the default source does not need. `ultralytics` pulls in
# `torch` and `torchvision`; install both CPU-only wheels first (a PyPI
# `torchvision` does not match CPU `torch`: `torchvision::nms does not exist`) so pip does not also pull the
# much larger CUDA build. `mediapipe` is installed alongside so
# PERCEIVE_POSE_BACKEND=mediapipe still works without a rebuild;
# PERCEIVE_POSE_BACKEND=scripted needs neither.
RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e "/app/services/perceive[mediapipe,yolo]"

# Bake the default YOLO11s-pose weights into the image at build time so the
# service never downloads them at runtime: a fresh or recreated container on
# a quiet night must not depend on internet access working (HANDOFF.md rule
# 4: "a crash must never look like a quiet night"). Ultralytics downloads a
# missing-by-name weights file to the current directory, so this runs from
# the same directory `perceive` itself runs from (the final `WORKDIR`
# below), and `PERCEIVE_YOLO_MODEL`'s default (`yolo11s-pose.pt`) resolves
# to this file rather than triggering a download.
WORKDIR /app/services/perceive
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"
WORKDIR /app

# The deployment image stays focused on `perceive`; evaluation installs the
# offline CLI and capture's real MotionGate only when explicitly requested.
COPY services/capture/ /app/services/capture/
COPY tools/video_eval/ /app/tools/video_eval/
RUN if [ "$INSTALL_VIDEO_EVAL" = "true" ]; then \
        pip install --no-cache-dir -e /app/services/capture \
          -e /app/tools/video_eval; \
    fi

WORKDIR /app/services/perceive

CMD ["python", "-m", "perceive"]
