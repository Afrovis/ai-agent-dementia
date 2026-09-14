FROM python:3.12-slim

ARG INSTALL_VIDEO_EVAL=false

WORKDIR /app

# MediaPipe's and Ultralytics' OpenCV dependency links against these at import time. Without
# them the Python package installs successfully in slim but the backend still
# crashes before its first frame with a missing `libGL.so.1`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY shared/ /app/shared/
COPY services/perceive/ /app/services/perceive/

# `PERCEIVE_POSE_BACKEND` defaults to `yolo` with `yolo11s-pose.pt`, chosen
# on the 2026-09-13 bedroom clips (HANDOFF.md section 12,
# docs/PERCEIVE_ACCURACY_2026-09-13.md). The `mediapipe` extra stays
# installed so falling back is an env change rather than a rebuild;
# PERCEIVE_POSE_BACKEND=scripted needs neither.
#
# Torch comes from the CPU-only index first: the PyPI Linux wheels pull about
# 3.3 GB of NVIDIA CUDA libraries that a Mac mini container can never use.
# Ultralytics then finds torch already satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e "/app/services/perceive[yolo,mediapipe]"

# Fetch the weights at build time, next to where `perceive` runs, so a
# container never needs the network to start and never downloads at night.
RUN python -c "from ultralytics.utils.downloads import attempt_download_asset as a; \
a('/app/services/perceive/yolo11s-pose.pt')"

# The deployment image stays focused on `perceive`; evaluation installs the
# offline CLI and capture's real MotionGate only when explicitly requested.
COPY services/capture/ /app/services/capture/
COPY tools/video_eval/ /app/tools/video_eval/
RUN if [ "$INSTALL_VIDEO_EVAL" = "true" ]; then \
        pip install --no-cache-dir -e /app/services/capture \
          -e "/app/services/perceive[yolo]" -e /app/tools/video_eval; \
    fi

WORKDIR /app/services/perceive

CMD ["python", "-m", "perceive"]
