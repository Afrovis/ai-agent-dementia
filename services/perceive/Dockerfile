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

# `PERCEIVE_POSE_BACKEND` defaults to `mediapipe` (HANDOFF.md section 12:
# "MediaPipe as the standing default"), so the `mediapipe` extra is
# installed here, unlike `capture`'s optional `camera` extra which the
# default source does not need. Switching to PERCEIVE_POSE_BACKEND=yolo
# needs the `yolo` extra installed instead; PERCEIVE_POSE_BACKEND=scripted
# needs neither.
RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e "/app/services/perceive[mediapipe]"

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
