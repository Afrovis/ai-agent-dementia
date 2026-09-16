# Build context: repository root (needs tools/video_eval and services/perceive's
# capture dependency, per HANDOFF.md section 4).
FROM python:3.12-slim

ARG GIT_SHA=unknown

WORKDIR /app

# Mirrors services/perceive/Dockerfile: mediapipe/opencv need these at import
# time, and video_eval remuxing needs ffmpeg.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY services/perceive/ /app/services/perceive/
COPY services/capture/ /app/services/capture/
COPY tools/video_eval/ /app/tools/video_eval/
COPY volunteer/common/ /app/volunteer/common/
COPY volunteer/worker/ /app/volunteer/worker/

# mediapipe==0.10.21, pinned in HANDOFF.md section 4, has no linux/arm64
# wheel on PyPI (only up to 0.10.18 there); services/perceive/Dockerfile
# does not pin it either, for the same reason, and takes whatever
# `mediapipe>=0.10,<1.0` resolves to on the build platform.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e "/app/services/perceive[mediapipe,yolo]" \
    && pip install --no-cache-dir -e /app/services/capture -e /app/tools/video_eval \
    && pip install --no-cache-dir -e /app/volunteer/common \
    && pip install --no-cache-dir -e /app/volunteer/worker

# Bake the default pose weights so a job never depends on internet access.
WORKDIR /app/volunteer/worker
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"

ENV GIT_SHA=${GIT_SHA}
ENV VOLUNTEER_NETWORK_MODE=none

CMD ["python", "-m", "volunteer_worker"]
