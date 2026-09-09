FROM python:3.12-slim

WORKDIR /app

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

WORKDIR /app/services/perceive

CMD ["python", "-m", "perceive"]
