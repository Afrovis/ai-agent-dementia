FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/capture/ /app/services/capture/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/capture

WORKDIR /app/services/capture

CMD ["python", "-m", "capture"]
