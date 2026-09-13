FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/embodiment/ /app/services/embodiment/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/embodiment \
    && python -m piper.download_voices --data-dir /app/models/piper en_US-lessac-medium

WORKDIR /app/services/embodiment

CMD ["python", "-m", "embodiment"]
