FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/notify/ /app/services/notify/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/notify

WORKDIR /app/services/notify

CMD ["python", "-m", "notify"]
