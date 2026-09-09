FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/listen/ /app/services/listen/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/listen

WORKDIR /app/services/listen

CMD ["python", "-m", "listen"]
