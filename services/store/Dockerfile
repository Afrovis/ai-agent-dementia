FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/store/ /app/services/store/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/store

WORKDIR /app/services/store

CMD ["python", "-m", "store"]
