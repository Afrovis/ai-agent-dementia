FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/embodiment/ /app/services/embodiment/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/embodiment

WORKDIR /app/services/embodiment

CMD ["python", "-m", "embodiment"]
