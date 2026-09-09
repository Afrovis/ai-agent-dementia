FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/perceive/ /app/services/perceive/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/perceive

WORKDIR /app/services/perceive

CMD ["python", "-m", "perceive"]
