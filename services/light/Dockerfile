FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/light/ /app/services/light/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/light

WORKDIR /app/services/light

CMD ["python", "-m", "light"]
