FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/dashboard/ /app/services/dashboard/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/dashboard

WORKDIR /app/services/dashboard

CMD ["python", "-m", "dashboard"]
