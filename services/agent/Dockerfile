FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/agent/ /app/services/agent/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/agent

WORKDIR /app/services/agent

CMD ["python", "-m", "agent"]
