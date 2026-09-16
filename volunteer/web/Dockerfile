FROM python:3.12-slim

WORKDIR /app

COPY volunteer/common/ /app/volunteer/common/
COPY volunteer/web/ /app/volunteer/web/

RUN pip install --no-cache-dir -e /app/volunteer/common \
    && pip install --no-cache-dir -e /app/volunteer/web

WORKDIR /app/volunteer/web

EXPOSE 8000

CMD ["python", "-m", "volunteer_web"]
