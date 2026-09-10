FROM python:3.12-slim

WORKDIR /app

COPY shared/ /app/shared/
COPY services/dashboard/ /app/services/dashboard/

RUN pip install --no-cache-dir -e /app/shared \
    && pip install --no-cache-dir -e /app/services/dashboard

WORKDIR /app/services/dashboard

# Matches DASHBOARD_PORT's default in .env.example; docker-compose.yml maps
# the actual configured port, this just documents it for `docker inspect`
# and anyone running the image directly.
EXPOSE 8444

CMD ["python", "-m", "dashboard"]
