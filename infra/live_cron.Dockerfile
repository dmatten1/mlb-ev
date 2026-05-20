FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/New_York

RUN apt-get update \
    && apt-get install -y --no-install-recommends cron tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo "$TZ" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-refresh.txt ./
RUN pip install --no-cache-dir -r requirements-refresh.txt

COPY src ./src

COPY infra/live_loop.crontab /etc/cron.d/mlb-ev-live
RUN chmod 0644 /etc/cron.d/mlb-ev-live

# Long-running: cron fires `live_refresh` on the schedule above. For a one-shot:
#   docker compose run --rm live python -m src.pipeline.live_refresh
CMD ["cron", "-f", "-L", "15"]
