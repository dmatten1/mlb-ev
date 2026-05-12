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

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

COPY infra/scheduler.crontab /etc/cron.d/odds-ingest
RUN chmod 0644 /etc/cron.d/odds-ingest

CMD ["cron", "-f", "-L", "15"]
