FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/New_York

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo "$TZ" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the heavier dependency set (pandas, sklearn, pybaseball, statsapi
# etc.). Kept in a separate requirements file so the lean odds-Lambda
# image stays small.
COPY requirements.txt requirements-refresh.txt ./
RUN pip install --no-cache-dir -r requirements-refresh.txt

COPY src ./src

# data/ is volume-mounted by docker-compose so the host filesystem owns
# the parquets. The image itself ships with no data.

# Default: one-shot run of the full daily refresh. Schedule by invoking
# this container daily (`docker compose run --rm refresh`) — keeping
# scheduling out of the image makes it trivial to delegate to either
# cron/launchd on the host or ECS/Fargate EventBridge in the cloud.
CMD ["python", "-m", "src.pipeline.daily_refresh"]
