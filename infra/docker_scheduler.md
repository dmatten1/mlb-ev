# Docker scheduler (paper trading)

Runs **`fetch_odds --live-after`** on a fixed daily cadence inside a long-lived container. Odds are written under `data/raw/odds/`, then **`live_refresh`** runs (outcomes, schedule, predict, bet log, CLV/outcomes, dashboard).

## Prerequisites

- **Docker** Desktop or Engine on an **always-on** machine (NAS, mini PC, VPS, home server). Your laptop can sleep.
- Repo root is the working directory; from there:
  - **`./data`** has your features parquets, model cache (`data/models/runs_model_bullpen_cached.pkl`), tracking output, etc.
  - **`.env`** contains **`ODDS_API_KEY`** and (if you load odds from S3) AWS credentials.

## Start

```bash
cd /path/to/mlb-ev
docker compose up -d --build scheduler
# or: make docker-scheduler-up
```

## Logs

```bash
docker compose logs -f scheduler
# or: make docker-scheduler-logs
```

## Timezone (Central by default)

`infra/scheduler.crontab` uses **`TZ=America/Chicago`** and fires at **8:00, 11:30, 17:00, 20:00** Central — the same **clock instants** as **9:00, 12:30, 18:00, 21:00 Eastern** (MLB’s usual ET windows). No code change needed for Eastern audiences: edit that file to `TZ=America/New_York` and hours **9, 12, 30, 18, 0, 21, 0**, then rebuild:

```bash
docker compose up -d --build scheduler
```

## Stop

```bash
docker compose stop scheduler
# or: make docker-scheduler-down
```

## Optional

- **`live`** service (`docker compose up -d live`): extra **`live_refresh` ~every 2h**; redundant with scheduler for bet logging (paper lock prevents double-booking) but uses more CPU.
- Run **`make refresh`** (full pipeline) at least periodically so training data and the Ridge cache stay current.

Do **not** run **Docker scheduler** and **macOS `make odds-schedule-on`** against the **same** `data/` volume at the **same** times — you would duplicate Odds API calls. Pick one host for automation.
