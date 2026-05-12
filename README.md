# mlb-ev

Dockerized Expected Value (EV) pipeline for MLB moneylines. Compute runs locally;
AWS is used sparingly (S3 for snapshots/artifacts, optional Lambda inference).

## Layout

```
src/
  ingest/      pull raw data (odds, MLB-StatsAPI, pybaseball)
  features/    build daily Parquet feature snapshots
  train/       fit + persist models
  backtest/    walk-forward EV backtests
  serve/       inference (local FastAPI / optional Lambda)
data/
  raw/         immutable provider responses (JSON)
  processed/   feature tables (Parquet)
infra/        Dockerfiles, deployment configs
models/       saved model artifacts
```

## Odds ingestion (The Odds API)

The Odds API free tier allows ~500 requests/month. Each call returns every
upcoming MLB game in one shot, so a sustainable cadence is ~3–8 snapshots/day
(~90–240/month) which leaves headroom for re-runs.

### 1. Configure

```bash
cp .env.example .env
# edit .env and set ODDS_API_KEY=...
```

### 2. Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.ingest.fetch_odds
```

### 3. Run in Docker

```bash
docker compose build ingest
docker compose run --rm ingest
```

Snapshots land in:

```
data/raw/odds/baseball_mlb/h2h/YYYY-MM-DD/YYYY-MM-DDTHH-MM-SSZ.json
```

Each file wraps the raw provider payload with capture metadata:

```json
{
  "fetched_at_utc": "...",
  "sport": "baseball_mlb",
  "regions": "us",
  "markets": "h2h",
  "odds_format": "american",
  "requests_used": "...",
  "requests_remaining": "...",
  "last_request_cost": "...",
  "game_count": 15,
  "data": [ /* unmodified Odds API response */ ]
}
```

The log line at the end of every run reports `remaining` so you can watch the
monthly budget without re-hitting the API.
