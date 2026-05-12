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

### 3. Run in Docker (one-shot)

```bash
docker compose build ingest
docker compose run --rm ingest
```

### 4. Run on a schedule (long-running container)

Fires the same ingest script at **9:00 AM, 12:30 PM, 6:00 PM, 9:00 PM
America/New_York** every day (DST handled automatically).

```bash
docker compose up -d --build scheduler   # start in the background
docker compose logs -f scheduler         # follow output, see each run
docker compose down                      # stop everything
```

The scheduler container has to be running at the scheduled times. On macOS that
means Docker Desktop must be running and the Mac must be awake; if the laptop
is asleep during a window, that snapshot is skipped (cron does not catch up).
Four windows/day × ~31 days ≈ **124 requests/month**, well under the 500 cap.

### 5. Run in AWS (laptop can sleep)

The same code can also run as an AWS Lambda triggered by EventBridge Scheduler,
writing snapshots to S3 instead of local disk. Costs are ~$0/month in the free
tier. Build the deployment ZIP with:

```bash
bash infra/build_lambda_zip.sh   # produces build/lambda.zip
```

See the project notes (or chat history) for the one-time AWS setup walkthrough.

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
