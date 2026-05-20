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

To **chain the light pipeline** after each successful snapshot (predictions, bet
log, dashboard), either pass **`--live-after`** or set in `.env`:
`MLB_EV_LIVE_AFTER_ODDS=1`. Use **`--no-live-after`** to force-disable for one
run.

The Docker **scheduler** image installs **`requirements-refresh.txt`** so cron can run
``fetch_odds --live-after`` end-to-end (same stack as ``live_refresh``). The slim **Lambda**
ZIP for odds-only ingestion stays on the minimal ``requirements.txt``.

### 3. Run in Docker (one-shot)

```bash
docker compose build ingest
docker compose run --rm ingest
```

## Paper trading semantics

Bet rows represent **frozen tickets**: the **first** logged recommendation per
`game_id` is kept forever (until outcomes settle); later runs cannot rewrite your
price or side when lines improve — see ``infra/paper_trade.md``.

### 4. Run on a schedule (long-running container, **recommended for 24/7**)

Uses **`infra/scheduler.crontab`**: default **`America/Chicago`** at **8:00 / 11:30 / 17:00 / 20:00** CT (same instants as **9 / 12:30 / 18 / 21** ET). See **`infra/docker_scheduler.md`**.

```bash
docker compose up -d --build scheduler   # or: make docker-scheduler-up
docker compose logs -f scheduler         # or: make docker-scheduler-logs
docker compose stop scheduler              # or: make docker-scheduler-down
```

The container must stay running on a host that does not sleep. **Docker Desktop on a sleeping laptop still skips runs** — use a small always-on box or a VPS.

Four windows/day × ~31 days ≈ **124 Odds API requests/month**, well under typical caps.

**Same schedule without Docker (Mac on, native Python):** from the repo, with
`.venv` and `.env` configured, run `make odds-schedule-on`. Align **System Settings → Date & Time** with the hours in `infra/launchd.odds.plist` (defaults assume **Chicago** system time; Eastern Mac: use **9 / 12:30 / 18 / 21** in the plist). Logs: `data/predictions/launchd.odds.{out,err}.log`. Remove with
`make odds-schedule-off`.

### 5. Full serverless pipeline (AWS, laptop off)

**EventBridge → odds Lambda (zip) → inference Lambda (ECR container) → S3 dashboard**

- Odds snapshots land in S3 (existing Lambda).
- Inference runs ``live_refresh`` (predict, paper bet log, CLV, HTML) and publishes ``index.html``.
- One-time upload of local parquets/model: ``bash infra/sync_artifacts_to_s3.sh``

Full walkthrough: **[infra/cloud_deploy.md](infra/cloud_deploy.md)**.

```bash
bash infra/build_lambda_zip.sh              # odds zip (redeploy after code changes)
bash infra/build_inference_lambda_image.sh  # ECR image for ML Lambda
```

## Live dashboard + bet tracker automation

**Full** morning run (Statcast, OAA, feature rebuild, train Ridge, predict,
track):

```bash
make refresh
# or: python -m src.pipeline.daily_refresh --year "$(date +%Y)"
```

**Light** loop — recent outcomes, schedule, tonight’s predict + `bet_log` rows,
CLV + outcome reconciliation + `data/tracking/bet_dashboard.html` — **without**
the heavy data steps. Reuses a cached Ridge model (written on the last full
train) so it is cheap enough for cron:

```bash
make live-refresh
# or: python -m src.pipeline.live_refresh
```

Schedule it:

- **macOS:** `make live-schedule-on` (every 2h while awake; logs under
  `data/predictions/launchd.live.*.log`). Keep `make schedule-on` for the once-
  daily full refresh (~07:00 ET).
- **Docker:** `make docker-live-up` — long-running `live` service with cron; see
  `infra/live_loop.crontab`. Ensure `./data` is mounted and `.env` has AWS
  creds if you read odds from S3.

Run **at least one full `make refresh` per day** during the season so training
parquets and the model cache stay fresh.

## Feature exploration (pybaseball)

A guided Jupyter notebook lives at `notebooks/01_pybaseball_explore.ipynb` —
one section per major `pybaseball` capability (FanGraphs season stats,
Statcast pitch/batter data, team-level, date-range, and a sandbox).

### Setup

```bash
source .venv/bin/activate   # the same venv used for ingestion
pip install -r requirements-explore.txt
jupyter lab notebooks/01_pybaseball_explore.ipynb
```

The first cell imports `src.features.pybaseball_utils` which exposes:

| Helper | Use |
|---|---|
| `enable_cache()` | turn on pybaseball's local disk cache (`~/.pybaseball`) so repeat queries are free |
| `save_raw(df, name)` | persist a DataFrame to `data/raw/pybaseball/<name>_<utc>.parquet` |
| `load_raw(name_or_prefix)` | reload the latest pull matching a prefix |
| `list_raw(prefix='')` | list saved pulls |
| `show(df)` | shape + head, for quick previews |

Saved pulls live under `data/raw/pybaseball/` and are gitignored.

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
