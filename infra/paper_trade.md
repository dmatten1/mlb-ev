# Autonomous paper trading (honest simulation)

## Behavior

`src.tracking.bet_log.log_recommendations` writes **one paper ticket per game**:

- The **first** time the slate contains `recommended ∈ {home, away}` for a `game_id` **and both lineups are actual** (posted to MLB), we insert a row.
- **Projected-lineup signals** (+EV on modal lineups) appear in `data/predictions/<date>.md` as *awaiting lineups* but are **not** logged until a later refresh sees actual lineup cards (~3h pre-game).
- After that pick exists, we **never** overwrite odds, fair price, EV, Kelly, model probability, book, or side — even if a later snapshot would show better prices or the model flips sides.
- **Postponed / stale:** `reconcile_voids` marks pending rows `void` (0 P/L, hidden from dashboard) when MLB status is Postponed/Cancelled, or when first pitch was **24h+** ago with no final score.
- **Doubleheaders:** each `game_id` is its own ticket; odds attach by **first-pitch time** (`game_datetime` ↔ odds `commence_time`), not just team + date.
- CLV (`reconcile_clv`) still compares your **frozen** entry fair probability to the **closing** line from snapshots — that matches reality (you beat or lost closing line from where you actually bet).

Intermediate odds history stays in `data/raw/odds/...` JSON snapshots.

## Where it runs while your laptop sleeps

| Option | Notes |
|--------|--------|
| **Docker scheduler** (`make docker-scheduler-up`) | **`infra/docker_scheduler.md`** — cron defaults to **America/Chicago** (8 / 11:30 / 17 / 20 local = same *instants* as 9 / 12:30 / 18 / 21 ET). **Always-on** host. Mount `./data` + `.env` (`ODDS_API_KEY`, optional AWS for S3 odds). |
| **AWS Lambda + EventBridge** | Same pattern as odds Lambda; chain or separately invoke refresh — keep snapshots on S3 and point loaders at `local_root=None`. |
| **macOS launchd** | `make odds-schedule-on` — laptop must be awake (same limitation as before). |

Run **`make refresh`** (full pipeline) at least daily during the season so training parquets and the Ridge cache stay fresh; the scheduler only runs the **light** chain after each odds pull.

If you also run the **`live`** compose service (`docker-live-up`), `live_refresh` fires about every two hours **in addition** to the scheduler — harmless for the bet log (paper locks prevent rewrites), but redundant CPU/API usage unless you want extra outcome/dashboard refreshes between odds snapshots.

## Optional cadence tweaks

If four snapshots/day leaves too wide a gap before first pitch for your goals, edit `infra/scheduler.crontab` (Docker) or `infra/launchd.odds.plist` (Mac) and redeploy — Odds API quota permitting.
