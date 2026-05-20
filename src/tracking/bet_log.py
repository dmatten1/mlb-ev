"""Persistent bet-tracking log: every recommendation written, with
CLV against the closing line and the eventual outcome.

The bet log is a single Parquet file (``data/tracking/bet_log.parquet``)
keyed by ``game_id``. The log is updated in three phases as a game
progresses through its lifecycle:

1. **Recommendation** (every ``daily_refresh`` / ``live_refresh`` run)
   - One row per game where ``recommended ∈ {'home', 'away'}``.
   - Captures the price at the moment we **first** logged that pick: model
     probability, de-vigged fair probability, edge, EV, Kelly, book.
   - **Paper-trading rule:** once a row exists with a committed side
     (``recommended_side ∈ {'home','away'}``), we **never** overwrite odds,
     fair price, EV, Kelly, model p, book, or recommendation fields — same
     as placing a real ticket when the signal fires. Later snapshots cannot
     “trade up” to better prices or flip sides.
   - Re-running *after* commence_time does not update recommendation fields
     either (game underway / finished — row stays audit-faithful).

2. **Closing-line value** (after game starts)
   - We re-pull the same game's price from the closest-to-first-pitch
     snapshot (without a 30-minute pre-game cutoff) and record it as
     ``closing_odds``. CLV = (closing fair_p − recommended fair_p) on
     the side we recommended — positive means the market moved
     toward our pick, the gold-standard "beat the close" signal.

3. **Outcome reconciliation** (after final score posts)
   - Joins final scores and writes ``outcome`` plus ``profit_units`` using
     **Kelly-scaled ``risk_units``** (:func:`src.model.betting.kelly_to_risk_units`
     vs. **pre–daily-cap** Kelly, using the slate's mean recommended Kelly
     as one unit when logged — see ``risk_ref_kelly``). Losses are **−risk_units**; wins pay
     **risk_units × (decimal − 1)**. The **Kelly (bankroll %) columns**
     after the slate’s daily stake cap are unchanged — those are for
     real-dollar sizing only.

Why one row per game (and not per snapshot)?
  - The bet log tracks the **paper ticket**: first pre-game signal we
    acted on. Full odds history stays in raw snapshot JSON / Parquet;
    intermediate model runs before that signal are intentionally not
    logged as separate bets.

Idempotency contract:
  - ``log_recommendations`` can be called repeatedly with the same or
    newer slate snapshots. Rows with an existing committed pick are never
    updated for recommendation fields; rows past ``commence_time`` are also
    skipped.
  - ``reconcile_clv`` and ``reconcile_outcomes`` are pure functions of
    the slate + outcomes data and can be re-run safely.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.betting import (
    DEFAULT_KELLY_CAP,
    DEFAULT_KELLY_FRACTION_MULT,
    kelly_to_risk_units,
    tracking_kelly_ref_fallback,
)

logger = logging.getLogger("tracking.bet_log")

DEFAULT_LOG_PATH = Path("data/tracking/bet_log.parquet")

# Columns the log carries. Documented in the module docstring.
LOG_COLUMNS: tuple[str, ...] = (
    "game_id", "game_date", "commence_time",
    "home_id", "away_id", "home_name", "away_name",
    "recommended_side",          # 'home' | 'away'
    "recommended_team",           # team name we picked
    "book",                       # which book had best price at rec time
    "odds_at_rec",                # American odds at recommendation time
    "decimal_at_rec",             # decimal odds at recommendation time
    "model_p",                    # model's prob on our chosen side
    "fair_p_at_rec",              # de-vigged market prob on our side at rec
    "edge_at_rec",                # model_p − fair_p
    "ev_at_rec",                  # expected profit per $1 staked
    "kelly_at_rec",               # recommended Kelly fraction (bankroll %) after daily cap
    "kelly_pre_daily",            # Kelly before slate-wide daily rescale (relative conviction)
    "risk_ref_kelly",             # slate mean rec-Kelly (=1u denominator) at log time
    "risk_units",                 # units at risk from :func:`kelly_to_risk_units`
    "rec_snapshot_ts",            # snapshot we sampled odds from
    "rec_logged_at",              # wall-clock when we wrote this row
    "lineup_source_home",         # 'projected' | 'actual'
    "lineup_source_away",
    "closing_odds",               # American odds closest to first pitch
    "closing_fair_p",             # de-vigged closing prob on our side
    "clv_pp",                     # closing_fair_p − fair_p_at_rec (in pct points)
    "closing_snapshot_ts",        # when the closing snapshot was captured
    "home_score", "away_score",
    "outcome",                    # 'won' | 'lost' | 'push' | 'pending'
    "profit_units",               # risk_units×(decimal−1) won; −risk_units lost; 0 push
)


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def load_log(path: Path | str = DEFAULT_LOG_PATH) -> pd.DataFrame:
    """Read the bet log. Returns empty DataFrame with the right schema
    if the file doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})
    df = pd.read_parquet(path)
    if "risk_units" not in df.columns:
        df["risk_units"] = 1.0
    else:
        df["risk_units"] = pd.to_numeric(df["risk_units"], errors="coerce").fillna(1.0)
    if "kelly_pre_daily" not in df.columns:
        df["kelly_pre_daily"] = pd.NA
    if "risk_ref_kelly" not in df.columns:
        df["risk_ref_kelly"] = pd.NA
    return df


def save_log(df: pd.DataFrame, path: Path | str = DEFAULT_LOG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reorder + ensure all expected columns exist.
    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    if "risk_units" in df.columns:
        df["risk_units"] = pd.to_numeric(df["risk_units"], errors="coerce").fillna(1.0)
    df[list(LOG_COLUMNS)].to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# Phase 1: log recommendations from a slate
# ---------------------------------------------------------------------------

def _now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def _row_has_committed_pick(row: dict) -> bool:
    """True if this log row already represents a placed paper bet."""
    rs = row.get("recommended_side")
    if rs is None:
        return False
    try:
        if pd.isna(rs):
            return False
    except TypeError:
        pass
    return str(rs) in ("home", "away")


def log_recommendations(
    slate: pd.DataFrame,
    log_path: Path | str = DEFAULT_LOG_PATH,
    *,
    now: pd.Timestamp | None = None,
) -> dict:
    """Append or update bet-log rows from a fresh slate.

    For each row in ``slate`` with ``recommended ∈ {'home', 'away'}``:
      * If no row exists for this ``game_id`` in the log, **insert** one.
      * If a row exists but already has a committed pick (paper ticket),
        **skip** — simulates never rewriting a ticket when lines improve.
      * If a row exists, no committed pick yet, and ``now < commence_time``,
        **update** it with the incoming slate (edge case: legacy rows).
      * If a row exists and the game has started (``now >= commence_time``),
        **skip** (preserve settled audit trail).

    Returns counts ``skipped_post_commence``, ``skipped_paper_locked``, and
    ``skipped_locked`` (= sum of both) for backward-compatible logging.
    """
    if now is None:
        now = _now_utc()
    log = load_log(log_path)
    has_log = not log.empty

    rec = slate[slate["recommended"].isin(["home", "away"])].copy()
    if rec.empty:
        return {
            "inserted": 0,
            "updated": 0,
            "skipped_locked": 0,
            "skipped_post_commence": 0,
            "skipped_paper_locked": 0,
        }

    # Build the per-game record for every recommendation in slate.
    new_rows: list[dict] = []
    for _, r in rec.iterrows():
        side = str(r["recommended"])
        team = r["home_name"] if side == "home" else r["away_name"]
        book_col = f"{side}_book"
        odds_col = f"{side}_price_american"
        decimal_col = f"{side}_decimal"
        fair_col = f"{side}_fair_p"
        edge_col = f"edge_{side}"
        kelly_col = f"kelly_{side}"
        model_p = float(r["p_home"] if side == "home" else 1 - r["p_home"])
        rec_row = {
            "game_id": int(r["game_id"]),
            "game_date": r["game_date"],
            "commence_time": r["commence_time"],
            "home_id": int(r["home_id"]),
            "away_id": int(r["away_id"]),
            "home_name": r["home_name"],
            "away_name": r["away_name"],
            "recommended_side": side,
            "recommended_team": team,
            "book": r.get(book_col),
            "odds_at_rec": float(r[odds_col]),
            "decimal_at_rec": float(r[decimal_col]),
            "model_p": model_p,
            "fair_p_at_rec": float(r[fair_col]),
            "edge_at_rec": float(r[edge_col]),
            "ev_at_rec": float(r["recommended_ev"]),
            "kelly_at_rec": float(r["recommended_kelly"]),
            "kelly_pre_daily": float(r["recommended_kelly_pre_daily"])
            if pd.notna(r.get("recommended_kelly_pre_daily")) else pd.NA,
            "risk_ref_kelly": float(r["risk_ref_kelly"])
            if pd.notna(r.get("risk_ref_kelly")) else pd.NA,
            "risk_units": float(r["risk_units"]) if pd.notna(r.get("risk_units")) else 1.0,
            "rec_snapshot_ts": r.get("snapshot_ts"),
            "rec_logged_at": now,
            "lineup_source_home": r.get("home_lineup_source"),
            "lineup_source_away": r.get("away_lineup_source"),
            # Phase 2 / 3 fields left blank for later reconciliation
            "closing_odds": pd.NA,
            "closing_fair_p": pd.NA,
            "clv_pp": pd.NA,
            "closing_snapshot_ts": pd.NA,
            "home_score": pd.NA,
            "away_score": pd.NA,
            "outcome": "pending",
            "profit_units": pd.NA,
        }
        new_rows.append(rec_row)
    incoming = pd.DataFrame(new_rows)
    incoming["commence_time"] = pd.to_datetime(incoming["commence_time"], utc=True)

    if not has_log:
        save_log(incoming, log_path)
        return {
            "inserted": len(incoming),
            "updated": 0,
            "skipped_locked": 0,
            "skipped_post_commence": 0,
            "skipped_paper_locked": 0,
        }

    log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
    inserted = updated = skip_post = skip_paper = 0
    existing_ids = set(log["game_id"].astype(int))
    out_rows: list[pd.Series] = list(log.to_dict("records"))
    by_id = {int(r["game_id"]): i for i, r in enumerate(out_rows)}

    now_ts = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
    for _, r in incoming.iterrows():
        gid = int(r["game_id"])
        if gid not in by_id:
            out_rows.append(r.to_dict())
            inserted += 1
            continue
        idx = by_id[gid]
        existing_commence = pd.Timestamp(out_rows[idx]["commence_time"])
        if existing_commence.tzinfo is None:
            existing_commence = existing_commence.tz_localize("UTC")
        if now_ts >= existing_commence:
            skip_post += 1
            continue
        if _row_has_committed_pick(out_rows[idx]):
            skip_paper += 1
            continue
        # Update in place — preserve phase 2 / 3 fields if already set.
        preserved = {
            k: out_rows[idx].get(k)
            for k in ("closing_odds", "closing_fair_p", "clv_pp",
                       "closing_snapshot_ts", "home_score", "away_score",
                       "outcome", "profit_units")
        }
        new_row = r.to_dict()
        for k, v in preserved.items():
            if pd.notna(v):
                new_row[k] = v
        out_rows[idx] = new_row
        updated += 1

    save_log(pd.DataFrame(out_rows), log_path)
    skip_locked = skip_post + skip_paper
    return {
        "inserted": inserted,
        "updated": updated,
        "skipped_locked": skip_locked,
        "skipped_post_commence": skip_post,
        "skipped_paper_locked": skip_paper,
    }


# ---------------------------------------------------------------------------
# Phase 2: CLV reconciliation against the closing line
# ---------------------------------------------------------------------------

def reconcile_clv(
    log_path: Path | str = DEFAULT_LOG_PATH,
    *,
    date_lo: str | None = None,
    date_hi: str | None = None,
) -> dict:
    """Fill ``closing_odds`` / ``closing_fair_p`` / ``clv_pp`` columns
    for every bet whose game has started but doesn't yet have CLV.

    Pulls the latest snapshot ≤ commence_time (no 30-minute pre-game
    cutoff) from the odds_loader, computes the de-vigged fair price
    on our recommended side, and records CLV in percentage points
    (positive = the market moved toward our pick).
    """
    from src.inference.odds_loader import (
        load_snapshots_long, best_lines_per_game, build_team_name_to_id,
        attach_team_ids,
    )
    from src.model.betting import remove_vig

    log = load_log(log_path)
    if log.empty:
        return {"computed": 0, "skipped": 0}

    log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
    now = _now_utc()
    needs = log[log["clv_pp"].isna() & (log["commence_time"] <= now)].copy()
    if needs.empty:
        return {"computed": 0, "skipped": int(log["clv_pp"].isna().sum())}

    if date_lo is None:
        date_lo = str(needs["commence_time"].min().date())
    if date_hi is None:
        date_hi = str(needs["commence_time"].max().date() + pd.Timedelta(days=1))

    odds_long = load_snapshots_long(date_lo=date_lo, date_hi=date_hi)
    # For CLV we want the LATEST snapshot before commence_time across
    # any of the bettor's books — close_window_minutes=0 lifts the
    # pre-game cutoff so we grab the literal closing line.
    closing = best_lines_per_game(
        odds_long, close_window_minutes=0, price_strategy="best",
    )
    # Match by (commence_time, home_team, away_team).
    closing["commence_time"] = pd.to_datetime(closing["commence_time"], utc=True)
    log_idx = log.set_index("game_id")

    from src.model.betting import american_to_implied
    p_h_imp = american_to_implied(closing["home_price_american"].to_numpy())
    p_a_imp = american_to_implied(closing["away_price_american"].to_numpy())
    fh, fa = remove_vig(p_h_imp, p_a_imp, method="proportional")
    closing["closing_home_fair_p"] = fh
    closing["closing_away_fair_p"] = fa

    computed = 0
    for gid, row in needs.iterrows():
        match = closing[(closing["commence_time"] == row["commence_time"])
                        & (closing["home_team"] == row["home_name"])
                        & (closing["away_team"] == row["away_name"])]
        if match.empty:
            continue
        m = match.iloc[0]
        side = row["recommended_side"]
        closing_fair = float(m["closing_home_fair_p"] if side == "home"
                              else m["closing_away_fair_p"])
        closing_odds = float(m["home_price_american"] if side == "home"
                              else m["away_price_american"])
        log.loc[gid, "closing_odds"] = closing_odds
        log.loc[gid, "closing_fair_p"] = closing_fair
        log.loc[gid, "clv_pp"] = (closing_fair - float(row["fair_p_at_rec"])) * 100.0
        log.loc[gid, "closing_snapshot_ts"] = m["snapshot_ts"]
        computed += 1

    save_log(log.reset_index() if "game_id" not in log.columns else log, log_path)
    return {"computed": computed, "skipped": int(log["clv_pp"].isna().sum())}


# ---------------------------------------------------------------------------
# Phase 3: outcome reconciliation
# ---------------------------------------------------------------------------

def reconcile_outcomes(
    log_path: Path | str = DEFAULT_LOG_PATH,
    *,
    outcomes_root: Path | str = Path("data/outcomes"),
    features_root: Path | str = Path("data/features"),
) -> dict:
    """Fill ``home_score`` / ``away_score`` / ``outcome`` / ``profit_units``
    columns for every bet whose game has finalized.

    Looks up final scores in two places, in order: the dedicated
    ``data/outcomes/outcomes_<year>.parquet`` if present (canonical),
    else the ``home_score`` / ``away_score`` columns on
    ``data/features/training_<year>.parquet`` (which the daily feature
    build already populates as games complete).
    """
    log = load_log(log_path)
    if log.empty:
        return {"resolved": 0, "pending": 0}

    outcomes_root = Path(outcomes_root)
    features_root = Path(features_root)
    years_needed = pd.to_datetime(log["commence_time"], utc=True).dt.year.unique()
    outs = []
    for y in years_needed:
        y = int(y)
        cands = (outcomes_root / f"outcomes_{y}.parquet",
                  features_root / f"training_{y}.parquet")
        for p in cands:
            if p.exists():
                df = pd.read_parquet(p, columns=["game_id", "home_score", "away_score"])
                outs.append(df)
                break
    if not outs:
        return {"resolved": 0, "pending": int(log["outcome"].eq("pending").sum())}
    outcomes = pd.concat(outs, ignore_index=True)
    outcomes = outcomes.dropna(subset=["home_score", "away_score"])
    outcomes = outcomes.drop_duplicates("game_id", keep="last")
    outcomes_idx = outcomes.set_index("game_id")

    def _profit(outcome: str, dec: float, risk: float) -> float:
        if outcome == "push":
            return 0.0
        if outcome == "won":
            return risk * (dec - 1.0)
        if outcome == "lost":
            return -risk
        return float("nan")

    # Kelly-scale risk_units + backfill pre-daily Kelly *before* we settle
    # any newly finished games so P/L uses the right stake.
    if "kelly_pre_daily" not in log.columns:
        log["kelly_pre_daily"] = pd.NA
    if "risk_ref_kelly" not in log.columns:
        log["risk_ref_kelly"] = pd.NA
    miss_pre = log["kelly_pre_daily"].isna() & log["kelly_at_rec"].notna()
    log.loc[miss_pre, "kelly_pre_daily"] = pd.to_numeric(
        log.loc[miss_pre, "kelly_at_rec"], errors="coerce")
    pre_num = pd.to_numeric(log["kelly_pre_daily"], errors="coerce")
    mask_k = pre_num.notna()
    if mask_k.any():
        fb = tracking_kelly_ref_fallback(DEFAULT_KELLY_CAP, DEFAULT_KELLY_FRACTION_MULT)
        ref_num = pd.to_numeric(log["risk_ref_kelly"], errors="coerce")
        ref_eff = ref_num.where(ref_num > 0, np.nan).fillna(fb)
        ru_arr = kelly_to_risk_units(
            pre_num[mask_k].to_numpy(dtype=float),
            ref_kelly=ref_eff[mask_k].to_numpy(dtype=float),
            kelly_cap=DEFAULT_KELLY_CAP,
        )
        log.loc[mask_k, "risk_units"] = np.asarray(ru_arr, dtype=float).ravel()

    resolved = 0
    for i, r in log.iterrows():
        if r["outcome"] != "pending":
            continue
        gid = int(r["game_id"])
        if gid not in outcomes_idx.index:
            continue
        o = outcomes_idx.loc[gid]
        hs, as_ = float(o["home_score"]), float(o["away_score"])
        side = r["recommended_side"]
        ru = float(r.get("risk_units") or 1.0)
        d = float(r["decimal_at_rec"])
        if hs == as_:
            outcome = "push"
            profit = _profit("push", d, ru)
        else:
            home_won = hs > as_
            won = (side == "home" and home_won) or (side == "away" and not home_won)
            outcome = "won" if won else "lost"
            profit = _profit(outcome, d, ru)
        log.loc[i, "home_score"] = hs
        log.loc[i, "away_score"] = as_
        log.loc[i, "outcome"] = outcome
        log.loc[i, "profit_units"] = profit
        resolved += 1

    # Recompute settled P/L (idempotent; picks up any risk_units / odds fixes).
    for i, r in log.iterrows():
        if r["outcome"] not in ("won", "lost", "push"):
            continue
        ru = float(r.get("risk_units") or 1.0)
        profit = _profit(
            str(r["outcome"]), float(r["decimal_at_rec"]), ru,
        )
        log.loc[i, "profit_units"] = profit

    save_log(log, log_path)
    pending = int(log["outcome"].eq("pending").sum())
    return {"resolved": resolved, "pending": pending}


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def filter_log_by_season(
    log: pd.DataFrame,
    season_year: int | None,
) -> pd.DataFrame:
    """Keep rows whose UTC ``commence_time`` calendar year equals *season_year*.

    *season_year* ``None`` leaves *log* unchanged (all seasons).
    """
    if season_year is None or log.empty:
        return log
    log = log.copy()
    log["commence_time"] = pd.to_datetime(log["commence_time"], utc=True)
    return log.loc[log["commence_time"].dt.year == int(season_year)].reset_index(
        drop=True,
    )


def summarize_frame(log: pd.DataFrame) -> dict:
    """Aggregate dashboard stats from an already-loaded log slice."""
    if log.empty:
        return {"n_bets": 0}
    settled = log[log["outcome"].isin(["won", "lost", "push"])].copy()
    n_total = len(log)
    n_settled = len(settled)
    n_pending = n_total - n_settled
    profit = float(settled["profit_units"].fillna(0).sum())
    wins = int((settled["outcome"] == "won").sum())
    losses = int((settled["outcome"] == "lost").sum())
    hit_rate = wins / max(wins + losses, 1)
    risk_sum = float(settled["risk_units"].fillna(1.0).astype(float).sum())
    roi_per_unit = profit / max(risk_sum, 1e-9)
    avg_ev = float(log["ev_at_rec"].mean()) if n_total else 0.0
    clv_rows = log[log["clv_pp"].notna()]
    avg_clv = float(clv_rows["clv_pp"].mean()) if len(clv_rows) else float("nan")
    clv_beat_rate = float((clv_rows["clv_pp"] > 0).mean()) if len(clv_rows) else float("nan")
    return {
        "n_bets": n_total,
        "n_settled": n_settled,
        "n_pending": n_pending,
        "n_wins": wins,
        "n_losses": losses,
        "hit_rate": hit_rate,
        "profit_units": profit,
        "roi_per_unit": roi_per_unit,
        "avg_ev_at_rec": avg_ev,
        "avg_clv_pp": avg_clv,
        "clv_beat_rate": clv_beat_rate,
    }


def summarize(
    log_path: Path | str = DEFAULT_LOG_PATH,
    *,
    season_year: int | None = None,
) -> dict:
    """Aggregate stats for the dashboard, optionally limited to one UTC year."""
    log = filter_log_by_season(load_log(log_path), season_year)
    return summarize_frame(log)
