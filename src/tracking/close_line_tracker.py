"""Counterfactual bet log: decisions at the closing line only.

Uses the **same model probabilities** from saved prediction slates (projected
lineups at predict time) but re-applies :func:`src.model.betting.annotate_bets`
against the **literal closing moneyline** (latest snapshot before first pitch).

This does **not** modify ``bet_log.parquet`` or the paper-trading dashboard.
It is a parallel what-if track for comparing early-line vs close-line entry.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.model.betting import DEFAULT_MAX_EDGE, annotate_bets
from src.tracking.bet_log import LOG_COLUMNS, filter_log_by_season

logger = logging.getLogger("tracking.close_line_tracker")

DEFAULT_PREDICTIONS_ROOT = Path("data/predictions")
DEFAULT_OUTCOMES_ROOT = Path("data/outcomes")
DEFAULT_FEATURES_ROOT = Path("data/features")
DEFAULT_RAW_OUTCOMES_ROOT = Path("data/raw/outcomes/baseball_mlb")


def _load_prediction_slates(
    predictions_root: Path,
    *,
    season_year: int | None,
) -> pd.DataFrame:
    """Load and dedupe per-game rows from ``data/predictions/*.parquet``."""
    predictions_root = Path(predictions_root)
    if not predictions_root.is_dir():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(predictions_root.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip predictions %s: %s", path.name, e)
            continue
        if df.empty or "game_id" not in df.columns or "p_home" not in df.columns:
            continue
        df = df.copy()
        df["_source_file"] = path.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_slates = pd.concat(frames, ignore_index=True)
    all_slates["commence_time"] = pd.to_datetime(all_slates["commence_time"], utc=True)
    all_slates = filter_log_by_season(all_slates, season_year)
    if all_slates.empty:
        return all_slates

    # Latest prediction file wins when the same game appears on multiple days.
    all_slates = all_slates.sort_values(["game_id", "_source_file"])
    all_slates = all_slates.drop_duplicates("game_id", keep="last")
    return all_slates.drop(columns=["_source_file"], errors="ignore")


def _closing_odds_for_slates(
    slates: pd.DataFrame,
    *,
    features_root: Path,
) -> pd.DataFrame:
    from src.inference.odds_loader import (
        attach_team_ids,
        best_lines_per_game,
        build_team_name_to_id,
        load_snapshots_long,
    )

    if slates.empty:
        return pd.DataFrame()

    ct_min = slates["commence_time"].min()
    ct_max = slates["commence_time"].max()
    date_lo = (ct_min - pd.Timedelta(days=1)).date().isoformat()
    date_hi = (ct_max + pd.Timedelta(days=1)).date().isoformat()

    odds_long = load_snapshots_long(date_lo=date_lo, date_hi=date_hi)
    if odds_long.empty:
        logger.warning("close-line tracker: no odds snapshots for %s..%s", date_lo, date_hi)
        return pd.DataFrame()

    closing = best_lines_per_game(
        odds_long,
        close_window_minutes=0,
        price_strategy="best",
    )
    if closing.empty:
        return pd.DataFrame()

    year = int(pd.Timestamp(ct_max).year)
    team_feat = Path(features_root) / f"training_{year}.parquet"
    if not team_feat.exists():
        team_feat = Path(features_root) / "training_2025.parquet"
    team_map = build_team_name_to_id(team_feat) if team_feat.exists() else {}
    closing = attach_team_ids(closing, team_map)
    closing["commence_time"] = pd.to_datetime(closing["commence_time"], utc=True)
    return closing


def _slate_to_log_rows(annotated: pd.DataFrame) -> pd.DataFrame:
    """Map an annotated close-line slate into bet-log-shaped rows."""
    rec = annotated[annotated["recommended"].isin(["home", "away"])].copy()
    if rec.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})

    rows: list[dict] = []
    for _, r in rec.iterrows():
        side = str(r["recommended"])
        team = r["home_name"] if side == "home" else r["away_name"]
        book_col = f"{side}_book"
        odds_col = f"{side}_price_american"
        decimal_col = f"{side}_decimal"
        fair_col = f"{side}_fair_p"
        edge_col = f"edge_{side}"
        model_p = float(r["p_home"] if side == "home" else 1 - r["p_home"])
        rows.append({
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
            "rec_logged_at": pd.Timestamp.now(tz="UTC"),
            "lineup_source_home": r.get("home_lineup_source"),
            "lineup_source_away": r.get("away_lineup_source"),
            "closing_odds": float(r[odds_col]),
            "closing_fair_p": float(r[fair_col]),
            "clv_pp": pd.NA,
            "closing_snapshot_ts": r.get("snapshot_ts"),
            "home_score": pd.NA,
            "away_score": pd.NA,
            "outcome": "pending",
            "profit_units": pd.NA,
        })

    out = pd.DataFrame(rows)
    for c in LOG_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[list(LOG_COLUMNS)]


def _settle_outcomes(
    log: pd.DataFrame,
    *,
    outcomes_root: Path,
    features_root: Path,
    raw_outcomes_root: Path,
) -> pd.DataFrame:
    """Fill outcome / P/L from the same sources as :func:`reconcile_outcomes`."""
    from src.features.outcomes_loader import load_outcomes

    if log.empty:
        return log

    log = log.copy()
    years_needed = pd.to_datetime(log["commence_time"], utc=True).dt.year.unique()
    outs: list[pd.DataFrame] = []
    for y in years_needed:
        y = int(y)
        year_frames: list[pd.DataFrame] = []
        training = Path(features_root) / f"training_{y}.parquet"
        if training.exists():
            year_frames.append(
                pd.read_parquet(training, columns=["game_id", "home_score", "away_score"])
            )
        rollup = Path(outcomes_root) / f"outcomes_{y}.parquet"
        if rollup.exists():
            year_frames.append(
                pd.read_parquet(rollup, columns=["game_id", "home_score", "away_score"])
            )
        df = load_outcomes(year=y, root=raw_outcomes_root)
        if not df.empty:
            year_frames.append(df[["game_id", "home_score", "away_score"]])
        outs.extend(year_frames)

    if not outs:
        return log

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

    for i, r in log.iterrows():
        if str(r["outcome"]) != "pending":
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

    return log


def build_close_line_log(
    *,
    predictions_root: Path | str = DEFAULT_PREDICTIONS_ROOT,
    season_year: int | None = 2026,
    outcomes_root: Path | str = DEFAULT_OUTCOMES_ROOT,
    features_root: Path | str = DEFAULT_FEATURES_ROOT,
    raw_outcomes_root: Path | str = DEFAULT_RAW_OUTCOMES_ROOT,
    ev_threshold: float = 0.0,
    max_edge: float | None = DEFAULT_MAX_EDGE,
) -> pd.DataFrame:
    """Build a counterfactual bet log using closing lines only."""
    slates = _load_prediction_slates(
        Path(predictions_root),
        season_year=season_year,
    )
    if slates.empty:
        logger.info("close-line tracker: no prediction slates found")
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})

    closing = _closing_odds_for_slates(slates, features_root=Path(features_root))
    if closing.empty:
        logger.info("close-line tracker: no closing odds matched")
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})

    join_cols = ["commence_time", "home_id", "away_id"]
    for col in join_cols:
        if col not in closing.columns:
            logger.warning("close-line tracker: closing odds missing %s", col)
            return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})

    slate_cols = [
        "game_id", "game_date", "commence_time", "home_id", "away_id",
        "home_name", "away_name", "p_home",
        "home_lineup_source", "away_lineup_source",
    ]
    slate_cols = [c for c in slate_cols if c in slates.columns]
    merged = slates[slate_cols].merge(
        closing,
        on=join_cols,
        how="inner",
        suffixes=("", "_close"),
    )
    if merged.empty:
        logger.info("close-line tracker: no slate games matched closing odds")
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})

    merged = merged.drop_duplicates("game_id", keep="first")
    has_price = merged["home_price_american"].notna() & merged["away_price_american"].notna()
    merged = merged.loc[has_price].copy()
    if merged.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LOG_COLUMNS})

    annotated = annotate_bets(
        merged,
        model_p_col="p_home",
        home_price_col="home_price_american",
        away_price_col="away_price_american",
        ev_threshold=ev_threshold,
        max_edge=max_edge,
    )
    log = _slate_to_log_rows(annotated)
    log = _settle_outcomes(
        log,
        outcomes_root=Path(outcomes_root),
        features_root=Path(features_root),
        raw_outcomes_root=Path(raw_outcomes_root),
    )
    logger.info(
        "close-line tracker: %d close-line bets (%d settled)",
        len(log),
        int(log["outcome"].isin(["won", "lost", "push"]).sum()) if not log.empty else 0,
    )
    return log
