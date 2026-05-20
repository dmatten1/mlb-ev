"""End-to-end inference: features + model -> bet recommendations.

Given a per-game feature frame (the same shape ``build_features`` emits
for training) and a trained runs model, produce per-game:

* ``home_runs_pred``, ``away_runs_pred`` — model output (HFA already
  applied via :func:`predict_runs`).
* ``p_home`` — Pythagorean win probability.
* When matched to odds: implied / fair probabilities, edge, EV both
  sides, Kelly fraction, recommended bet side.

This is the function the daily ``predict-tonight`` job will call, and
also what the backtest harness loops over historical dates with.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.model.betting import annotate_bets
from src.model.evaluate import pythag_win_prob
from src.model.runs_model import (
    DEFAULT_HFA_RUNS_BONUS, RunsModel, predict_runs,
)

logger = logging.getLogger("inference.predict_slate")


def predict_games(
    games: pd.DataFrame,
    runs_model: RunsModel,
    *,
    home_field_advantage_runs: float = DEFAULT_HFA_RUNS_BONUS,
    pythag_exponent: float = 1.83,
) -> pd.DataFrame:
    """Add ``home_runs_pred``, ``away_runs_pred``, ``p_home`` columns.

    No odds required — useful for "what does the model think?" reports
    without an EV column.
    """
    out = predict_runs(
        runs_model, games, home_field_advantage_runs=home_field_advantage_runs,
    )
    out["p_home"] = pythag_win_prob(
        out["home_runs_pred"].to_numpy(),
        out["away_runs_pred"].to_numpy(),
        exponent=pythag_exponent,
    )
    return out


def _odds_join_keys(games: pd.DataFrame) -> pd.DataFrame:
    """Normalize game_date + team ids to dtypes the odds frame uses."""
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.tz_localize(None).dt.normalize()
    g["home_id"] = pd.to_numeric(g["home_id"], errors="coerce").astype("Int64")
    g["away_id"] = pd.to_numeric(g["away_id"], errors="coerce").astype("Int64")
    return g


def _odds_with_game_date(odds: pd.DataFrame) -> pd.DataFrame:
    """Derive a ``game_date`` column from the odds frame's commence_time.

    The Odds API ``commence_time`` is UTC. MLB games west of UTC-5 might
    technically start one ET calendar day earlier than their UTC date,
    but every team's first pitch is in the evening local time, so the
    UTC date and the conventional "game date" coincide except for very
    late-PT games (which start ~03:00 UTC next day). We use America/New_York
    as the canonical baseball calendar.
    """
    o = odds.copy()
    o["commence_time"] = pd.to_datetime(o["commence_time"], utc=True)
    o["game_date"] = (
        o["commence_time"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    )
    o["home_id"] = pd.to_numeric(o["home_id"], errors="coerce").astype("Int64")
    o["away_id"] = pd.to_numeric(o["away_id"], errors="coerce").astype("Int64")
    return o


def predict_slate(
    games: pd.DataFrame,
    runs_model: RunsModel,
    odds: pd.DataFrame,
    *,
    home_field_advantage_runs: float = DEFAULT_HFA_RUNS_BONUS,
    pythag_exponent: float = 1.83,
    ev_threshold: float = 0.0,
    max_edge: float | None = 0.07,
    kelly_fraction_mult: float = 0.0625,
    kelly_cap: float | None = 0.01,
    daily_stake_cap: float | None = 0.05,
    vig_method: str = "proportional",
    require_odds: bool = True,
) -> pd.DataFrame:
    """Full pipeline: features + odds -> bet recommendations.

    Parameters
    ----------
    games
        Per-game features (training_<year>.parquet format). Must
        include ``game_id``, ``game_date``, ``home_id``, ``away_id``,
        and whatever feature columns the model was trained on.
    runs_model
        A trained :class:`RunsModel`.
    odds
        Per-game closing-line frame from
        :func:`src.inference.odds_loader.best_lines_per_game` after
        :func:`attach_team_ids`. Must have ``commence_time``,
        ``home_id``, ``away_id``, ``home_price_american``,
        ``away_price_american``.
    ev_threshold
        Minimum EV (per dollar) required for ``recommended`` to be
        populated. Default 0.0 — small-edge bets are where the model's
        value lives, so we don't filter them out.
    max_edge
        Maximum |model_p − fair_p| on the recommended side. Bets where
        the model thinks it has a huge edge tend to be model errors, not
        value (the market knows things our features don't). Default
        0.07 (7pp). Set ``None`` to disable.
    kelly_fraction_mult
        Multiplier on full Kelly. Default 0.0625 (1/16 Kelly) — very
        conservative; MLB moneylines are high-variance enough that
        trading some growth rate for much tighter drawdowns pays off.
        Targets ~1% daily bankroll exposure on a typical slate.
    kelly_cap
        Hard ceiling on bet-as-fraction-of-bankroll (default 1%).
    daily_stake_cap
        Hard ceiling on the slate's combined stake (default 5%). If
        per-bet kellys sum above this, every stake is rescaled
        proportionally so the day's total equals the cap.
    require_odds
        If True (default), drop games without an odds match. If False,
        emit a row per game with NaN odds columns (useful for the
        "what does the model say even without a line?" report).

    Returns a DataFrame with one row per game, columns:
        game_id, game_date, home_id, away_id, home_name, away_name,
        home_runs_pred, away_runs_pred, p_home,
        home_price_american, away_price_american, home_fair_p, away_fair_p,
        edge_home, edge_away, ev_home, ev_away,
        kelly_home, kelly_away, recommended, recommended_kelly,
        recommended_ev, ...
    """
    preds = predict_games(
        games, runs_model,
        home_field_advantage_runs=home_field_advantage_runs,
        pythag_exponent=pythag_exponent,
    )

    if odds is None or odds.empty:
        if require_odds:
            return preds.iloc[0:0].copy()
        # No odds at all — return predictions with NaN price cols.
        for c in ("home_price_american", "away_price_american"):
            preds[c] = np.nan
        return preds

    g = _odds_join_keys(preds)
    o = _odds_with_game_date(odds)
    join_cols = ["game_date", "home_id", "away_id"]
    odds_keep = join_cols + ["commence_time", "snapshot_ts", "n_books",
                              "home_price_american", "away_price_american",
                              "home_book", "away_book"]
    # Older odds frames (e.g. tests) may not carry the book columns.
    odds_keep = [c for c in odds_keep if c in o.columns]
    # Doubleheaders share (date, home, away) — without a true game start
    # time on the games side we can't disambiguate the second game from
    # the first. For v1 we collapse duplicate odds entries to the LAST
    # pre-game snapshot, accepting that the second game of a doubleheader
    # will be matched against the wrong (game-1) odds row sometimes.
    # When we add game_start_time to the features parquet we can swap in
    # an exact merge_asof match.
    odds_dedup = (
        o[odds_keep]
        .sort_values("commence_time")
        .drop_duplicates(join_cols, keep="last")
    )
    games_dedup = g.drop_duplicates(["game_id"], keep="first")
    merged = games_dedup.merge(
        odds_dedup, on=join_cols, how="inner" if require_odds else "left"
    )
    merged = merged.drop_duplicates("game_id", keep="first")
    if merged.empty:
        return merged
    has_price = merged["home_price_american"].notna() & merged["away_price_american"].notna()
    annotated = merged.loc[has_price].copy()
    if annotated.empty:
        if not require_odds:
            return merged
        return annotated

    annotated = annotate_bets(
        annotated,
        model_p_col="p_home",
        home_price_col="home_price_american",
        away_price_col="away_price_american",
        ev_threshold=ev_threshold,
        max_edge=max_edge,
        kelly_fraction_mult=kelly_fraction_mult,
        kelly_cap=kelly_cap,
        daily_stake_cap=daily_stake_cap,
        vig_method=vig_method,
    )

    out_cols = [
        "game_id", "game_date", "commence_time",
        "home_id", "away_id", "home_name", "away_name",
        "home_score", "away_score",  # carried through when present (backtest)
        "home_runs_pred", "away_runs_pred", "p_home",
        "snapshot_ts", "n_books",
        "home_price_american", "away_price_american",
        "home_book", "away_book",  # which book offered each side's best price
        "home_decimal", "away_decimal",
        "home_implied", "away_implied",
        "home_fair_p", "away_fair_p", "vig",
        "edge_home", "edge_away",
        "ev_home", "ev_away",
        "kelly_home", "kelly_away",
        "recommended", "recommended_ev", "recommended_kelly",
        "recommended_kelly_pre_daily", "risk_ref_kelly", "risk_units",
        # Lineup provenance (projected slate only — carried through when present).
        "home_lineup_source", "away_lineup_source",
    ]
    out_cols = [c for c in out_cols if c in annotated.columns]
    return annotated[out_cols].reset_index(drop=True)
