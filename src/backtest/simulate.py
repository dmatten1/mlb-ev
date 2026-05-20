"""Backtest harness: replay model predictions against historical odds.

Given a per-game ``slate`` frame from :func:`predict_slate` (or a
DataFrame in the same shape: ``home_score``, ``away_score``,
``recommended``, ``recommended_kelly``, ``home_price_american``,
``away_price_american``) and a starting bankroll, simulate the
chronological sequence of bets.

For each bet:
  * Stake = ``recommended_kelly × current_bankroll``
  * Settle = stake × (decimal_odds − 1) if win, else −stake

Tracks bankroll trajectory plus aggregate stats:
  * Total bets, win count, hit rate
  * Total wagered, total profit, ROI
  * Max drawdown, peak bankroll, final bankroll
  * CLV (closing line value): how the bet's market line compared to a
    later (closer to commence) line — but we don't have intra-game line
    movement here so CLV is omitted for v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.model.betting import american_to_decimal


@dataclass
class BacktestResult:
    bankroll_history: pd.DataFrame  # per-bet trajectory
    summary: dict                   # aggregate stats
    by_day: pd.DataFrame            # daily P/L aggregation
    bets: pd.DataFrame              # per-bet detail


def simulate_bankroll(
    slate: pd.DataFrame,
    *,
    starting_bankroll: float = 100.0,
    min_kelly: float = 0.0,
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
    chronological_col: str = "commence_time",
) -> BacktestResult:
    """Run a chronological bankroll simulation over the recommended bets.

    Parameters
    ----------
    slate
        Per-game frame from :func:`predict_slate` joined with actuals.
        Must include ``recommended`` (str: 'home', 'away', or ''),
        ``recommended_kelly`` (float), ``home_price_american``,
        ``away_price_american``, the score columns, and a chronological
        ordering column.
    starting_bankroll
        Initial bankroll. Doesn't affect ROI percentages but matters for
        bet sizes (Kelly is multiplicative).
    min_kelly
        Skip any "recommended" bet whose Kelly fraction is below this.
        Useful for filtering out marginal bets at very small Kelly sizes.
    home_score_col, away_score_col
        Columns in the slate giving the final score for settlement.
        Bets with missing scores are skipped (games not yet completed).
    chronological_col
        Column to sort by for bet sequencing. Defaults to
        ``commence_time``.

    Returns a :class:`BacktestResult`.
    """
    df = slate.copy()
    df = df[df["recommended"].isin(["home", "away"])]
    df = df[df["recommended_kelly"] >= min_kelly]
    df = df.dropna(subset=[home_score_col, away_score_col])
    df = df.sort_values(chronological_col).reset_index(drop=True)
    if df.empty:
        return BacktestResult(
            bankroll_history=pd.DataFrame(),
            summary={
                "starting_bankroll": starting_bankroll,
                "final_bankroll": starting_bankroll,
                "n_bets": 0, "n_wins": 0,
            },
            by_day=pd.DataFrame(),
            bets=pd.DataFrame(),
        )

    bankroll = float(starting_bankroll)
    peak = bankroll
    max_dd = 0.0
    history: list[dict] = []

    for i, row in df.iterrows():
        side = row["recommended"]
        kelly = float(row["recommended_kelly"])
        stake = bankroll * kelly
        price_col = (
            "home_price_american" if side == "home" else "away_price_american"
        )
        decimal_odds = float(american_to_decimal(float(row[price_col])))
        home_score = float(row[home_score_col])
        away_score = float(row[away_score_col])
        bet_won = (
            (side == "home" and home_score > away_score)
            or (side == "away" and away_score > home_score)
        )
        # Pushes are vanishingly rare in MLB moneylines (extra innings
        # always force a winner). If equal scores show up, treat as push.
        if home_score == away_score:
            profit = 0.0
            bet_won = None
        elif bet_won:
            profit = stake * (decimal_odds - 1.0)
        else:
            profit = -stake
        bankroll += profit
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, drawdown)
        history.append({
            "bet_idx": i,
            "commence_time": row.get(chronological_col),
            "game_date": row.get("game_date"),
            "home_name": row.get("home_name"),
            "away_name": row.get("away_name"),
            "side": side,
            "kelly": kelly,
            "stake": stake,
            "decimal_odds": decimal_odds,
            "home_score": home_score,
            "away_score": away_score,
            "won": bet_won,
            "profit": profit,
            "bankroll_after": bankroll,
            "drawdown": drawdown,
            "model_p_home": row.get("p_home"),
            "fair_p_home": row.get("home_fair_p"),
            "ev_at_bet": row.get("recommended_ev"),
        })

    hist_df = pd.DataFrame(history)

    n_bets = len(hist_df)
    settled = hist_df[hist_df["won"].notna()]
    n_wins = int(settled["won"].astype(int).sum())
    n_losses = len(settled) - n_wins
    hit_rate = n_wins / len(settled) if len(settled) else float("nan")
    total_wagered = float(hist_df["stake"].sum())
    total_profit = float(hist_df["profit"].sum())
    roi = total_profit / total_wagered if total_wagered > 0 else float("nan")
    growth = bankroll / starting_bankroll - 1.0

    summary = {
        "starting_bankroll": starting_bankroll,
        "final_bankroll": bankroll,
        "peak_bankroll": peak,
        "max_drawdown_pct": max_dd,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "hit_rate": hit_rate,
        "total_wagered": total_wagered,
        "total_profit": total_profit,
        "roi": roi,
        "bankroll_growth_pct": growth,
        "avg_stake": float(hist_df["stake"].mean()) if n_bets else 0.0,
        "avg_kelly": float(hist_df["kelly"].mean()) if n_bets else 0.0,
        "avg_ev_at_bet": float(hist_df["ev_at_bet"].mean()) if n_bets else 0.0,
    }

    by_day = (
        hist_df.assign(date=pd.to_datetime(hist_df["game_date"]).dt.date)
        .groupby("date")
        .agg(
            n_bets=("bet_idx", "count"),
            wagered=("stake", "sum"),
            profit=("profit", "sum"),
            wins=("won", lambda s: int(s.dropna().astype(int).sum())),
            bankroll_end=("bankroll_after", "last"),
        )
        .reset_index()
    )
    by_day["roi"] = np.where(by_day["wagered"] > 0,
                              by_day["profit"] / by_day["wagered"],
                              np.nan)

    return BacktestResult(
        bankroll_history=hist_df[["bet_idx", "commence_time", "bankroll_after", "drawdown"]],
        summary=summary,
        by_day=by_day,
        bets=hist_df,
    )


def threshold_sweep(
    slate: pd.DataFrame,
    *,
    ev_thresholds: Sequence[float] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20),
    max_edges: Sequence[float | None] = (None,),
    starting_bankroll: float = 100.0,
    re_threshold_recommendation: bool = True,
    home_score_col: str = "home_score",
    away_score_col: str = "away_score",
) -> pd.DataFrame:
    """Try multiple (ev_threshold, max_edge) pairs and return a comparison table.

    With ``re_threshold_recommendation=True``, re-derives ``recommended``
    at each (threshold, edge_cap) combo so a slate originally annotated
    with one setting can be replayed under another.

    ``max_edges`` lets you sweep an upper-bound cap on |model − market|
    edge (since empirically very-large edges are model errors, not
    value). ``None`` = no cap.
    """
    rows = []
    for max_edge in max_edges:
        for th in ev_thresholds:
            if re_threshold_recommendation:
                slate_th = slate.copy()
                home_pick = (
                    (slate_th["ev_home"] > slate_th["ev_away"])
                    & (slate_th["ev_home"] > th)
                )
                away_pick = (
                    (slate_th["ev_away"] > slate_th["ev_home"])
                    & (slate_th["ev_away"] > th)
                )
                if max_edge is not None:
                    home_pick &= slate_th["edge_home"].abs() <= max_edge
                    away_pick &= slate_th["edge_away"].abs() <= max_edge
                slate_th["recommended"] = np.where(
                    home_pick, "home",
                    np.where(away_pick, "away", ""),
                )
                slate_th["recommended_kelly"] = np.where(
                    slate_th["recommended"] == "home", slate_th["kelly_home"],
                    np.where(slate_th["recommended"] == "away", slate_th["kelly_away"], 0.0),
                )
                slate_th["recommended_ev"] = np.where(
                    slate_th["recommended"] == "home", slate_th["ev_home"],
                    np.where(slate_th["recommended"] == "away", slate_th["ev_away"], 0.0),
                )
            else:
                slate_th = slate
            result = simulate_bankroll(
                slate_th, starting_bankroll=starting_bankroll,
                home_score_col=home_score_col, away_score_col=away_score_col,
            )
            rows.append({
                "ev_threshold": th,
                "max_edge": max_edge if max_edge is not None else float("inf"),
                **{k: v for k, v in result.summary.items()
                   if k in {"n_bets", "n_wins", "hit_rate", "total_wagered",
                             "total_profit", "roi", "bankroll_growth_pct",
                             "max_drawdown_pct", "avg_ev_at_bet"}}
            })
    return pd.DataFrame(rows)
