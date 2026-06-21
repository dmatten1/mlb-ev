"""Team RS baseline + Ridge residual adjustments (prototype v2).

    μ_side = shrunk_team_rs_30d + Ridge(Δlineup, ΔSP, BP, ...)

Shrunk baseline blends rolling park-adjusted team RS with league average:
    w = min(1, n_games_in_window / 30)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.model.evaluate import pythag_win_prob
from src.model.prototype.team_runs_baseline import (
    TeamBaselineConfig,
    attach_team_offense_norms,
    build_baselines_for_games,
)
from src.model.runs_model import DEFAULT_HFA_RUNS_BONUS, predict_runs, stack_sides

DEFAULT_TEAM_BASELINE_PATH = Path("data/models/runs_model_team_baseline_v2.pkl")

# Adjustment-only features (Ridge learns residuals on top of team RS anchor).
ADJUSTMENT_FEATURE_COLS: tuple[str, ...] = (
    "lineup_delta",
    "off_xwOBA_matchup_adj",
    "off_xwOBA_30d_matchup_adj",
    "off_pitch_rv_matchup",
    "opp_sp_SIERA",
    "opp_sp_SIERA_30d",
    "opp_sp_xwOBA_matchup_adj",
    "opp_sp_pitch_rv_matchup",
    "opp_bp_xwOBA_matchup_adj",
    "opp_bp_SIERA_matchup",
    "is_home",
)


@dataclass
class TeamBaselineRunsModel:
    """Shrunk team RS anchor + Ridge residual adjusters."""

    adj_model: Ridge
    adj_scaler: StandardScaler
    feature_cols: tuple[str, ...]
    league_avg_runs: float
    baseline_config: TeamBaselineConfig
    train_n: int
    impute_values: pd.Series
    version: str = "v2-team-rs-baseline"


def _stack_with_baselines(games: pd.DataFrame) -> pd.DataFrame:
    """Like stack_sides but carries per-side offense baseline + deltas."""
    pieces = []
    for side, opp in (("home", "away"), ("away", "home")):
        off_prefix = f"{side}_off_"
        sp_prefix = f"{opp}_sp_"
        bp_prefix = f"{opp}_bp_"
        out = pd.DataFrame()
        for c in ("game_id", "game_date", "season_year", "home_name", "away_name"):
            if c in games.columns:
                out[c] = games[c].to_numpy()
        out["side"] = side
        out["is_home"] = 1 if side == "home" else 0
        out["runs"] = games[f"{side}_score"].to_numpy()
        out["off_baseline"] = games[f"{side}_off_baseline"].to_numpy()

        off_norm_col = f"{side}_off_norm_30d"
        off_cum_col = f"{side}_off_xwOBA_matchup_adj"
        off_roll_col = f"{side}_off_xwOBA_30d_matchup_adj"
        if off_norm_col in games.columns and off_cum_col in games.columns:
            cum = games[off_cum_col].astype(float)
            norm = games[off_norm_col].astype(float)
            out["lineup_delta"] = (cum - norm).to_numpy()
        else:
            out["lineup_delta"] = np.nan

        for c in games.columns:
            if c.startswith(off_prefix):
                out[f"off_{c[len(off_prefix):]}"] = games[c].to_numpy()
            elif c.startswith(sp_prefix):
                out[f"opp_sp_{c[len(sp_prefix):]}"] = games[c].to_numpy()
            elif c.startswith(bp_prefix):
                out[f"opp_bp_{c[len(bp_prefix):]}"] = games[c].to_numpy()
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def _design_matrix(
    stacked: pd.DataFrame,
    feature_cols: tuple[str, ...],
    impute: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    X = stacked[list(feature_cols)].astype(float).copy()
    if impute is None:
        impute = X.median(numeric_only=True)
    X = X.fillna(impute)
    return X.to_numpy(), impute


def prepare_games_with_baselines(
    games: pd.DataFrame,
    history_games: pd.DataFrame,
    *,
    config: TeamBaselineConfig | None = None,
) -> pd.DataFrame:
    """Attach shrunk baselines, eligibility flag, and lineup norms."""
    cfg = config or TeamBaselineConfig()
    with_base, _, _ = build_baselines_for_games(history_games, games, config=cfg)
    return attach_team_offense_norms(with_base, history_games, rolling_days=cfg.rolling_days)


def train_team_baseline_model(
    games_train: pd.DataFrame,
    history_games: pd.DataFrame | None = None,
    *,
    feature_cols: tuple[str, ...] = ADJUSTMENT_FEATURE_COLS,
    alpha: float = 1.0,
    config: TeamBaselineConfig | None = None,
) -> TeamBaselineRunsModel:
    """Fit Ridge on residuals after shrunk team RS baseline."""
    cfg = config or TeamBaselineConfig()
    hist = games_train if history_games is None else history_games
    prepared = prepare_games_with_baselines(games_train, hist, config=cfg)
    _, league_avg, _ = build_baselines_for_games(hist, games_train, config=cfg)

    stacked = _stack_with_baselines(prepared).dropna(subset=["runs", "off_baseline"])
    y_resid = stacked["runs"].astype(float) - stacked["off_baseline"].astype(float)
    X_raw, impute = _design_matrix(stacked, feature_cols)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    model = Ridge(alpha=alpha)
    model.fit(X, y_resid)

    return TeamBaselineRunsModel(
        adj_model=model,
        adj_scaler=scaler,
        feature_cols=feature_cols,
        league_avg_runs=league_avg,
        baseline_config=cfg,
        train_n=len(stacked),
        impute_values=impute,
    )


def predict_team_baseline_runs(
    model: TeamBaselineRunsModel,
    games: pd.DataFrame,
    history_games: pd.DataFrame,
) -> pd.DataFrame:
    """Return games with ``home_runs_pred_tb``, ``away_runs_pred_tb``, ``p_home_tb``."""
    prepared = prepare_games_with_baselines(games, history_games, config=model.baseline_config)
    stacked = _stack_with_baselines(prepared).copy()
    X_raw, _ = _design_matrix(stacked, model.feature_cols, impute=model.impute_values)
    X = model.adj_scaler.transform(X_raw)
    resid = model.adj_model.predict(X)
    stacked = stacked.assign(
        runs_pred=stacked["off_baseline"].astype(float) + resid,
    )

    pivoted = (
        stacked.pivot(index="game_id", columns="side", values="runs_pred")
        .rename(columns={"home": "home_runs_pred_tb", "away": "away_runs_pred_tb"})
        .reset_index()
    )
    out = games.merge(pivoted, on="game_id", how="left")
    if DEFAULT_HFA_RUNS_BONUS != 0.0:
        out["home_runs_pred_tb"] = out["home_runs_pred_tb"] + DEFAULT_HFA_RUNS_BONUS
    out["p_home_tb"] = pythag_win_prob(
        out["home_runs_pred_tb"].to_numpy(),
        out["away_runs_pred_tb"].to_numpy(),
    )
    # Carry baseline metadata from prepared frame
    meta_cols = [
        "home_off_baseline", "away_off_baseline",
        "home_team_rs_raw_roll", "away_team_rs_raw_roll",
        "home_team_season_games", "away_team_season_games",
        "home_team_rs_cal_n", "away_team_rs_cal_n",
        "league_avg_raw_runs",
        "prototype_bet_eligible", "prototype_min_team_games",
    ]
    out = out.drop(columns=[c for c in meta_cols if c in out.columns], errors="ignore")
    out = out.merge(
        prepared[["game_id"] + [c for c in meta_cols if c in prepared.columns]],
        on="game_id",
        how="left",
    )
    return out


def save_team_baseline_model(
    model: TeamBaselineRunsModel,
    path: Path | str = DEFAULT_TEAM_BASELINE_PATH,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_team_baseline_model(path: Path | str = DEFAULT_TEAM_BASELINE_PATH) -> TeamBaselineRunsModel:
    with Path(path).open("rb") as f:
        return pickle.load(f)
