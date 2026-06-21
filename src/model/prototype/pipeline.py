"""Train / predict the offline prototype model.

Reuses production :class:`~src.model.runs_model.RunsModel` for μ (run means)
and layers NB simulation + optional tail calibration on top.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.evaluate import pythag_win_prob
from src.model.prototype.nb_win_prob import (
    DispersionParams,
    estimate_dispersion,
    simulate_p_home,
)
from src.model.prototype.tail_calibrator import TailCalibrator, fit_tail_calibration
from src.model.runs_model import (
    BULLPEN_FEATURE_COLS,
    RunsModel,
    predict_runs,
    predict_side_runs,
    stack_sides,
    train_runs_model,
)

DEFAULT_PROTOTYPE_PATH = Path("data/models/runs_model_prototype_v1.pkl")


@dataclass
class PrototypeModel:
    """Serializable prototype artifact (separate from production cache)."""

    runs_model: RunsModel
    dispersion: DispersionParams
    tail_cal: TailCalibrator | None
    n_sim: int = 8000
    version: str = "v1-nb-tail"


def train_prototype(
    games_train: pd.DataFrame,
    games_cal: pd.DataFrame | None = None,
    *,
    feature_cols: tuple[str, ...] = BULLPEN_FEATURE_COLS,
    fit_tail_cal: bool = True,
    n_sim: int = 8000,
) -> PrototypeModel:
    """Fit μ (Ridge), dispersion α, and optional tail calibrator.

    Parameters
    ----------
    games_train
        Wide per-game training frame (completed games with scores).
    games_cal
        Held-out slice for tail isotonic (defaults to last 20% of train
        chronologically if omitted).
    """
    runs_model = train_runs_model(games_train, feature_cols)

    stacked = stack_sides(games_train).dropna(subset=["runs"])
    mu = predict_side_runs(runs_model, stacked)
    dispersion = estimate_dispersion(stacked["runs"].to_numpy(), mu)

    tail_cal: TailCalibrator | None = None
    if fit_tail_cal:
        cal_games = games_cal
        if cal_games is None and "game_date" in games_train.columns:
            gd = pd.to_datetime(games_train["game_date"])
            cutoff = gd.quantile(0.80)
            cal_games = games_train[gd >= cutoff].copy()
        if cal_games is not None and not cal_games.empty:
            cal_preds = predict_prototype_raw(
                cal_games,
                runs_model=runs_model,
                dispersion=dispersion,
                n_sim=n_sim,
            )
            y = cal_games["home_win"].astype(float).to_numpy() if "home_win" in cal_games.columns else (
                (cal_games["home_score"] > cal_games["away_score"]).astype(float).to_numpy()
            )
            tail_cal = fit_tail_calibration(cal_preds["p_home_nb"].to_numpy(), y)

    return PrototypeModel(
        runs_model=runs_model,
        dispersion=dispersion,
        tail_cal=tail_cal,
        n_sim=n_sim,
    )


def predict_prototype_raw(
    games: pd.DataFrame,
    *,
    runs_model: RunsModel,
    dispersion: DispersionParams,
    n_sim: int = 8000,
    home_field_advantage_runs: float | None = None,
) -> pd.DataFrame:
    """Add ``home_runs_pred``, ``away_runs_pred``, ``p_home_pythag``, ``p_home_nb``."""
    from src.model.runs_model import DEFAULT_HFA_RUNS_BONUS

    hfa = DEFAULT_HFA_RUNS_BONUS if home_field_advantage_runs is None else home_field_advantage_runs
    out = predict_runs(runs_model, games, home_field_advantage_runs=hfa)
    out["p_home_pythag"] = pythag_win_prob(
        out["home_runs_pred"].to_numpy(),
        out["away_runs_pred"].to_numpy(),
    )
    out["p_home_nb"] = simulate_p_home(
        out["home_runs_pred"].to_numpy(),
        out["away_runs_pred"].to_numpy(),
        alpha=dispersion.alpha,
        n_sim=n_sim,
    )
    return out


def predict_prototype(
    games: pd.DataFrame,
    model: PrototypeModel,
    *,
    home_field_advantage_runs: float | None = None,
) -> pd.DataFrame:
    """Production-shaped output with baseline + prototype probability columns."""
    out = predict_prototype_raw(
        games,
        runs_model=model.runs_model,
        dispersion=model.dispersion,
        n_sim=model.n_sim,
        home_field_advantage_runs=home_field_advantage_runs,
    )
    out["p_home"] = out["p_home_pythag"]
    out["p_home_proto"] = out["p_home_nb"]
    if model.tail_cal is not None:
        out["p_home_proto_cal"] = model.tail_cal.transform(out["p_home_nb"].to_numpy())
    else:
        out["p_home_proto_cal"] = out["p_home_nb"]
    return out


def save_prototype(model: PrototypeModel, path: Path | str = DEFAULT_PROTOTYPE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_prototype(path: Path | str = DEFAULT_PROTOTYPE_PATH) -> PrototypeModel:
    with Path(path).open("rb") as f:
        return pickle.load(f)
