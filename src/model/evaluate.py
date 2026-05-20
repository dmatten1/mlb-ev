"""Backtest helpers for the runs / Pythag pipeline.

Provides:

* ``runs_metrics``   — MAE / RMSE / R^2 for the runs regression.
* ``pythag_metrics`` — accuracy / log loss / Brier when we collapse
  (home_runs_pred, away_runs_pred) into a Pythagorean win probability
  and compare to the binary outcome.
* ``calibration_table`` / ``calibration_plot`` — does p_home actually
  correspond to the empirical home win rate at each probability bucket?

Baselines we compare against in the notebook:
1. Always pick home (≈ 54% by home-field advantage).
2. Predict league-mean runs for both sides → constant p_home = 0.5.
3. Pick the side with the lower opposing-starter SIERA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.sabermetrics import pythagorean_win_prob

EPS = 1e-9


@dataclass
class RunsMetrics:
    n: int
    mae: float
    rmse: float
    r2: float
    bias: float  # mean(pred - actual). Positive => over-predicts runs.


def runs_metrics(actual: np.ndarray | pd.Series,
                 predicted: np.ndarray | pd.Series) -> RunsMetrics:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    err = p - a
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2)) or EPS
    return RunsMetrics(
        n=int(mask.sum()),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        r2=float(1 - ss_res / ss_tot),
        bias=float(err.mean()),
    )


@dataclass
class PythagMetrics:
    n: int
    accuracy: float
    log_loss: float
    brier: float
    home_win_rate: float
    mean_p_home: float


def pythag_win_prob(home_runs_pred: np.ndarray | pd.Series,
                    away_runs_pred: np.ndarray | pd.Series,
                    *, exponent: float = 1.83) -> np.ndarray:
    """Vectorized Pythag. Mirrors sabermetrics.pythagorean_win_prob but
    works element-wise on arrays."""
    h = np.maximum(np.asarray(home_runs_pred, dtype=float), EPS)
    a = np.maximum(np.asarray(away_runs_pred, dtype=float), EPS)
    return h ** exponent / (h ** exponent + a ** exponent)


def pythag_metrics(home_win_actual: np.ndarray | pd.Series,
                   p_home: np.ndarray | pd.Series) -> PythagMetrics:
    y = np.asarray(home_win_actual, dtype=float)
    p = np.clip(np.asarray(p_home, dtype=float), EPS, 1 - EPS)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    return PythagMetrics(
        n=int(mask.sum()),
        accuracy=float(((p > 0.5).astype(int) == y.astype(int)).mean()),
        log_loss=float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        brier=float(np.mean((p - y) ** 2)),
        home_win_rate=float(y.mean()),
        mean_p_home=float(p.mean()),
    )


def calibration_table(p_home: np.ndarray | pd.Series,
                      home_win_actual: np.ndarray | pd.Series,
                      *, bins: int = 10) -> pd.DataFrame:
    """Bucket predictions into ``bins`` quantiles and compare predicted
    vs actual home-win rate. Good calibration => columns track each other.
    """
    df = pd.DataFrame({
        "p_home": np.asarray(p_home, dtype=float),
        "home_win": np.asarray(home_win_actual, dtype=float),
    }).dropna()
    df["bucket"] = pd.qcut(df["p_home"], bins, labels=False, duplicates="drop")
    g = df.groupby("bucket").agg(
        n=("home_win", "size"),
        p_home_mean=("p_home", "mean"),
        actual_home_win=("home_win", "mean"),
    )
    g["calibration_gap"] = g["actual_home_win"] - g["p_home_mean"]
    return g.round(4)
