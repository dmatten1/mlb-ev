"""Negative-binomial scoring simulation for P(home wins).

Uses Ridge run means (μ) from the production runs model and a global
overdispersion parameter α fit from training residuals:

    Var(runs | μ) ≈ μ + μ² / α

Larger α → closer to Poisson; smaller α → heavier tails (more blowouts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DispersionParams:
    """Global NB overdispersion fit on stacked side-level residuals."""

    alpha: float
    train_n: int
    residual_var_mean: float


def estimate_dispersion(
    actual_runs: np.ndarray,
    predicted_runs: np.ndarray,
    *,
    min_alpha: float = 0.5,
    max_alpha: float = 50.0,
) -> DispersionParams:
    """Method-of-moments α from side-level (actual, predicted) pairs."""
    a = np.asarray(actual_runs, dtype=float)
    p = np.asarray(predicted_runs, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p)) & (p > 0)
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return DispersionParams(alpha=5.0, train_n=0, residual_var_mean=float("nan"))

    resid = a - p
    var = float(np.var(resid))
    mu_mean = float(np.mean(p))
    mu2_mean = float(np.mean(p ** 2))
    # E[(r-μ)²] ≈ E[μ + μ²/α] on stacked sides → solve for α.
    denom = max(mu2_mean, 1e-6)
    alpha = mu2_mean / max(var - mu_mean, 0.05)
    alpha = float(np.clip(alpha, min_alpha, max_alpha))
    return DispersionParams(alpha=alpha, train_n=int(len(a)), residual_var_mean=var)


def _nb_n_p(mu: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """SciPy ``nbinom(n, p)`` params with mean μ and Var = μ + μ²/α."""
    mu = np.maximum(np.asarray(mu, dtype=float), 0.05)
    var = mu + (mu ** 2) / max(alpha, 0.1)
    var = np.maximum(var, mu + 1e-4)
    p = mu / var
    p = np.clip(p, 1e-9, 1 - 1e-9)
    n = mu * p / (1 - p)
    n = np.maximum(n, 1e-6)
    return n, p


def simulate_p_home(
    home_mu: np.ndarray,
    away_mu: np.ndarray,
    *,
    alpha: float,
    n_sim: int = 8000,
    seed: int | None = 42,
) -> np.ndarray:
    """Monte Carlo P(home_runs > away_runs) + 0.5·P(tie).

    Ties are rare in baseball but included for completeness.
    """
    h_mu = np.asarray(home_mu, dtype=float)
    a_mu = np.asarray(away_mu, dtype=float)
    n_games = len(h_mu)
    out = np.full(n_games, np.nan)

    valid = ~(np.isnan(h_mu) | np.isnan(a_mu))
    if not valid.any():
        return out

    rng = np.random.default_rng(seed)
    h_n, h_p = _nb_n_p(h_mu[valid], alpha)
    a_n, a_p = _nb_n_p(a_mu[valid], alpha)

    # (n_sim, n_valid) draws
    h_draws = rng.negative_binomial(h_n, h_p, size=(n_sim, valid.sum()))
    a_draws = rng.negative_binomial(a_n, a_p, size=(n_sim, valid.sum()))
    home_wins = (h_draws > a_draws).mean(axis=0)
    ties = (h_draws == a_draws).mean(axis=0)
    out[valid] = home_wins + 0.5 * ties
    return np.clip(out, 1e-9, 1 - 1e-9)
