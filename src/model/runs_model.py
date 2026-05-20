"""Baseline runs-projection model (matchup engine v1).

Predicts the runs ONE team will score in ONE game from:

* That team's lineup-weighted offensive composites (``home_off_*`` /
  ``away_off_*`` columns from the training parquet).
* The opposing starter's point-in-time pitcher features (``away_sp_*`` /
  ``home_sp_*``).
* A small ``is_home`` flag.

The model is side-symmetric — we stack each game into TWO training rows
(home-as-offense + away-as-offense, with the relevant columns renamed
side-agnostically). A single Ridge regression learns the runs mapping
once. At inference we call it twice per game (once with home as the
offense, once with away) and feed the two outputs into Pythag for win
probability.

Why Ridge for v1:
* Interpretable coefficients — useful when reasoning about which
  features actually drive the bet.
* Stable in the presence of correlated features (SIERA, K%, xwOBA
  all move together).
* No nonlinearity baked in — matches the linear "baseline" framing
  before we layer on the structural multiplicative effects (park,
  OAA) from ``src.features.matchup``.

Why the feature list is a parameter:
* The Option-B work (rolling windows, park factor, OAA, rest days)
  will add columns to ``build_features.py``. Anything new just needs
  to be added to the ``feature_cols`` argument here — no surgery on
  the training/prediction code.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Default feature lists.
#
# Why this exact set: SIERA is mathematically a function of K%/BB%/GB%/FB%/PU%,
# and xwOBA absorbs HR rate. Including all of them simultaneously gives Ridge
# enough collinearity that it redistributes weight arbitrarily — most visibly,
# `opp_sp_SIERA` flips to a negative coefficient even though more strikeouts
# from the opposing pitcher should suppress runs. We instead keep one summary
# stat from each "tier":
#   * SIERA       — skill summary (encodes K%/BB%/contact mix)
#   * xwOBA       — outcome summary (encodes HR/contact quality, park-adjusted)
#   * Barrel%     — quality-of-contact (independent of the above)
#   * HR% / BB%   — discipline-side hitter signals not fully captured by xwOBA
# Coefficients with this set come out signed the way baseball intuition demands.
#
# We use the **park-adjusted** xwOBA variant (multiplied per hitter by the
# wOBA-weighted personalized park factor built from Savant hit-type splits)
# so the model doesn't have to learn park effects from a one-hot — they're
# already baked into the feature.
# ---------------------------------------------------------------------------

DEFAULT_OFF_FEATURES: tuple[str, ...] = (
    "off_xwOBA_matchup_adj",
    "off_Barrel_pct",
    "off_HR_pct",
    "off_BB_pct",
)
DEFAULT_OPP_SP_FEATURES: tuple[str, ...] = (
    "opp_sp_SIERA",
    "opp_sp_xwOBA_matchup_adj",
    "opp_sp_Barrel_pct",
)
DEFAULT_CONTEXT_FEATURES: tuple[str, ...] = ("is_home",)

DEFAULT_FEATURE_COLS: tuple[str, ...] = (
    *DEFAULT_OFF_FEATURES,
    *DEFAULT_OPP_SP_FEATURES,
    *DEFAULT_CONTEXT_FEATURES,
)

# Pre-matchup-adjustment baseline — same feature mix but using the
# park-neutral xwOBA. Use to ablate the matchup model lift.
BASELINE_FEATURE_COLS: tuple[str, ...] = (
    "off_xwOBA", "off_Barrel_pct", "off_HR_pct", "off_BB_pct",
    "opp_sp_SIERA", "opp_sp_xwOBA", "opp_sp_Barrel_pct",
    "is_home",
)

# Pure-rolling variant: last-30-day metrics with full matchup adjustment.
ROLLING_FEATURE_COLS: tuple[str, ...] = (
    "off_xwOBA_30d_matchup_adj", "off_Barrel_pct_30d",
    "off_HR_pct_30d", "off_BB_pct_30d",
    "opp_sp_SIERA_30d", "opp_sp_xwOBA_30d_matchup_adj",
    "opp_sp_Barrel_pct_30d",
    "is_home",
)

# Combined: cumulative + rolling. Lets the model trade off stable
# season-long signal against short-term form.
COMBINED_FEATURE_COLS: tuple[str, ...] = (
    *DEFAULT_FEATURE_COLS,
    "off_xwOBA_30d_matchup_adj",
    "opp_sp_xwOBA_30d_matchup_adj",
)

# Bullpen features (the opp_bp_* fields are LINEUP-RESOLVED: per-slot
# routing to the opposing-team BP pool of matching handedness, then
# lineup-weighted composite over the lineup that faces it. The xwOBA
# variant additionally has park × OAA matchup adjustment baked in.)
#
# Cumulative > Rolling for the BP pool — empirically. Why: the pool
# itself is already a rolling 30-day window of "who's been pitching
# recently", so we're double-windowing if we ALSO use 30-day rate
# stats for each pool member. Per-reliever rate stats are noisy on
# 30-day samples (50-70 IP/season -> ~10-15 IP in 30 days). Using
# season-cumulative per-reliever stats over a rolling pool gives us
# "stable skill × current usage" — the cleanest BP signal.
#
# Barrel% on BP is omitted: it's collinear with BP xwOBA in the
# small-sample regime relievers live in, and Ridge gives it wrong-sign
# coefficients that hurt accuracy.
BULLPEN_BP_FEATURES: tuple[str, ...] = (
    "opp_bp_xwOBA_matchup_adj",
    "opp_bp_SIERA_matchup",
)

# Production feature set. SP cum + rolling (matchup-adjusted) + BP cum.
# This is what trained at 55.5% accuracy / Brier 0.2459 / LogLoss 0.6850
# on 2023+2024 -> 2025 with HFA=0.27 (best to date).
BULLPEN_FEATURE_COLS: tuple[str, ...] = (
    *COMBINED_FEATURE_COLS,
    *BULLPEN_BP_FEATURES,
)

# Kitchen-sink (for diagnosing collinearity).
KITCHEN_SINK_FEATURE_COLS: tuple[str, ...] = (
    "off_xwOBA", "off_Barrel_pct", "off_HR_pct", "off_K_pct", "off_BB_pct",
    "opp_sp_SIERA", "opp_sp_xwOBA", "opp_sp_Barrel_pct",
    "opp_sp_K_pct", "opp_sp_BB_pct", "opp_sp_HR_pct",
    "is_home",
)


# ---------------------------------------------------------------------------
# Side stacking
# ---------------------------------------------------------------------------

def stack_sides(games: pd.DataFrame) -> pd.DataFrame:
    """Return a long-form DataFrame: 2 rows per game (one per scoring side).

    Renames the side-specific columns to side-agnostic ones:
        home_off_xwOBA / away_off_xwOBA  ->  off_xwOBA
        home_sp_SIERA  / away_sp_SIERA   ->  opp_sp_SIERA   (NOTE: OPP, not own)

    Auto-discovers every ``<side>_off_*`` and ``<opp>_sp_*`` column in the
    wide frame, so newly-added features (park-adjusted, rolling-30d, etc.)
    flow through without any code changes here.

    Target column ``runs`` is the score that side put up. ``opponent_runs``
    is preserved for evaluation.
    """
    pieces = []
    for side, opp in (("home", "away"), ("away", "home")):
        off_prefix = f"{side}_off_"
        sp_prefix = f"{opp}_sp_"
        bp_prefix = f"{opp}_bp_"
        out = pd.DataFrame()
        for c in ("game_id", "game_date", "season_year",
                  "home_name", "away_name"):
            if c in games.columns:
                out[c] = games[c].to_numpy()
        out["side"] = side
        out["is_home"] = 1 if side == "home" else 0
        out["runs"] = games[f"{side}_score"].to_numpy()
        out["opponent_runs"] = games[f"{opp}_score"].to_numpy()

        for c in games.columns:
            if c.startswith(off_prefix):
                out[f"off_{c[len(off_prefix):]}"] = games[c].to_numpy()
            elif c.startswith(sp_prefix):
                out[f"opp_sp_{c[len(sp_prefix):]}"] = games[c].to_numpy()
            elif c.startswith(bp_prefix):
                out[f"opp_bp_{c[len(bp_prefix):]}"] = games[c].to_numpy()
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------

@dataclass
class RunsModel:
    """Trained Ridge regressor + the scaler + the feature list it was
    fit on. ``predict_runs(model, games)`` reuses all three."""

    model: Ridge
    scaler: StandardScaler
    feature_cols: tuple[str, ...]
    target_mean: float
    target_std: float
    train_n: int
    impute_values: pd.Series  # per-feature median used to fill NaN at inference


def _design_matrix(
    stacked: pd.DataFrame,
    feature_cols: Sequence[str],
    impute: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    """Pull ``feature_cols`` out of the stacked frame, median-impute NaN.

    Returns ``(X, imputation_used)`` so the caller can persist the imputer.
    """
    X = stacked[list(feature_cols)].astype(float).copy()
    if impute is None:
        impute = X.median(numeric_only=True)
    X = X.fillna(impute)
    return X.to_numpy(), impute


def train_runs_model(
    games_train: pd.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS,
    *,
    alpha: float = 1.0,
    drop_target_na: bool = True,
) -> RunsModel:
    """Fit a Ridge regression on stacked sides.

    Parameters
    ----------
    games_train
        Wide per-game training parquet (output of ``build_training_features``).
    feature_cols
        Columns (post side-stacking) to feed the model. Defaults to the curated
        slim list; pass an extended list to layer in Option-B features.
    alpha
        Ridge L2 penalty. ``1.0`` is a sensible default for standardized
        features at this sample size.
    drop_target_na
        Drop training rows whose ``runs`` column is missing (shouldn't
        happen for completed games but guards against partial pulls).
    """
    stacked = stack_sides(games_train)
    if drop_target_na:
        stacked = stacked.dropna(subset=["runs"]).reset_index(drop=True)

    X_raw, impute_values = _design_matrix(stacked, feature_cols)
    y = stacked["runs"].astype(float).to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    return RunsModel(
        model=model,
        scaler=scaler,
        feature_cols=tuple(feature_cols),
        target_mean=float(np.mean(y)),
        target_std=float(np.std(y)),
        train_n=len(y),
        impute_values=impute_values,
    )


def save_runs_model(model: RunsModel, path: Path | str) -> None:
    """Persist a :class:`RunsModel` for fast reuse (e.g. hourly live refresh)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_runs_model(path: Path | str) -> RunsModel:
    """Load ``RunsModel`` from :func:`save_runs_model`."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def predict_side_runs(
    runs_model: RunsModel,
    stacked: pd.DataFrame,
) -> np.ndarray:
    """Run the model on a stacked DataFrame and return predicted runs."""
    X_raw, _ = _design_matrix(
        stacked, runs_model.feature_cols, impute=runs_model.impute_values,
    )
    X = runs_model.scaler.transform(X_raw)
    return runs_model.model.predict(X)


# Home-field advantage bonus, in runs, added to the home team's predicted
# runs before downstream Pythag conversion. Historical HFA in MLB has
# drifted lower over the past decade (juicier balls, universal DH,
# pitch clock), but home WIN rate remains ~52-54% — meaning HFA partly
# shows up as "clutch / batting last" rather than aggregate run
# differential. A modest constant bonus closes the gap that Pythag
# alone can't capture from the runs model's symmetric output.
DEFAULT_HFA_RUNS_BONUS: float = 0.27


def predict_runs(
    runs_model: RunsModel,
    games: pd.DataFrame,
    *,
    home_field_advantage_runs: float = DEFAULT_HFA_RUNS_BONUS,
) -> pd.DataFrame:
    """Per-game home/away run projections.

    Returns ``games`` with two new columns: ``home_runs_pred``,
    ``away_runs_pred``.

    ``home_field_advantage_runs`` is added to the home team's predicted
    runs AFTER the Ridge regression produces its symmetric per-side
    forecast. Park, defense, and matchup adjustments are already baked
    into the features, so this bonus is the residual HFA (batting last,
    travel rest, familiarity, etc.) that the symmetric model can't see.
    Pass ``0.0`` to disable.
    """
    stacked = stack_sides(games)
    stacked["runs_pred"] = predict_side_runs(runs_model, stacked)
    pivoted = (
        stacked.pivot(index="game_id", columns="side", values="runs_pred")
        .rename(columns={"home": "home_runs_pred", "away": "away_runs_pred"})
        .reset_index()
    )
    if home_field_advantage_runs != 0.0:
        pivoted["home_runs_pred"] = (
            pivoted["home_runs_pred"] + home_field_advantage_runs
        )
    return games.merge(pivoted, on="game_id", how="left")


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def coefficient_table(runs_model: RunsModel) -> pd.DataFrame:
    """Inspect the trained coefficients (standardized scale)."""
    return (
        pd.DataFrame({
            "feature": runs_model.feature_cols,
            "coef_std": runs_model.model.coef_,
            "abs_coef": np.abs(runs_model.model.coef_),
        })
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
        .drop(columns="abs_coef")
    )
