"""Probability calibration via Platt scaling (default) or isotonic regression.

The runs regression (``runs_model.py``) outputs runs predictions, which we
turn into a home-win probability via Pythagorean expectation. Empirically
this pipeline produces probabilities that are MISCALIBRATED — the
ranking is right but the absolute numbers are off (mean predicted
``p_home`` ~ 0.498 vs actual home-win rate ~ 0.543, a +5pp uniform gap).

Calibration fixes that without retraining the underlying runs model.

Two methods supported:

* **Platt scaling** (default): a 2-parameter logistic regression mapping
  ``p_raw`` -> ``p_cal``. Equation:

  .. math::
      p_{\\text{cal}} = \\frac{1}{1 + \\exp(A \\cdot p_{\\text{raw}} + B)}

  Fits ``A`` (slope) and ``B`` (intercept) on the held-out calibration
  slice via standard logistic-regression MLE. The intercept absorbs the
  uniform shift; the slope re-tilts if the model is over- or
  underconfident.

* **Isotonic regression**: a non-parametric, monotone-increasing piecewise
  function mapping ``p_raw`` -> ``p_cal``. More flexible than Platt but
  needs more calibration data; can overfit individual buckets.

Both are wrapped in a common ``Calibrator`` class so callers can swap
between them.

Fitting protocol (mandatory):
* Calibrate on a chronological held-out slice the runs model has NOT
  seen — otherwise the calibration parameters are biased.
* For production, after fitting calibration on the held-out slice you
  can re-train the runs model on ALL available data and reuse the
  same calibration. This is the standard trick to avoid throwing away
  training data; works because the calibration shape is roughly stable
  across the training window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["platt", "isotonic"]


@dataclass
class Calibrator:
    """A fitted Platt / isotonic transform on a 1-D probability input.

    Use ``fit_calibration`` to construct; this dataclass just wraps the
    fitted model + method for ``transform`` calls. Pickleable via the
    standard ``sklearn`` model interfaces inside.
    """

    model: LogisticRegression | IsotonicRegression
    method: CalibrationMethod
    train_n: int  # number of (p_raw, y) pairs the calibration was fit on
    train_p_mean: float
    train_y_rate: float
    eps: float = 1e-9

    def transform(self, p_raw: np.ndarray | pd.Series) -> np.ndarray:
        """Map raw probabilities -> calibrated probabilities."""
        p = np.asarray(p_raw, dtype=float)
        # Guard against NaN — caller should pre-filter, but we don't
        # want sklearn to throw on a stray missing value mid-batch.
        nan_mask = np.isnan(p)
        if nan_mask.any():
            p_safe = np.where(nan_mask, 0.5, p)
        else:
            p_safe = p

        p_clipped = np.clip(p_safe, self.eps, 1 - self.eps)
        if self.method == "platt":
            out = self.model.predict_proba(p_clipped.reshape(-1, 1))[:, 1]
        else:  # isotonic
            out = self.model.transform(p_clipped)

        out = np.clip(out, self.eps, 1 - self.eps)
        if nan_mask.any():
            out[nan_mask] = np.nan
        return out


def fit_calibration(
    p_raw: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    method: CalibrationMethod = "platt",
) -> Calibrator:
    """Fit a calibration transform on a held-out (p_raw, y) slice.

    Inputs are aligned 1-D arrays. NaN rows in either are dropped.

    Returns a fitted ``Calibrator`` ready for ``.transform(p_new)``.
    """
    p = np.asarray(p_raw, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = ~(np.isnan(p) | np.isnan(yv))
    p, yv = p[mask], yv[mask].astype(int)
    if len(p) == 0:
        raise ValueError("No valid (p_raw, y) pairs for calibration.")

    if method == "platt":
        # C huge -> effectively unregularized logistic regression on a
        # single feature. Platt's original formulation includes MAP
        # smoothing for tiny samples; with our 500-2400 game slices the
        # plain MLE is fine.
        clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        clf.fit(p.reshape(-1, 1), yv)
        model: LogisticRegression | IsotonicRegression = clf
    elif method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, yv)
        model = iso
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    return Calibrator(
        model=model,
        method=method,
        train_n=len(p),
        train_p_mean=float(p.mean()),
        train_y_rate=float(yv.mean()),
    )


def platt_parameters(cal: Calibrator) -> tuple[float, float]:
    """Return (A, B) for a Platt calibrator in the canonical form
    ``p_cal = 1 / (1 + exp(A * p_raw + B))``.

    sklearn fits ``p_cal = sigmoid(w * p_raw + b)``, where
    ``sigmoid(z) = 1 / (1 + exp(-z))``. To match Platt's form we negate:
    ``A = -w``, ``B = -b``.
    """
    if cal.method != "platt":
        raise ValueError("platt_parameters only valid for method='platt'")
    w = float(cal.model.coef_[0][0])
    b = float(cal.model.intercept_[0])
    return -w, -b
