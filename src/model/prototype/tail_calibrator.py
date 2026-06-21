"""Region-split isotonic calibration for probability tails.

Fits separate isotonic maps for:
  * dogs:   p < dog_cut
  * middle: dog_cut <= p <= fav_cut
  * favorites: p > fav_cut

Games outside a region pass through unchanged for that region's calibrator;
each point is transformed by the calibrator whose region contains it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.model.calibration import Calibrator, fit_calibration


@dataclass
class TailCalibrator:
    """Piecewise isotonic calibration for favorite / middle / dog regions."""

    dog: Calibrator | None
    middle: Calibrator | None
    fav: Calibrator | None
    dog_cut: float = 0.40
    fav_cut: float = 0.60
    eps: float = 1e-9

    def transform(self, p_raw: np.ndarray | pd.Series) -> np.ndarray:
        p = np.asarray(p_raw, dtype=float).copy()
        out = p.copy()
        nan_mask = np.isnan(p)
        for mask, cal in (
            (p < self.dog_cut, self.dog),
            ((p >= self.dog_cut) & (p <= self.fav_cut), self.middle),
            (p > self.fav_cut, self.fav),
        ):
            if cal is None or not mask.any():
                continue
            out[mask] = cal.transform(p[mask])
        out = np.clip(out, self.eps, 1 - self.eps)
        out[nan_mask] = np.nan
        return out


def fit_tail_calibration(
    p_raw: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    dog_cut: float = 0.40,
    fav_cut: float = 0.60,
    min_region_n: int = 40,
) -> TailCalibrator:
    """Fit up to three isotonic calibrators on disjoint p buckets."""
    p = np.asarray(p_raw, dtype=float)
    yv = np.asarray(y, dtype=float)

    def _fit_region(lo: float, hi: float, inclusive_hi: bool) -> Calibrator | None:
        if inclusive_hi:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        m &= ~(np.isnan(p) | np.isnan(yv))
        if m.sum() < min_region_n:
            return None
        return fit_calibration(p[m], yv[m], method="isotonic")

    return TailCalibrator(
        dog=_fit_region(0.0, dog_cut, inclusive_hi=False),
        middle=_fit_region(dog_cut, fav_cut, inclusive_hi=True),
        fav=_fit_region(fav_cut, 1.0, inclusive_hi=True),
        dog_cut=dog_cut,
        fav_cut=fav_cut,
    )


def tail_calibration_table(
    p_raw: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    dog_cut: float = 0.40,
    fav_cut: float = 0.60,
) -> pd.DataFrame:
    """Summarize calibration gap in dog / middle / favorite regions."""
    p = np.asarray(p_raw, dtype=float)
    yv = np.asarray(y, dtype=float)
    regions = [
        ("dog", p < dog_cut),
        ("middle", (p >= dog_cut) & (p <= fav_cut)),
        ("fav", p > fav_cut),
    ]
    rows: list[dict] = []
    for name, mask in regions:
        m = mask & ~(np.isnan(p) | np.isnan(yv))
        if not m.any():
            rows.append({"region": name, "n": 0, "p_mean": np.nan,
                         "actual_rate": np.nan, "gap": np.nan})
            continue
        p_mean = float(p[m].mean())
        actual = float(yv[m].mean())
        rows.append({
            "region": name,
            "n": int(m.sum()),
            "p_mean": round(p_mean, 4),
            "actual_rate": round(actual, 4),
            "gap": round(actual - p_mean, 4),
        })
    return pd.DataFrame(rows)
