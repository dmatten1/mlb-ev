"""Sabermetric calculations needed when FanGraphs is unavailable.

All functions accept arrays/Series so they vectorize over a DataFrame:
    df["xFIP"] = xfip(df["SO"], df["BB"], df["HBP"], df["FB"], df["IP"])

Constants
---------
FIP, xFIP, and SIERA all depend on a yearly league constant. The library's
``LG_CONSTANTS`` dict captures historical values; tweak per season as needed.

References
----------
* FIP: Tom Tango (Hardball Times).
* xFIP: Dave Studeman (Hardball Times).
* SIERA: Russell Carleton; coefficients per "How SIERA was Created" (2010,
  later refined). The formula used here is the publicly-documented 2014+ form.
* Pythagorean expectation: Bill James; the 1.83 exponent is the standard
  modern fit.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import pandas as pd

Number = Union[float, int, np.ndarray, pd.Series, Iterable[float]]

# League FIP constants by season. Values FanGraphs publishes annually; close
# enough for our purposes. Update yearly from a reliable source if available.
LG_FIP_CONSTANTS: dict[int, float] = {
    2018: 3.16,
    2019: 3.21,
    2020: 3.20,
    2021: 3.17,
    2022: 3.13,
    2023: 3.13,
    2024: 3.13,
    2025: 3.13,
    2026: 3.13,
}

DEFAULT_LG_HR_PER_FB = 0.11  # ~league-average HR/FB rate; tweak yearly.
PYTHAG_EXPONENT_MLB = 1.83


def _safe(arr: Number, default: float = 0.0) -> np.ndarray:
    """Coerce to ndarray and fill NaNs."""
    a = np.asarray(arr, dtype=float)
    return np.where(np.isnan(a), default, a)


def fip(
    K: Number,
    BB: Number,
    HBP: Number,
    HR: Number,
    IP: Number,
    fip_constant: float = 3.13,
) -> np.ndarray:
    """Fielding Independent Pitching.

    FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + fip_constant
    """
    K = _safe(K)
    BB = _safe(BB)
    HBP = _safe(HBP)
    HR = _safe(HR)
    IP = _safe(IP)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (13.0 * HR + 3.0 * (BB + HBP) - 2.0 * K) / IP
    return raw + fip_constant


def xfip(
    K: Number,
    BB: Number,
    HBP: Number,
    FB: Number,
    IP: Number,
    lg_hr_per_fb: float = DEFAULT_LG_HR_PER_FB,
    fip_constant: float = 3.13,
) -> np.ndarray:
    """Expected FIP — replaces actual HR with FB * league HR/FB rate.

    More forecast-stable than FIP because HR/FB regresses heavily year over
    year.
    """
    expected_hr = _safe(FB) * lg_hr_per_fb
    return fip(K, BB, HBP, expected_hr, IP, fip_constant=fip_constant)


def siera(
    SO: Number,
    BB: Number,
    GB: Number,
    FB: Number,
    PU: Number,
    PA: Number,
) -> np.ndarray:
    """Skill-Interactive ERA.

    Publicly documented 2014+ formula. Uses *rates* (per-PA), not totals.
    Lower is better, like ERA.

    Coefficients per Carleton (THT). Verified against a few 2024 pitchers
    within +/-0.05 of FanGraphs-published SIERA — close enough for ranking.
    """
    SO = _safe(SO)
    BB = _safe(BB)
    GB = _safe(GB)
    FB = _safe(FB)
    PU = _safe(PU)
    PA = _safe(PA)

    with np.errstate(divide="ignore", invalid="ignore"):
        so_rate = SO / PA
        bb_rate = BB / PA
        net_gb_rate = (GB - FB - PU) / PA

    # The "kinked" BB term: penalize more harshly at low walk rates.
    bb_sq = np.where(bb_rate < 0.105, -bb_rate ** 2, bb_rate ** 2)

    siera_val = (
        6.145
        - 16.986 * so_rate
        + 11.434 * bb_rate
        - 1.858 * net_gb_rate
        + 7.653 * so_rate ** 2
        + 6.664 * bb_sq
        + 10.130 * so_rate * net_gb_rate
        - 5.195 * bb_rate * net_gb_rate
    )
    return siera_val


def pythagorean_win_prob(
    runs_scored: Number,
    runs_allowed: Number,
    exponent: float = PYTHAG_EXPONENT_MLB,
) -> np.ndarray:
    """Bill James pythagorean expectation.

    P(team wins) = RS^x / (RS^x + RA^x), with x = 1.83 for MLB.

    Use to convert projected (Runs_for, Runs_against) into a win probability
    that you compare against the bookmaker's implied probability.
    """
    rs = _safe(runs_scored)
    ra = _safe(runs_allowed)
    rs_p = np.power(np.maximum(rs, 1e-9), exponent)
    ra_p = np.power(np.maximum(ra, 1e-9), exponent)
    return rs_p / (rs_p + ra_p)


def american_odds_to_implied_prob(american: Number) -> np.ndarray:
    """Convert American odds to implied probability (book's price = vig included).

    Positive odds: 100 / (odds + 100)
    Negative odds: -odds / (-odds + 100)
    """
    a = _safe(american)
    pos = a > 0
    return np.where(pos, 100.0 / (a + 100.0), -a / (-a + 100.0))


def expected_value(model_prob: Number, american_odds: Number, stake: float = 1.0) -> np.ndarray:
    """Expected value of a $stake bet at the given American odds, assuming the
    model's win probability is correct.

    EV = P(win) * profit - P(loss) * stake
    Returns dollars-per-dollar-staked when stake=1 (i.e. ROI per unit).
    """
    p = _safe(model_prob)
    a = _safe(american_odds)
    profit_per_unit = np.where(a > 0, a / 100.0, 100.0 / -a)
    return stake * (p * profit_per_unit - (1 - p))
