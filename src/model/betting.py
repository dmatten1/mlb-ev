"""Betting math: odds conversions, EV, and Kelly sizing.

Pure-function utilities for the inference + backtest pipelines. No
external state, no I/O. Most callers will work in batches via the
DataFrame-vectorized helpers at the bottom.

Conventions used throughout:

* **American odds** — sportsbook standard. ``-110`` means "risk 110 to
  win 100"; ``+150`` means "risk 100 to win 150". A favorite has
  negative odds, an underdog positive.
* **Decimal odds** — total return per 1.0 staked. ``2.0`` = even
  money (risk 1 to win 1, return 2 total). Easier for math.
* **Implied probability** — the probability the bookmaker's odds
  correspond to. For decimal odds ``d``, implied = ``1/d``.
* **Vig** (juice / overround) — bookmakers price both sides such that
  implied probabilities sum to >1. The excess is the house margin.
  Typical MLB moneylines run ~4-5% vig.

For positive-EV betting we compare our **model probability** to the
**fair (de-vigged) market probability**. Edge = model_p − fair_p.

Kelly criterion: optimal-growth bet size as a fraction of bankroll.
``f* = (b·p − q) / b`` where ``b`` is net decimal odds (``d − 1``),
``p`` is true win prob, ``q = 1 − p``. Full Kelly maximizes log
growth but has high variance — most practitioners use 1/4 or 1/2
Kelly to dampen drawdowns at minor cost to long-run growth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Defaults kept in sync with :func:`annotate_bets`.
DEFAULT_KELLY_CAP: float | None = 0.01
DEFAULT_KELLY_FRACTION_MULT: float = 0.0625
# When ``risk_ref_kelly`` is unknown (legacy rows), use
# ``cap * fraction_mult * TRACKING_KELLY_REF_FALLBACK_MULT`` (~0.00125 at defaults)
# so typical pre-daily Kellys map near 1u instead of slamming the 0.5 floor.
TRACKING_KELLY_REF_FALLBACK_MULT: float = 2.0


def tracking_kelly_ref_fallback(
    kelly_cap: float | None = None,
    fraction_mult: float | None = None,
) -> float:
    """Denominator Kelly for tracker sizing when a slate-level ref was not stored."""
    cap = DEFAULT_KELLY_CAP if kelly_cap is None else kelly_cap
    cap_f = float(0.01 if cap is None or cap <= 0 else cap)
    fm = float(
        DEFAULT_KELLY_FRACTION_MULT if fraction_mult is None else fraction_mult,
    )
    return cap_f * fm * TRACKING_KELLY_REF_FALLBACK_MULT

# ---------------------------------------------------------------------------
# Odds conversions
# ---------------------------------------------------------------------------

def american_to_decimal(american: float | np.ndarray) -> float | np.ndarray:
    """Convert American odds to decimal odds.

    ``-110`` -> ``1.909...``  (1 + 100/110)
    ``+150`` -> ``2.500``     (1 + 150/100)
    """
    a = np.asarray(american, dtype=float)
    out = np.where(
        a >= 0,
        1.0 + a / 100.0,
        1.0 + 100.0 / np.abs(a),
    )
    # Preserve scalar in / scalar out.
    return float(out) if np.ndim(american) == 0 else out


def decimal_to_implied(decimal_odds: float | np.ndarray) -> float | np.ndarray:
    """Decimal odds -> implied probability. ``1 / d``."""
    d = np.asarray(decimal_odds, dtype=float)
    out = np.where(d > 0, 1.0 / d, np.nan)
    return float(out) if np.ndim(decimal_odds) == 0 else out


def american_to_implied(american: float | np.ndarray) -> float | np.ndarray:
    """One-step American -> implied probability."""
    return decimal_to_implied(american_to_decimal(american))


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------

def remove_vig(
    p_home_implied: float | np.ndarray,
    p_away_implied: float | np.ndarray,
    *,
    method: str = "proportional",
) -> tuple[np.ndarray, np.ndarray]:
    """Strip the bookmaker's margin so the two sides sum to 1.

    Two methods:

    * ``"proportional"`` (default) — divide each side by the sum.
      Simple, widely used, slightly biases toward favorites in
      practice but is the standard MLB-modeling convention.
    * ``"power"`` — solve for the exponent ``k`` such that
      ``p_home^k + p_away^k = 1``. More accurate at extreme prices
      (heavy favorites) where proportional under-corrects, but the
      moneyline range we'll be in (~-300 to +300) makes the
      difference < 1% probability points either way.

    Returns ``(p_home_fair, p_away_fair)`` with the same shape as the
    inputs. Both arrays satisfy ``p_home + p_away ≈ 1``.
    """
    p_h = np.asarray(p_home_implied, dtype=float)
    p_a = np.asarray(p_away_implied, dtype=float)
    if method == "proportional":
        total = p_h + p_a
        with np.errstate(divide="ignore", invalid="ignore"):
            fair_h = np.where(total > 0, p_h / total, np.nan)
            fair_a = np.where(total > 0, p_a / total, np.nan)
    elif method == "power":
        # Power method: find k such that p_h^k + p_a^k = 1 (per-game).
        # Vectorized via bisection.
        def _solve_k(ph: float, pa: float) -> float:
            if not (np.isfinite(ph) and np.isfinite(pa) and ph > 0 and pa > 0):
                return np.nan
            lo, hi = 0.5, 2.0
            for _ in range(64):
                mid = 0.5 * (lo + hi)
                s = ph ** mid + pa ** mid
                if s > 1.0:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        ks = np.array([_solve_k(float(ph), float(pa))
                        for ph, pa in zip(p_h.ravel(), p_a.ravel())])
        fair_h = (p_h.ravel() ** ks).reshape(p_h.shape)
        fair_a = (p_a.ravel() ** ks).reshape(p_a.shape)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return fair_h, fair_a


def vig_pct(
    p_home_implied: float | np.ndarray,
    p_away_implied: float | np.ndarray,
) -> float | np.ndarray:
    """Bookmaker margin as a fraction. ``(p_h + p_a) - 1.0``."""
    out = np.asarray(p_home_implied, dtype=float) + np.asarray(p_away_implied, dtype=float) - 1.0
    return float(out) if np.ndim(p_home_implied) == 0 else out


# ---------------------------------------------------------------------------
# EV and Kelly
# ---------------------------------------------------------------------------

def expected_value(
    model_p: float | np.ndarray,
    decimal_odds: float | np.ndarray,
) -> float | np.ndarray:
    """Expected profit per 1 unit staked.

    Formula: ``model_p * (d − 1) − (1 − model_p)``
           = ``model_p * d − 1``.

    Positive => model thinks this side is +EV at this price.
    EV = 0.05 => 5 cents of expected profit per dollar bet.
    """
    p = np.asarray(model_p, dtype=float)
    d = np.asarray(decimal_odds, dtype=float)
    out = p * d - 1.0
    return float(out) if np.ndim(model_p) == 0 and np.ndim(decimal_odds) == 0 else out


def kelly_fraction(
    model_p: float | np.ndarray,
    decimal_odds: float | np.ndarray,
    *,
    fraction: float = 1.0,
    cap: float | None = 0.10,
) -> float | np.ndarray:
    """Kelly bet size as a fraction of bankroll.

    Full Kelly: ``f* = (b·p − q) / b`` where b = decimal_odds − 1.

    Parameters
    ----------
    model_p
        Our estimate of the true probability of winning.
    decimal_odds
        The decimal odds offered on this side.
    fraction
        Multiplier on full Kelly. ``0.25`` = quarter Kelly. Most
        practitioners use 0.25-0.5 to dampen variance at small cost
        to growth rate. Default 1.0 (full Kelly).
    cap
        Hard maximum on bet-as-fraction-of-bankroll. ``0.10`` means
        never bet more than 10% on a single game even if Kelly says
        to. Guards against model overconfidence + bankroll wipeout.
        Set to ``None`` to disable.

    Returns 0 when the model edge is negative (don't bet).
    """
    p = np.asarray(model_p, dtype=float)
    d = np.asarray(decimal_odds, dtype=float)
    b = d - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_star = np.where(b > 0, (b * p - (1 - p)) / b, 0.0)
    f_star = np.maximum(f_star, 0.0) * float(fraction)
    if cap is not None:
        f_star = np.minimum(f_star, float(cap))
    return float(f_star) if (np.ndim(model_p) == 0 and np.ndim(decimal_odds) == 0) else f_star


def kelly_to_risk_units(
    kelly: float | np.ndarray,
    *,
    ref_kelly: float | np.ndarray | None = None,
    kelly_cap: float | None = DEFAULT_KELLY_CAP,
    anchor_frac: float = 0.5,
    min_units: float = 0.5,
    max_units: float = 2.0,
) -> float | np.ndarray:
    """Map **fraction-of-bankroll Kelly** (pre–daily-cap) to **units at risk**.

    **Preferred:** pass ``ref_kelly`` = the **mean** recommended Kelly on
    the slate (see :func:`annotate_bets`). Then average stake is ~**1.0**
    unit before clipping; stronger/weaker edges spread toward ``max_units``
    / ``min_units`` (default 2.0 / 0.5).

    **Fallback:** if ``ref_kelly`` is omitted, ``ref = anchor_frac *
    kelly_cap`` (legacy). That denominator is much larger than typical
    1/16-Kelly stakes and pushes almost everything to the ``min_units``
    floor — avoid unless you know you want it.

    Tracker P/L uses pre–daily-cap Kelly so relative conviction between
    games is visible; bankroll % still comes from ``recommended_kelly``
    after the slate daily stake cap.
    """
    k = np.asarray(kelly, dtype=float)

    if ref_kelly is not None:
        ref = np.asarray(ref_kelly, dtype=float)
        bad = (ref <= 0) | ~np.isfinite(ref)
        if np.any(bad):
            fb = tracking_kelly_ref_fallback(kelly_cap)
            ref = np.where(bad, fb, ref)
        out = k / ref
    elif kelly_cap is not None and float(kelly_cap) > 0:
        ref = float(kelly_cap) * float(anchor_frac)
        out = k / ref
    else:
        out = np.ones_like(k, dtype=float)

    out = np.clip(out, float(min_units), float(max_units))
    # Return Python float only for true scalar Kelly.
    if k.ndim == 0:
        return float(out)
    return out


# ---------------------------------------------------------------------------
# DataFrame-level batch helpers (the API the inference/backtest layers use)
# ---------------------------------------------------------------------------

def annotate_bets(
    df: pd.DataFrame,
    *,
    model_p_col: str = "p_home",
    home_price_col: str = "home_price_american",
    away_price_col: str = "away_price_american",
    ev_threshold: float = 0.0,
    max_edge: float | None = 0.07,
    kelly_fraction_mult: float = 0.0625,
    kelly_cap: float | None = DEFAULT_KELLY_CAP,
    daily_stake_cap: float | None = 0.05,
    vig_method: str = "proportional",
) -> pd.DataFrame:
    """Add columns: decimal odds, implied probs, fair probs (de-vigged),
    per-side EV, per-side Kelly fraction, recommended bet side.

    ``recommended`` is one of ``"home"``, ``"away"``, or ``""`` (no bet)
    based on which side has higher EV, gated by ``ev_threshold`` and
    capped by ``max_edge``.

    Parameters
    ----------
    ev_threshold
        Minimum EV (model expected profit per $1) to recommend a bet.
        Default 0.0 — any non-negative EV is considered. Small-edge bets
        are where the model actually adds value over the market, so we
        don't filter them out.
    max_edge
        Maximum |edge| (model_p − fair_p) on the recommended side. Bets
        beyond this cap are skipped — empirically, large model-market
        disagreements are model errors, not edges. The market knows
        things our features don't (injuries, recent form, weather, ump
        assignment). Default 0.07 = 7pp. Set ``None`` to disable.
    kelly_fraction_mult
        Multiplier on full Kelly. Default **0.0625 (1/16 Kelly)** — MLB
        moneylines are very high-variance even on +EV bets, so we trade
        long-run growth for much smaller drawdowns. Per-bet sizing is
        ~1/4 of what quarter-Kelly would suggest. Targets ~1% daily
        bankroll exposure on a typical slate.
    kelly_cap
        Hard cap on bet-as-fraction-of-bankroll. Default 0.01 (1%).
        Prevents any one bet from dominating daily variance.
    daily_stake_cap
        Cap on **total** stake across the slate. Default 0.05 (5%). If
        the per-bet kellys sum above this, every recommended stake on
        the slate is scaled down proportionally so the sum equals the
        cap. Set ``None`` to disable. This caps a single day's risk
        even when many small +EV bets pile up.

    Returns a copy of ``df`` with new columns, including ``risk_ref_kelly``
    (slate-mean recommended Kelly = **1 tracking unit** denominator) and
    ``risk_units`` (clamped 0.5–2).
    """
    out = df.copy()
    dh = american_to_decimal(out[home_price_col].to_numpy())
    da = american_to_decimal(out[away_price_col].to_numpy())
    out["home_decimal"] = dh
    out["away_decimal"] = da
    out["home_implied"] = decimal_to_implied(dh)
    out["away_implied"] = decimal_to_implied(da)
    out["vig"] = vig_pct(out["home_implied"].to_numpy(), out["away_implied"].to_numpy())
    fair_h, fair_a = remove_vig(
        out["home_implied"].to_numpy(),
        out["away_implied"].to_numpy(),
        method=vig_method,
    )
    out["home_fair_p"] = fair_h
    out["away_fair_p"] = fair_a

    model_p_home = out[model_p_col].to_numpy()
    model_p_away = 1.0 - model_p_home
    out["edge_home"] = model_p_home - out["home_fair_p"].to_numpy()
    out["edge_away"] = model_p_away - out["away_fair_p"].to_numpy()
    out["ev_home"] = expected_value(model_p_home, dh)
    out["ev_away"] = expected_value(model_p_away, da)
    out["kelly_home"] = kelly_fraction(
        model_p_home, dh, fraction=kelly_fraction_mult, cap=kelly_cap,
    )
    out["kelly_away"] = kelly_fraction(
        model_p_away, da, fraction=kelly_fraction_mult, cap=kelly_cap,
    )

    # Recommended side: whichever has higher EV, gated by threshold + edge cap.
    home_pick = (out["ev_home"] > out["ev_away"]) & (out["ev_home"] > ev_threshold)
    away_pick = (out["ev_away"] > out["ev_home"]) & (out["ev_away"] > ev_threshold)
    if max_edge is not None:
        home_pick &= out["edge_home"].abs() <= max_edge
        away_pick &= out["edge_away"].abs() <= max_edge
    out["recommended"] = np.where(home_pick, "home", np.where(away_pick, "away", ""))
    out["recommended_kelly"] = np.where(
        out["recommended"] == "home", out["kelly_home"],
        np.where(out["recommended"] == "away", out["kelly_away"], 0.0),
    )
    out["recommended_ev"] = np.where(
        out["recommended"] == "home", out["ev_home"],
        np.where(out["recommended"] == "away", out["ev_away"], 0.0),
    )
    # Kelly *before* the slate-wide daily cap — preserves relative conviction
    # between games when the daily stake cap rescales every row equally.
    pre_daily = np.where(
        out["recommended"] == "home", out["kelly_home"],
        np.where(out["recommended"] == "away", out["kelly_away"], 0.0),
    ).astype(float)
    out["recommended_kelly_pre_daily"] = pre_daily

    mask_rec = out["recommended"].isin(["home", "away"]).to_numpy()
    rec_k = pre_daily[mask_rec]
    if rec_k.size > 0:
        ref_slate = float(np.mean(rec_k))
        if not np.isfinite(ref_slate) or ref_slate <= 0:
            ref_slate = tracking_kelly_ref_fallback(kelly_cap, kelly_fraction_mult)
    else:
        ref_slate = tracking_kelly_ref_fallback(kelly_cap, kelly_fraction_mult)

    ru = np.full(len(out), np.nan, dtype=float)
    if mask_rec.any():
        ru[mask_rec] = kelly_to_risk_units(
            pre_daily[mask_rec],
            ref_kelly=ref_slate,
            kelly_cap=kelly_cap,
        )
    out["risk_units"] = ru

    rr = np.full(len(out), np.nan, dtype=float)
    rr[mask_rec] = ref_slate
    out["risk_ref_kelly"] = rr

    # Daily-total cap: rescale all recommended stakes proportionally so
    # the slate's total stake doesn't exceed daily_stake_cap. Keeps the
    # per-bet ranking intact — every bet shrinks by the same multiplier.
    if daily_stake_cap is not None and daily_stake_cap > 0:
        total = float(out["recommended_kelly"].sum())
        if total > daily_stake_cap:
            scale = daily_stake_cap / total
            out["recommended_kelly"] = out["recommended_kelly"] * scale
            out["daily_stake_scaled"] = True
            out["daily_stake_scale_factor"] = scale
        else:
            out["daily_stake_scaled"] = False
            out["daily_stake_scale_factor"] = 1.0

    return out
