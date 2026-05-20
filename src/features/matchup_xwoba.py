"""Matchup-based xwOBA adjustment.

The architecture (replacing the older per-hitter event-count formula in
``handedness.personalized_park_factor``):

1. **Blend the contact profile** for each pitcher × hitter pairing:
   ``blend_t = 0.6 * pitcher_t + 0.4 * hitter_t``  for ``t in (GB, FB, LD, PU)``.
   Pitcher gets the heavier weight because the pitcher controls launch
   angle more than the hitter does (per decision #4 in matchup.py).

2. **Park factor by flight type.** Each flight type maps to the venue's
   Savant park factor for the dominant hit outcome of that flight type,
   keyed by hitter handedness:

       GB -> 1.0 (park-neutral — ground balls are mostly singles where
                  park geometry barely matters)
       LD -> mean(index_2b, index_3b) (line drives are extra-base hits)
       FB -> index_hr (non-HR FBs are mostly outs, so HR factor dominates)
       PU -> 1.0 (almost always outs)

   The matchup park multiplier is a wOBA-weighted average over the four
   flight types using league-average wOBA on each BIP type:

   .. math::
       PF = \\frac{\\sum_t \\text{blend}_t \\cdot w^{wOBA}_t \\cdot pf_t}
                  {\\sum_t \\text{blend}_t \\cdot w^{wOBA}_t}

3. **Defense factor by flight type.** Infield OAA suppresses GB-share
   xwOBA; outfield OAA suppresses LD+FB-share xwOBA. The OAA values are
   summed across the OPPOSING fielders by position (DH excluded), with
   per-handedness OAA splits applied so RHH facing a RHB-strong defender
   gets the RHB OAA component.

   .. math::
       D = 1 - k \\cdot \\big[
                  IF_{OAA} \\cdot \\text{blend}_{GB}
                + OF_{OAA} \\cdot (\\text{blend}_{FB} + \\text{blend}_{LD})
                \\big]

   ``k`` is calibrated so that under a league-average flight mix (GB%
   ≈ 0.43, FB% ≈ 0.33, LD% ≈ 0.20, PU% ≈ 0.04) the all-flights effect
   recovers ``matchup.apply_defense_oaa``'s ``oaa_runs_per_unit=0.0011``
   total. That makes the two flavors comparable instead of double-counting.

4. **Apply** to the hitter's park-neutral xwOBA:

       xwOBA_matchup_adj = xwOBA × PF × D

   Then PA-weight across the lineup.

The module is pure functions — no DataFrame I/O — so it can be unit-tested
and called from either ``build_features.py`` (historical) or a live
prediction path that wires today's actual lineup + projected positions in.
"""

from __future__ import annotations

from typing import Iterable

# wOBA event weights (same as handedness.WOBA_EVENT_WEIGHTS, repeated
# here for self-containment).
WOBA_BB_TYPE_VALUES: dict[str, float] = {
    # League-average wOBA conditional on the BIP type.
    # Sourced from public FanGraphs / Savant league splits; stable
    # year-to-year within a few thousandths.
    "GB": 0.210,
    "FB": 0.340,
    "LD": 0.685,
    "PU": 0.010,
}

DEFAULT_PITCHER_BLEND_WEIGHT: float = 0.60

# Defense scaling — see derivation in the module docstring.
# League-average mix gives 0.43*IF + 0.53*OF (FB+LD) = 0.96 of OAA exposure
# per BIP. We want the all-effect to equal oaa_runs_per_unit = 0.0011 for
# total team OAA, so k * 0.96 * total_oaa = 0.0011 * total_oaa
# => k ≈ 0.00115 per OAA unit per BIP-share. Tightened to the same
# value used elsewhere for clean back-comparison.
DEFAULT_OAA_PER_FLIGHT_K: float = 0.00115

# League-average event-outcome distribution conditional on BIP flight type.
# Sourced from public Statcast leaderboards (rounded to two decimals).
# Used to translate per-event park factors -> per-flight park factors.
# Probabilities are P(event | flight), so they don't sum to 1 — the
# residual is "out" which has 0 wOBA weight and so drops out of the
# weighted average.
EVENT_PROBS_BY_FLIGHT: dict[str, dict[str, float]] = {
    "GB": {"1B": 0.23, "2B": 0.01, "3B": 0.00, "HR": 0.00},
    "LD": {"1B": 0.50, "2B": 0.15, "3B": 0.02, "HR": 0.03},
    "FB": {"1B": 0.02, "2B": 0.03, "3B": 0.01, "HR": 0.09},
    "PU": {"1B": 0.00, "2B": 0.00, "3B": 0.00, "HR": 0.00},
}

# wOBA event weights — used to weight the events under each flight type
# when collapsing the per-event park factors into a per-flight factor.
_WOBA_WEIGHTS: dict[str, float] = {
    "1B": 0.880, "2B": 1.270, "3B": 1.620, "HR": 2.100,
}


def blend_contact_profile(
    pitcher_gb: float, pitcher_fb: float, pitcher_ld: float, pitcher_pu: float,
    hitter_gb: float, hitter_fb: float, hitter_ld: float, hitter_pu: float,
    pitcher_weight: float = DEFAULT_PITCHER_BLEND_WEIGHT,
) -> tuple[float, float, float, float]:
    """Per decision #4: 60/40 weighting favoring the pitcher.

    Returns ``(blend_gb, blend_fb, blend_ld, blend_pu)``, re-normalized to
    sum to 1.0 (the inputs are sometimes NaN-filled or slightly off).
    """
    pw = float(pitcher_weight)
    hw = 1.0 - pw
    raw = (
        pw * (pitcher_gb or 0.0) + hw * (hitter_gb or 0.0),
        pw * (pitcher_fb or 0.0) + hw * (hitter_fb or 0.0),
        pw * (pitcher_ld or 0.0) + hw * (hitter_ld or 0.0),
        pw * (pitcher_pu or 0.0) + hw * (hitter_pu or 0.0),
    )
    s = sum(raw)
    if s <= 0:
        return (0.43, 0.33, 0.20, 0.04)  # league-avg fallback
    return tuple(x / s for x in raw)  # type: ignore[return-value]


def flight_park_factors(hit_type_pfs: dict[str, float] | None
                         ) -> dict[str, float]:
    """Map a venue's per-hit-type park factors to per-flight park factors,
    using a wOBA-weighted average over each flight's event distribution.

    For each flight type:

    .. math::
        pf_{\\text{flight}} = \\frac{\\sum_e p(e|\\text{flight}) \\cdot w_e \\cdot pf_e}
                                     {\\sum_e p(e|\\text{flight}) \\cdot w_e}

    Per the user's instruction, GB stays park-neutral (the geometry doesn't
    matter for grounders). LD and FB get computed from
    ``EVENT_PROBS_BY_FLIGHT``, which assigns most LD value to 1B+2B (not
    just 2B/3B — a naive 2B/3B average over-weights the rare triples and
    gives Coors LD-pf of ~1.7 when the actual is closer to 1.2).
    """
    if hit_type_pfs is None:
        return {"GB": 1.0, "FB": 1.0, "LD": 1.0, "PU": 1.0}

    def _collapse(flight: str) -> float:
        probs = EVENT_PROBS_BY_FLIGHT[flight]
        num = den = 0.0
        for event, p in probs.items():
            w = _WOBA_WEIGHTS[event]
            pf = hit_type_pfs.get(event, 1.0)
            num += p * w * pf
            den += p * w
        return num / den if den > 0 else 1.0

    return {
        "GB": 1.0,            # user override: GBs are park-neutral
        "FB": _collapse("FB"),
        "LD": _collapse("LD"),
        "PU": 1.0,
    }


def matchup_park_factor(
    blend_gb: float, blend_fb: float, blend_ld: float, blend_pu: float,
    flight_pfs: dict[str, float],
) -> float:
    """wOBA-weighted park multiplier across the blended contact distribution."""
    num = (
        blend_gb * WOBA_BB_TYPE_VALUES["GB"] * flight_pfs["GB"]
      + blend_fb * WOBA_BB_TYPE_VALUES["FB"] * flight_pfs["FB"]
      + blend_ld * WOBA_BB_TYPE_VALUES["LD"] * flight_pfs["LD"]
      + blend_pu * WOBA_BB_TYPE_VALUES["PU"] * flight_pfs["PU"]
    )
    den = (
        blend_gb * WOBA_BB_TYPE_VALUES["GB"]
      + blend_fb * WOBA_BB_TYPE_VALUES["FB"]
      + blend_ld * WOBA_BB_TYPE_VALUES["LD"]
      + blend_pu * WOBA_BB_TYPE_VALUES["PU"]
    )
    return num / den if den > 0 else 1.0


def matchup_defense_factor(
    blend_gb: float, blend_fb: float, blend_ld: float,
    infield_oaa: float, outfield_oaa: float,
    k: float = DEFAULT_OAA_PER_FLIGHT_K,
) -> float:
    """Per-flight defense multiplier.

    Good infield defense suppresses GB-share xwOBA; good outfield defense
    suppresses (FB + LD) share. PU-share is ignored (almost always outs).
    """
    suppression = (
        k * infield_oaa  * blend_gb
      + k * outfield_oaa * (blend_fb + blend_ld)
    )
    return 1.0 - suppression


def matchup_xwoba_adjustment(
    hitter_xwoba: float,
    pitcher_flight: tuple[float, float, float, float],
    hitter_flight: tuple[float, float, float, float],
    hit_type_pfs: dict[str, float] | None,
    infield_oaa: float,
    outfield_oaa: float,
    *,
    pitcher_weight: float = DEFAULT_PITCHER_BLEND_WEIGHT,
    k: float = DEFAULT_OAA_PER_FLIGHT_K,
) -> tuple[float, float, float, tuple[float, float, float, float]]:
    """End-to-end: returns (adjusted_xwoba, park_factor, defense_factor, blend).

    Steps 1+2a+2b chained — convenient for ``build_features``. Also
    returns the intermediate quantities for diagnostics.
    """
    blend = blend_contact_profile(
        *pitcher_flight, *hitter_flight, pitcher_weight=pitcher_weight,
    )
    flight_pfs = flight_park_factors(hit_type_pfs)
    pf = matchup_park_factor(*blend, flight_pfs)
    d  = matchup_defense_factor(blend[0], blend[1], blend[2],
                                 infield_oaa, outfield_oaa, k=k)
    return hitter_xwoba * pf * d, pf, d, blend
