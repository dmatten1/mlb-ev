"""Matchup-scoring data model + composable scoring primitives.

Bottom-up matchup engine. Given two lineups, two starters, two bullpens,
two defenses, and a park, project per-team runs and convert to win prob::

    home_runs, away_runs = score_matchup(MatchupInput(...))
    p_home = pythagorean_win_prob(home_runs, away_runs)

Architecture decisions (locked in via design checkpoint)
--------------------------------------------------------
1. **Lineup weighting** — PA share by batting order (``LINEUP_PA_WEIGHTS``),
   normalized to sum to 1.0. Pinch hitters / bench bats are excluded; the
   starting nine carry all of the team's expected PAs.
2. **SP / bullpen weighting** — starter weight = ``clamp(expected_ip / 9,
   0.25, 0.85)``, bullpen takes the remainder (equal-weighted across
   listed RPs).
3. **Defense (OAA)** — applied as a scalar multiplier on contact-allowed
   run contribution. Higher OAA → more would-be hits become outs → lower
   projected runs against. *v2:* replace scalar with a learned interaction
   term once we have a calibrated MVP and labeled training data.
4. **Park & weather** — weather is intentionally omitted in MVP. Park
   factor is applied ONLY to the *air-ball share* of projected runs.
   Pipeline:

   a. Blend pitcher and hitter GB/FB/LD with a 60/40 pitcher-favored mix
      (``blend_contact_profile``).
   b. Weight each contact type by league wOBA on contact
      (GB .210, FB .340, LD .685).
   c. ``air_share = (FB·wFB + LD·wLD) / total``; ``ground_share = 1 - air_share``.
   d. ``final = baseline · (ground_share + air_share · park_run_factor)``.

5. **Projected lineup** — when MLB hasn't posted today's lineup yet, fall
   back to the team's last-7-day modal lineup, *platoon-split* by the
   opposing starter's handedness (LHP vs RHP). Resolved upstream in
   ``src/features/lineup_projection.py``.

Lookahead-bias guard
--------------------
Every feature passed in must be derived from data available *before*
first pitch of the game being scored. The data pipeline timestamps every
snapshot in UTC so this is enforceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants (locked-in design parameters)
# ---------------------------------------------------------------------------

# Raw PA share by batting order (1-9), pre-normalization. Approximate
# league averages for a 9-inning game. Per decision #1 we normalize these
# to sum to 1.0 so pinch hitters / bench bats carry zero weight.
_RAW_LINEUP_PA_SHARES: tuple[float, ...] = (
    0.122, 0.116, 0.111, 0.107, 0.103, 0.099, 0.095, 0.091, 0.088,
)
_TOTAL_RAW = sum(_RAW_LINEUP_PA_SHARES)
LINEUP_PA_WEIGHTS: tuple[float, ...] = tuple(
    w / _TOTAL_RAW for w in _RAW_LINEUP_PA_SHARES
)
assert abs(sum(LINEUP_PA_WEIGHTS) - 1.0) < 1e-9, "LINEUP_PA_WEIGHTS must sum to 1.0"

# Decision #2 — SP/bullpen blend bounds.
SP_WEIGHT_MIN: float = 0.25
SP_WEIGHT_MAX: float = 0.85
LEAGUE_AVG_IP_PER_START: float = 5.5  # fallback when starter.expected_ip missing

# Decision #4 — league-average wOBA by contact type. Used to convert a
# blended contact profile into expected-run shares. Update annually.
LG_WOBA_GB: float = 0.210
LG_WOBA_FB: float = 0.340
LG_WOBA_LD: float = 0.685

# Pitcher vs. hitter weighting in the contact-profile blend.
PITCHER_BATTED_BALL_WEIGHT: float = 0.6
HITTER_BATTED_BALL_WEIGHT: float = 1.0 - PITCHER_BATTED_BALL_WEIGHT


# ---------------------------------------------------------------------------
# Profile dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HitterProfile:
    """One hitter's offense-side feature vector. Computed from Statcast + BR.

    All metrics are point-in-time (as of the snapshot date). NaN / None means
    insufficient sample; downstream code falls back to a positional / league
    average rather than treating the player as 0.
    """
    player_id: int                 # MLBAM key
    name: str
    bats: str                      # "L" / "R" / "S"
    wrc_plus: Optional[float] = None
    barrel_pct: Optional[float] = None
    pullair_pct: Optional[float] = None
    xwoba: Optional[float] = None
    iso: Optional[float] = None
    gb_pct: Optional[float] = None   # for contact-profile blend
    fb_pct: Optional[float] = None
    ld_pct: Optional[float] = None
    pitch_run_values: dict[str, float] = field(default_factory=dict)


@dataclass
class PitcherProfile:
    """One pitcher's pitching-side feature vector."""
    player_id: int
    name: str
    throws: str                    # "L" / "R"
    role: str                      # "SP" or "RP"
    siera: Optional[float] = None
    xfip: Optional[float] = None
    barrel_pct_allowed: Optional[float] = None
    gb_pct: Optional[float] = None
    fb_pct: Optional[float] = None
    ld_pct: Optional[float] = None
    expected_ip_per_start: Optional[float] = None    # blends SP vs bullpen
    pitch_mix: dict[str, float] = field(default_factory=dict)
    pitch_run_values: dict[str, float] = field(default_factory=dict)


@dataclass
class TeamDefense:
    """Defense profile, modulates batted-ball damage as a scalar (decision #3)."""
    team: str
    infield_oaa: float = 0.0
    outfield_oaa: float = 0.0


@dataclass
class GameContext:
    """Park + rest. Weather is intentionally omitted (decision #4)."""
    venue: str
    park_run_factor: float = 1.0          # 1.0 = neutral
    home_rest_days: Optional[int] = None
    away_rest_days: Optional[int] = None


@dataclass
class MatchupInput:
    """Everything needed to score one MLB game."""
    home_lineup: list[HitterProfile]      # length 9, in batting order
    away_lineup: list[HitterProfile]
    home_starter: PitcherProfile
    away_starter: PitcherProfile
    home_bullpen: list[PitcherProfile]    # remaining relievers, weighted equally
    away_bullpen: list[PitcherProfile]
    home_defense: TeamDefense
    away_defense: TeamDefense
    context: GameContext


# ---------------------------------------------------------------------------
# Composite helpers — pure math, fully implemented
# ---------------------------------------------------------------------------

def _is_missing(v) -> bool:
    """True if v is None or NaN."""
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return False


def lineup_composite(lineup: list[HitterProfile], metric: str) -> float:
    """Weighted average of one HitterProfile field using ``LINEUP_PA_WEIGHTS``.

    Missing values (None/NaN) are skipped and the remaining weights are
    re-normalized over the present positions. Returns NaN if every hitter
    is missing the requested metric.

    Example::

        wrc_plus_team = lineup_composite(home_lineup, "wrc_plus")
    """
    if len(lineup) != len(LINEUP_PA_WEIGHTS):
        raise ValueError(
            f"lineup must have exactly {len(LINEUP_PA_WEIGHTS)} hitters, "
            f"got {len(lineup)}"
        )
    num = 0.0
    denom = 0.0
    for hitter, weight in zip(lineup, LINEUP_PA_WEIGHTS):
        val = getattr(hitter, metric)
        if _is_missing(val):
            continue
        num += float(val) * weight
        denom += weight
    return num / denom if denom > 0 else float("nan")


def starter_weight(expected_ip_per_start: Optional[float]) -> float:
    """SP weight = clamp(expected_ip / 9, SP_WEIGHT_MIN, SP_WEIGHT_MAX).

    Falls back to LEAGUE_AVG_IP_PER_START when input is missing.
    """
    ip = expected_ip_per_start
    if _is_missing(ip):
        ip = LEAGUE_AVG_IP_PER_START
    raw = float(ip) / 9.0
    return max(SP_WEIGHT_MIN, min(SP_WEIGHT_MAX, raw))


def pitcher_staff_composite(
    starter: PitcherProfile,
    bullpen: list[PitcherProfile],
    metric: str,
) -> float:
    """Blend starter (weighted by expected IP/9, clamped) with bullpen
    (equal-weighted) for a single ``PitcherProfile`` field.

    If the starter value is missing, returns the bullpen mean (or NaN if
    the bullpen is also empty/missing).
    """
    sp_w = starter_weight(starter.expected_ip_per_start)
    bp_w = 1.0 - sp_w

    sp_val = getattr(starter, metric)
    bp_vals = [
        getattr(p, metric) for p in bullpen
    ]
    bp_vals = [float(v) for v in bp_vals if not _is_missing(v)]

    if _is_missing(sp_val):
        return float(np.mean(bp_vals)) if bp_vals else float("nan")
    if not bp_vals:
        return float(sp_val)
    return sp_w * float(sp_val) + bp_w * (sum(bp_vals) / len(bp_vals))


def blend_contact_profile(
    pitcher_gb: float, pitcher_fb: float, pitcher_ld: float,
    hitter_gb: float, hitter_fb: float, hitter_ld: float,
    pitcher_weight: float = PITCHER_BATTED_BALL_WEIGHT,
) -> tuple[float, float, float]:
    """Blend pitcher and hitter GB/FB/LD with the agreed pitcher-favored mix.

    Returns ``(gb, fb, ld)``. With both sides summing to 1.0, the output
    sums to 1.0; if they don't, the output preserves the input scale.
    """
    hw = 1.0 - pitcher_weight
    gb = pitcher_weight * pitcher_gb + hw * hitter_gb
    fb = pitcher_weight * pitcher_fb + hw * hitter_fb
    ld = pitcher_weight * pitcher_ld + hw * hitter_ld
    return gb, fb, ld


def air_share_from_contact(
    gb: float, fb: float, ld: float,
) -> tuple[float, float]:
    """Convert a contact distribution into ``(ground_share, air_share)`` of
    *expected runs* via league-average wOBA on each contact type.

    Returned shares sum to 1.0. Degenerate input (zero total) splits evenly.
    """
    gb_run = gb * LG_WOBA_GB
    fb_run = fb * LG_WOBA_FB
    ld_run = ld * LG_WOBA_LD
    total = gb_run + fb_run + ld_run
    if total <= 0:
        return 0.5, 0.5
    return gb_run / total, (fb_run + ld_run) / total


def apply_park_adjustment(
    baseline_runs: float,
    gb: float, fb: float, ld: float,
    park_run_factor: float,
) -> float:
    """Split baseline runs into ground vs air via the blended contact
    profile, then multiply ONLY the air share by ``park_run_factor``.

    park_run_factor > 1.0 = hitter-friendly (Coors, ~1.20);
    < 1.0 = pitcher-friendly (Petco, ~0.95);
    1.0 = neutral.
    """
    ground_share, air_share = air_share_from_contact(gb, fb, ld)
    return baseline_runs * (ground_share + air_share * park_run_factor)


def apply_defense_oaa(
    runs_against: float,
    defense: TeamDefense,
    oaa_runs_per_unit: float = 0.0011,
) -> float:
    """Scalar OAA adjustment (decision #3).

    Sizing: 1 OAA ≈ 0.8 runs prevented over a full season (Statcast OAA
    × run-expectancy of an out). Over a 162-game season at ~4.5 RA/game,
    that's ``0.8 / 162 / 4.5 ≈ 0.0011`` as a per-game multiplicative
    shrinkage per OAA unit. So:

    * +25 team OAA (elite) → ~2.75% reduction in opposing runs (~0.12 R/game).
    * -20 team OAA (Coors-bad) → ~2.2% increase.

    ``infield_oaa`` and ``outfield_oaa`` are summed for now. A future
    version can route ground balls through infield OAA only and air balls
    through outfield OAA only — at that point the function would need the
    contact distribution as input.

    ``oaa_runs_per_unit`` is the calibration knob; tune against the MVP
    once a labeled training set exists.
    """
    total_oaa = defense.infield_oaa + defense.outfield_oaa
    return runs_against * (1.0 - oaa_runs_per_unit * total_oaa)


# ---------------------------------------------------------------------------
# Top-level scoring — still a stub pending baseline-runs model selection
# ---------------------------------------------------------------------------

def project_runs(
    offense: list[HitterProfile],
    opposing_starter: PitcherProfile,
    opposing_bullpen: list[PitcherProfile],
    opposing_defense: TeamDefense,
    context: GameContext,
) -> float:
    """End-to-end: from raw profiles to a projected runs total.

    Pipeline (per locked-in decisions):

    1. Lineup-side composites via ``lineup_composite`` (decision #1).
    2. Staff-side composites via ``pitcher_staff_composite`` (decision #2).
    3. Combine composites into a "neutral-context expected runs" baseline.
       The combiner is the one piece still TBD — likely a linear regression
       fit on Statcast/BR features against actual runs scored, calibrated
       once we have a history window of saved snapshots.
    4. ``apply_defense_oaa(...)`` — scalar OAA shrinkage (decision #3).
    5. ``apply_park_adjustment(...)`` — air-share-only park factor
       (decision #4).
    """
    raise NotImplementedError(
        "project_runs awaits the baseline-runs combiner (step 3). All other "
        "primitives are implemented and unit-testable."
    )


def score_matchup(m: MatchupInput) -> tuple[float, float]:
    """Returns (home_projected_runs, away_projected_runs)."""
    home_runs = project_runs(
        m.home_lineup, m.away_starter, m.away_bullpen, m.away_defense, m.context
    )
    away_runs = project_runs(
        m.away_lineup, m.home_starter, m.home_bullpen, m.home_defense, m.context
    )
    return home_runs, away_runs
