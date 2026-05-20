"""Build a per-batter handedness lookup and a venue (home_team) ->
park_factor table keyed by handedness.

Two small lookups, both built from existing local data — no network calls.

* ``build_batter_handedness(statcast)`` — for each MLBAM batter id, take
  the modal ``stand`` they appeared with. Switch hitters resolve to
  whichever side they batted from more often in the data; that's a
  v1 simplification (proper v2 would split by opposing pitcher hand).
* ``build_park_factor_lookup(park_factor_df)`` — pivots the Savant pull
  into a dict keyed by ``(team_abbrev, stand)`` so build_features can do
  fast lookups without re-loading the parquet per game.

Also exposes ``MLBAM_TEAM_ID_TO_ABBREV`` because Savant uses MLBAM team
IDs (``main_team_id``) but Statcast / outcomes data uses abbreviations
(``LAD``, ``NYY``, etc.). One canonical mapping in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# MLBAM team_id -> abbreviation used in Statcast ``home_team`` / ``away_team``.
# Built from a Savant team listing; stable.
MLBAM_TEAM_ID_TO_ABBREV: dict[int, str] = {
    108: "LAA",  # Angels
    109: "ARI",  # Diamondbacks
    110: "BAL",  # Orioles
    111: "BOS",  # Red Sox
    112: "CHC",  # Cubs
    113: "CIN",  # Reds
    114: "CLE",  # Guardians
    115: "COL",  # Rockies
    116: "DET",  # Tigers
    117: "HOU",  # Astros
    118: "KC",   # Royals
    119: "LAD",  # Dodgers
    120: "WSH",  # Nationals
    121: "NYM",  # Mets
    133: "OAK",  # Athletics (Sacramento in 2025+ but team id is stable)
    134: "PIT",  # Pirates
    135: "SD",   # Padres
    136: "SEA",  # Mariners
    137: "SF",   # Giants
    138: "STL",  # Cardinals
    139: "TB",   # Rays
    140: "TEX",  # Rangers
    141: "TOR",  # Blue Jays
    142: "MIN",  # Twins
    143: "PHI",  # Phillies
    144: "ATL",  # Braves
    145: "CWS",  # White Sox
    146: "MIA",  # Marlins
    147: "NYY",  # Yankees
    158: "MIL",  # Brewers
}


def build_batter_handedness(statcast: pd.DataFrame) -> pd.DataFrame:
    """Return ``{batter, stand}`` (one row per MLBAM id), taking the modal
    ``stand`` per batter.

    Switch hitters (some PAs L, some R) resolve to whichever side they used
    more often in the data. Good enough for v1; v2 should split by opposing
    pitcher handedness.
    """
    if "batter" not in statcast.columns or "stand" not in statcast.columns:
        raise ValueError("statcast must have 'batter' and 'stand' columns")
    df = statcast[["batter", "stand"]].dropna()
    counts = df.value_counts().reset_index(name="n")
    # mode = first row per batter after sorting by n desc
    counts = counts.sort_values(["batter", "n"], ascending=[True, False])
    modal = counts.drop_duplicates("batter", keep="first")
    return modal[["batter", "stand"]].reset_index(drop=True)


def build_park_factor_lookup(
    park_factor_df: pd.DataFrame,
    *,
    factor_col: str = "index_xwobacon",
) -> dict[tuple[str, str], float]:
    """``(team_abbrev, stand) -> park_factor`` (as a float, i.e. 1.02 not 102).

    Single-column lookup. Use ``build_hit_type_park_factor_lookup`` if you
    want the full hit-type breakdown.
    """
    out: dict[tuple[str, str], float] = {}
    for _, row in park_factor_df.iterrows():
        team_id = int(row["main_team_id"])
        abbrev = MLBAM_TEAM_ID_TO_ABBREV.get(team_id)
        if abbrev is None:
            continue
        stand = row["bat_side"]
        factor = float(row[factor_col]) / 100.0
        out[(abbrev, stand)] = factor
    return out


def build_hit_type_park_factor_lookup(
    park_factor_df: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, float]]:
    """``(team_abbrev, stand) -> {event_type: park_factor}`` for all hit types.

    Used by ``personalized_park_factor`` to compute a hitter-specific
    park multiplier that respects their actual hit-type mix — a high-HR
    pull hitter benefits more from a HR-friendly park than a slap hitter.

    Returned dict keys: ``'1B'``, ``'2B'``, ``'3B'``, ``'HR'``, ``'BB'``.
    HBP isn't in Savant's table and is small anyway — we treat it as
    park-neutral (factor 1.0) wherever the formula needs it.
    """
    cols = {
        "1B": "index_1b",
        "2B": "index_2b",
        "3B": "index_3b",
        "HR": "index_hr",
        "BB": "index_bb",
    }
    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in park_factor_df.iterrows():
        team_id = int(row["main_team_id"])
        abbrev = MLBAM_TEAM_ID_TO_ABBREV.get(team_id)
        if abbrev is None:
            continue
        stand = row["bat_side"]
        out[(abbrev, stand)] = {
            event: float(row[col]) / 100.0
            for event, col in cols.items()
        }
    return out


def load_park_factor_lookup(
    parquet_path: Path | str,
    *,
    factor_col: str = "index_xwobacon",
) -> dict[tuple[str, str], float]:
    """Convenience: read a parquet written by ``fetch_park_factors`` and
    return the simple single-column lookup dict.
    """
    return build_park_factor_lookup(
        pd.read_parquet(parquet_path), factor_col=factor_col,
    )


def load_hit_type_park_factor_lookup(
    parquet_path: Path | str,
) -> dict[tuple[str, str], dict[str, float]]:
    """Convenience: read a parquet written by ``fetch_park_factors`` and
    return the full hit-type lookup.
    """
    return build_hit_type_park_factor_lookup(pd.read_parquet(parquet_path))


# wOBA event weights (FanGraphs constants, league-stable to ~4 decimals).
# K and outs have weight 0 (don't contribute to wOBA).
WOBA_EVENT_WEIGHTS: dict[str, float] = {
    "BB": 0.690, "HBP": 0.720,
    "1B": 0.880, "2B": 1.270, "3B": 1.620, "HR": 2.100,
}


def personalized_park_factor(
    event_counts: dict[str, float],
    venue_park_factors: dict[str, float] | None,
) -> float:
    """Hitter-specific park multiplier for wOBA-style metrics.

    Implements the wOBA-weighted decomposition:

        multiplier = sum_e ( w_e * count_e * pf_e )
                   / sum_e ( w_e * count_e )

    where ``w_e`` is the wOBA event weight (constant), ``count_e`` is the
    hitter's count of event ``e`` (cumulative or rolling), and ``pf_e``
    is the park factor for that event in the relevant venue × handedness.

    Park-neutral events (BB, HBP) get ``pf_e = 1.0``. If a hit-type
    factor is missing from the venue map, defaults to 1.0 for that
    event. If the whole venue map is None (venue not in Savant data),
    returns 1.0.

    Effect size: small for contact-and-walk hitters, large for sluggers
    in HR parks. Coors RHB pf_HR=1.12 lifts a 5%-HR hitter's wOBA by
    ~1.5% but a 7.5%-HR hitter's by ~2.5%, and so on.
    """
    if venue_park_factors is None:
        return 1.0
    weighted_park = 0.0
    weighted_neutral = 0.0
    for event, w in WOBA_EVENT_WEIGHTS.items():
        count = float(event_counts.get(event, 0.0) or 0.0)
        if count == 0:
            continue
        pf = venue_park_factors.get(event, 1.0)
        weighted_park += w * count * pf
        weighted_neutral += w * count
    if weighted_neutral <= 0:
        return 1.0
    return weighted_park / weighted_neutral
