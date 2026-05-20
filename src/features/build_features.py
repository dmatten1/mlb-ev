"""Assemble the per-game training feature set.

Integration layer combining everything we've built:

1. **Outcomes** (``outcomes_loader``) — labels: home/away score, win.
2. **Lineups**  (``lineup_loader``)   — who actually played, with each
   slot's defensive position (boxscore-derived).
3. **Cumulative + rolling Statcast aggregates** (``cumulative``) — point-in-time
   features for every player. Cumulative = season-to-date. Rolling = last 30 days.
4. **Handedness-split park factors** (``fetch_park_factors`` + ``handedness``)
   — Savant's per-hit-type ``(venue, hitter_stand) -> {1B, 2B, 3B, HR, BB}``.
5. **Per-player OAA** (``fetch_oaa``) — Savant's outs-above-average split
   by handedness-of-batter, summed across the OPPOSING team's eight non-DH
   fielders.

Output: ``data/features/training_<year>.parquet`` — one row per game.

xwOBA attaches in four flavors so the model layer can pick whichever
version it wants:

* ``<side>_off_xwOBA``                  — season-cumulative, park-neutral
* ``<side>_off_xwOBA_matchup_adj``      — season-cumulative, after the
                                          full matchup model: blended
                                          GB/FB/LD/PU × per-flight park
                                          factor × per-handedness OAA defense.
* ``<side>_off_xwOBA_30d``              — last-30-day, park-neutral
* ``<side>_off_xwOBA_30d_matchup_adj``  — last-30-day, full matchup model

Same four for the opposing-starter columns (``<side>_sp_xwOBA*``),
computed symmetrically (matchup PF/Def averaged over the OPPOSING lineup
using their handednesses).

Matchup-adjusted columns equal the unadjusted ones whenever park lookup
or OAA data are unavailable.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("features.build_features")

from src.features.bullpen import (
    build_bullpen_features,
    RELIEVER_FEATURE_COLS,
)
from src.features.cumulative import (
    build_cumulative,
    build_rolling,
    compute_rate_stats,
    join_features_asof,
    load_statcast,
)
from src.features.handedness import (
    MLBAM_TEAM_ID_TO_ABBREV,
    build_batter_handedness,
    load_hit_type_park_factor_lookup,
)
from src.features.lineup_loader import DEFAULT_OUTPUT_ROOT as LINEUPS_ROOT
from src.features.matchup import LINEUP_PA_WEIGHTS
from src.features.matchup_xwoba import matchup_xwoba_adjustment
from src.features.outcomes_loader import load_outcomes
from src.features.team_defense import (
    build_oaa_lookup, count_games_by_position, infield_outfield_oaa, load_oaa,
)

DEFAULT_OUTPUT_ROOT = Path("data/features")
DEFAULT_PARK_FACTORS_PATH = Path("data/park_factors/park_factors_2024_rolling3.parquet")
DEFAULT_OAA_ROOT = Path("data/oaa")

# Pitcher (starter) features we attach per game.
PITCHER_FEATURE_COLS: tuple[str, ...] = (
    "SIERA", "K_pct", "BB_pct", "HR_pct",
    "GB_pct", "FB_pct", "LD_pct", "PU_pct",
    "Barrel_pct", "xwOBA", "PA_cum",
)

# Batter features we aggregate across the lineup.
BATTER_FEATURE_COLS: tuple[str, ...] = (
    "K_pct", "BB_pct", "HR_pct",
    "GB_pct", "FB_pct", "LD_pct",
    "Barrel_pct", "xwOBA", "PA_cum",
)

PITCHER_MIN_PA: int = 25
BATTER_MIN_PA: int = 25
# Rolling has much less sample. A bench bat could have 0 PA in last 30d
# even if they've played some this season. Set a floor to drop those rows
# (the lineup re-normalizer will compensate).
PITCHER_MIN_PA_ROLLING: int = 50
BATTER_MIN_PA_ROLLING: int = 15


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _apply_pa_floor(df: pd.DataFrame, feature_cols: list[str],
                    pa_col: str, min_pa: int) -> pd.DataFrame:
    """NaN-out rate stats for players below the PA-sample floor."""
    if min_pa <= 0:
        return df
    out = df.copy()
    mask = (out[pa_col] < min_pa) | out[pa_col].isna()
    for c in feature_cols:
        if c == pa_col or c not in out.columns:
            continue
        out.loc[mask, c] = np.nan
    return out


def _lineup_composite_long(
    long_df: pd.DataFrame,
    value_col: str,
    *,
    group_cols: tuple[str, ...] = ("game_id", "side"),
    slot_col: str = "slot",
) -> pd.Series:
    """PA-weighted lineup composite using ``LINEUP_PA_WEIGHTS``, with
    re-normalization within each group when some hitters are missing.
    """
    weights_by_slot = pd.Series(
        list(LINEUP_PA_WEIGHTS),
        index=pd.Index(range(1, len(LINEUP_PA_WEIGHTS) + 1), name=slot_col),
        name="weight",
    )
    df = long_df[[*group_cols, slot_col, value_col]].copy()
    df["weight"] = df[slot_col].map(weights_by_slot)
    df = df[df[value_col].notna()]
    df["weighted"] = df[value_col].astype(float) * df["weight"]
    agg = df.groupby(list(group_cols)).agg(num=("weighted", "sum"),
                                            den=("weight", "sum"))
    return (agg["num"] / agg["den"]).rename(value_col)


# ---------------------------------------------------------------------------
# Pitcher join
# ---------------------------------------------------------------------------

def _attach_pitcher_features(
    games: pd.DataFrame,
    cum_p: pd.DataFrame,
    *,
    side: str,
    suffix: str = "",
) -> pd.DataFrame:
    """Join starter rate features for one side.

    Pre-renames cumulative columns to include ``suffix`` BEFORE the join
    so a second call (rolling, suffix='_30d') doesn't duplicate names.
    """
    by_col = f"{side}_starter_id"
    prefix = f"{side}_sp_"
    rate_cols = list(PITCHER_FEATURE_COLS)
    if suffix:
        cum_p = cum_p.rename(columns={c: f"{c}{suffix}" for c in rate_cols})
        feature_cols = [f"{c}{suffix}" for c in rate_cols]
    else:
        feature_cols = rate_cols
    return join_features_asof(
        games, cum_p, by=by_col, prefix=prefix, feature_cols=feature_cols,
    )


# ---------------------------------------------------------------------------
# Batter composite (no park adjustment here — that's in matchup attachment)
# ---------------------------------------------------------------------------

def _attach_batter_composites(
    games: pd.DataFrame,
    lineups_long: pd.DataFrame,
    cum_b: pd.DataFrame,
    *,
    suffix: str = "",
    min_pa: int = BATTER_MIN_PA,
) -> pd.DataFrame:
    """Per-(game, side) lineup-weighted composite over BATTER_FEATURE_COL.

    Plain park-neutral composite. Matchup-adjusted xwOBA is computed in
    a separate pass (``_attach_matchup_adjustments``) so it can use the
    pitcher's flight profile + team defense + handedness-keyed park
    factors all together.
    """
    long_with_date = lineups_long.merge(
        games[["game_id", "game_date"]], on="game_id", how="inner",
    )
    joined = join_features_asof(
        long_with_date, cum_b, by="player_id", prefix="",
        feature_cols=list(BATTER_FEATURE_COLS),
    )
    joined = _apply_pa_floor(
        joined, feature_cols=list(BATTER_FEATURE_COLS),
        pa_col="PA_cum", min_pa=min_pa,
    )

    composites = [_lineup_composite_long(joined, col)
                  for col in BATTER_FEATURE_COLS]
    comp = pd.concat(composites, axis=1).reset_index()
    wide = comp.pivot(index="game_id", columns="side",
                      values=list(BATTER_FEATURE_COLS))
    wide.columns = [f"{side_}_off_{feat}{suffix}"
                    for feat, side_ in wide.columns]
    wide = wide.reset_index()
    return games.merge(wide, on="game_id", how="left")


# ---------------------------------------------------------------------------
# Matchup adjustment (flight × park × defense) — replaces old Pathway B
# ---------------------------------------------------------------------------

# Flight columns we look up per hitter and per pitcher. Same names live
# on the cumulative aggregator output (cum_b / cum_p) and on the wide
# games frame (with side prefixes).
_FLIGHT_COLS: tuple[str, ...] = ("GB_pct", "FB_pct", "LD_pct", "PU_pct")


def _team_defense_columns(
    games: pd.DataFrame,
    oaa_R: dict[tuple[int, str], float],
    oaa_L: dict[tuple[int, str], float],
) -> pd.DataFrame:
    """Add ``home_def_inf_R/L``, ``home_def_of_R/L`` (and ``away_def_*``)
    to ``games``: per-handedness OAA sums across the eight non-DH fielders.

    These are computed once per game from the boxscore lineup positions
    and reused for every hitter slot in the matchup pass.
    """
    games = games.copy()
    for side in ("home", "away"):
        ids_col = f"{side}_lineup"
        pos_col = f"{side}_lineup_positions"
        if pos_col not in games.columns:
            games[f"{side}_def_inf_R"] = 0.0
            games[f"{side}_def_inf_L"] = 0.0
            games[f"{side}_def_of_R"] = 0.0
            games[f"{side}_def_of_L"] = 0.0
            continue
        inf_R, of_R, inf_L, of_L = [], [], [], []

        def _to_list(v) -> list:
            if v is None:
                return []
            if hasattr(v, "__len__") and len(v) == 0:
                return []
            return list(v)

        for _, row in games.iterrows():
            ids = _to_list(row[ids_col])
            pos = _to_list(row[pos_col])
            i_r, o_r = infield_outfield_oaa(ids, pos, oaa_R)
            i_l, o_l = infield_outfield_oaa(ids, pos, oaa_L)
            inf_R.append(i_r); of_R.append(o_r)
            inf_L.append(i_l); of_L.append(o_l)
        games[f"{side}_def_inf_R"] = inf_R
        games[f"{side}_def_inf_L"] = inf_L
        games[f"{side}_def_of_R"] = of_R
        games[f"{side}_def_of_L"] = of_L
    return games


def _attach_matchup_adjustments(
    games: pd.DataFrame,
    lineups_long: pd.DataFrame,
    cum_b: pd.DataFrame,
    hit_type_lookup: dict[tuple[str, str], dict[str, float]],
    batter_handedness: dict[int, str],
    *,
    suffix: str = "",
    min_pa: int = BATTER_MIN_PA,
) -> pd.DataFrame:
    """Compute and attach matchup-adjusted xwOBA for BOTH the lineup
    (offense composite) and the opposing starter (pitcher xwOBA-allowed).

    For every (game, batting-side, lineup-slot) row:

        1. Get hitter's flight profile (GB%/FB%/LD%/PU%) and handedness.
        2. Get opposing starter's flight profile from the wide frame
           (``<opp>_sp_<flight>{suffix}``).
        3. Blend 60/40 favoring the pitcher.
        4. Look up the venue's hit-type park factors for the hitter's
           handedness, collapse to per-flight park factors, and compute
           the matchup park multiplier.
        5. Look up the OPPOSING-team OAA (infield + outfield) for this
           hitter's handedness, compute the matchup defense multiplier.
        6. Per-slot adjusted xwOBA for the OFFENSE = hitter_xwOBA × PF × Def.
        7. Per-slot adjusted xwOBA for the PITCHER = opposing-pitcher xwOBA
           × PF × Def. (Same PF/Def — different baseline — different
           composite target.)

    Then composite both with ``LINEUP_PA_WEIGHTS``:

        home_off_xwOBA{suffix}_matchup_adj  = avg over home-lineup slots
        away_off_xwOBA{suffix}_matchup_adj  = avg over away-lineup slots
        away_sp_xwOBA{suffix}_matchup_adj   = avg over home-lineup slots (away pitcher faces home lineup)
        home_sp_xwOBA{suffix}_matchup_adj   = avg over away-lineup slots
    """
    games = games.copy()
    # Bring hitter-level flight + xwOBA into the long frame.
    long_with_meta = lineups_long.merge(
        games[["game_id", "game_date", "home_id",
               "home_sp_xwOBA" + suffix, "away_sp_xwOBA" + suffix,
               *(f"home_sp_{c}{suffix}" for c in _FLIGHT_COLS),
               *(f"away_sp_{c}{suffix}" for c in _FLIGHT_COLS),
               "home_def_inf_R", "home_def_inf_L",
               "home_def_of_R",  "home_def_of_L",
               "away_def_inf_R", "away_def_inf_L",
               "away_def_of_R",  "away_def_of_L"]],
        on="game_id", how="inner",
    )
    joined = join_features_asof(
        long_with_meta, cum_b, by="player_id", prefix="",
        feature_cols=["xwOBA", "PA_cum", *list(_FLIGHT_COLS)],
    )
    joined = _apply_pa_floor(
        joined, feature_cols=["xwOBA", *_FLIGHT_COLS],
        pa_col="PA_cum", min_pa=min_pa,
    )
    joined["venue_abbrev"] = joined["home_id"].map(MLBAM_TEAM_ID_TO_ABBREV)
    joined["batter_stand"] = joined["player_id"].map(batter_handedness)

    def _row_matchup(row):
        side = row["side"]
        opp = "away" if side == "home" else "home"
        # Opposing pitcher's flight profile.
        opp_pitcher_flight = tuple(
            float(row.get(f"{opp}_sp_{c}{suffix}", 0) or 0)
            for c in _FLIGHT_COLS
        )
        # Hitter's own flight.
        hitter_flight = tuple(
            float(row.get(c, 0) or 0) for c in _FLIGHT_COLS
        )
        stand = row["batter_stand"]
        venue_factors = hit_type_lookup.get(
            (row["venue_abbrev"], stand)
        )
        # OAA: defenders behind the OPPOSING pitcher are the OPP team.
        # The defense factor uses OAA conditional on the HITTER's stand.
        if stand == "R":
            inf_oaa = row.get(f"{opp}_def_inf_R", 0.0)
            of_oaa = row.get(f"{opp}_def_of_R", 0.0)
        elif stand == "L":
            inf_oaa = row.get(f"{opp}_def_inf_L", 0.0)
            of_oaa = row.get(f"{opp}_def_of_L", 0.0)
        else:
            # Unknown handedness: fall back to "total" approximation
            inf_oaa = (row.get(f"{opp}_def_inf_R", 0.0)
                       + row.get(f"{opp}_def_inf_L", 0.0)) / 2
            of_oaa = (row.get(f"{opp}_def_of_R", 0.0)
                      + row.get(f"{opp}_def_of_L", 0.0)) / 2

        adj, pf, d, _ = matchup_xwoba_adjustment(
            hitter_xwoba=float(row.get("xwOBA", 0.0) or 0.0),
            pitcher_flight=opp_pitcher_flight,
            hitter_flight=hitter_flight,
            hit_type_pfs=venue_factors,
            infield_oaa=float(inf_oaa or 0.0),
            outfield_oaa=float(of_oaa or 0.0),
        )
        # Pitcher adjusted xwOBA-allowed for this slot uses the SAME PF/Def
        # multiplier but the opposing pitcher's solo xwOBA as the baseline.
        opp_pitcher_xwoba = float(row.get(f"{opp}_sp_xwOBA{suffix}", 0) or 0)
        return pd.Series({
            "xwOBA_matchup_adj": adj,
            "pitcher_xwOBA_matchup_adj": opp_pitcher_xwoba * pf * d,
        })

    matchup = joined.apply(_row_matchup, axis=1)
    joined = pd.concat([joined, matchup], axis=1)

    # Offense composite (averaging over the side's own lineup).
    off_comp = _lineup_composite_long(joined, "xwOBA_matchup_adj")
    off_wide = (
        off_comp.unstack("side")
        .rename_axis(columns=None)
        .rename(columns={
            "home": f"home_off_xwOBA{suffix}_matchup_adj",
            "away": f"away_off_xwOBA{suffix}_matchup_adj",
        })
        .reset_index()
    )

    # Pitcher composite (away pitcher faces home lineup, so the home-side
    # composite of pitcher_xwOBA_matchup_adj IS the away pitcher's number).
    sp_comp = _lineup_composite_long(joined, "pitcher_xwOBA_matchup_adj")
    sp_wide = (
        sp_comp.unstack("side")
        .rename_axis(columns=None)
        .rename(columns={
            "home": f"away_sp_xwOBA{suffix}_matchup_adj",
            "away": f"home_sp_xwOBA{suffix}_matchup_adj",
        })
        .reset_index()
    )
    games = games.merge(off_wide, on="game_id", how="left")
    games = games.merge(sp_wide,  on="game_id", how="left")
    return games


# ---------------------------------------------------------------------------
# Bullpen attachment
# ---------------------------------------------------------------------------

# Per-handedness BP feature columns we surface BEFORE matchup adjustment.
# Mirrors the slim feature set the user picked: xwOBA + SIERA + Barrel%.
# Flight rates come along for the ride because the matchup pass needs
# them (treat the pool as a "one composite pitcher" with these flight rates).
_BP_RAW_FEATURES: tuple[str, ...] = (
    "xwOBA", "SIERA", "Barrel_pct",
    "GB_pct", "FB_pct", "LD_pct", "PU_pct",
)


def _attach_bp_raw_composites(
    games: pd.DataFrame,
    composites_df: pd.DataFrame,
    *,
    suffix: str = "",
) -> pd.DataFrame:
    """Merge per-team BP composites into ``games`` for both sides.

    ``composites_df`` has columns ``team_id``, ``game_date``, ``pool_size``,
    ``R_pool_size``, ``L_pool_size``, and ``R_<feat>`` / ``L_<feat>`` for
    each feature. We emit on the games frame:

        home_bp_R_<feat>{suffix}, home_bp_L_<feat>{suffix},
        home_bp_pool_size{suffix}, home_bp_R_pool_size{suffix}, ...
        (and the away_ mirror)

    Matchup-adjusted versions are computed in a follow-up pass.
    """
    if composites_df is None or composites_df.empty:
        return games
    games = games.copy()
    feat_cols = [c for c in composites_df.columns
                 if c.startswith("R_") or c.startswith("L_")]
    keep = ["team_id", "game_date", "pool_size", *feat_cols]
    comp = composites_df[keep].copy()
    comp["game_date"] = pd.to_datetime(comp["game_date"])
    comp["team_id"] = comp["team_id"].astype("int64")
    games["game_date"] = pd.to_datetime(games["game_date"])

    for side, team_col in (("home", "home_id"), ("away", "away_id")):
        renamed = comp.rename(columns={
            "team_id": team_col,
            "pool_size": f"{side}_bp_pool_size{suffix}",
            **{c: f"{side}_bp_{c}{suffix}" for c in feat_cols},
        })
        games[team_col] = games[team_col].astype("int64")
        games = games.merge(renamed, on=[team_col, "game_date"], how="left")
    return games


def _attach_bullpen_matchup_adjustments(
    games: pd.DataFrame,
    lineups_long: pd.DataFrame,
    cum_b: pd.DataFrame,
    hit_type_lookup: dict[tuple[str, str], dict[str, float]],
    batter_handedness: dict[int, str],
    *,
    suffix: str = "",
    min_pa: int = BATTER_MIN_PA,
) -> pd.DataFrame:
    """Apply matchup adjustment (park × OAA) to the BP composites,
    routing each hitter to the OPPOSING team's bullpen pool of the
    matching handedness (LHB faces LHP, RHB faces RHP).

    Produces, per game:
        home_bp_xwOBA{suffix}_matchup_adj
            — visiting team's BP performance ADJUSTED for the home lineup
              (this is what HOME OFFENSE 'sees' from the opposing pen)
            ... wait, naming is confusing. See below for the convention.
        home_bp_SIERA{suffix}_matchup_adj_lineup
        home_bp_Barrel_pct{suffix}_matchup_adj_lineup
        (and ``away_bp_*`` mirror)

    Naming convention: ``home_bp_<feat>{suffix}_matchup_adj`` refers to
    the HOME team's bullpen, evaluated against the AWAY lineup it'll
    face. So ``away_bp_xwOBA_matchup_adj`` = how the away pen's xwOBA
    looks weighted by the home lineup's slot composition and platoon
    splits. Symmetric to the SP convention (``home_sp_xwOBA_matchup_adj``
    = home pitcher facing away lineup).

    Per-slot routing:
        * LH batter -> opp_bp_L_<feat> (LH reliever pool)
        * RH batter -> opp_bp_R_<feat>
        * Switch / unknown -> opp_bp_R_<feat> (R pool dominates pen rosters)
    """
    if not {f"home_bp_R_xwOBA{suffix}", f"away_bp_R_xwOBA{suffix}"}.issubset(games.columns):
        return games  # No BP data — skip.
    games = games.copy()

    # The pitcher-side fields we need PER SIDE OF THE PEN.
    bp_feat_cols = [
        f"home_bp_R_xwOBA{suffix}", f"home_bp_L_xwOBA{suffix}",
        f"home_bp_R_SIERA{suffix}", f"home_bp_L_SIERA{suffix}",
        f"home_bp_R_Barrel_pct{suffix}", f"home_bp_L_Barrel_pct{suffix}",
        f"home_bp_R_GB_pct{suffix}", f"home_bp_R_FB_pct{suffix}",
        f"home_bp_R_LD_pct{suffix}", f"home_bp_R_PU_pct{suffix}",
        f"home_bp_L_GB_pct{suffix}", f"home_bp_L_FB_pct{suffix}",
        f"home_bp_L_LD_pct{suffix}", f"home_bp_L_PU_pct{suffix}",
        f"away_bp_R_xwOBA{suffix}", f"away_bp_L_xwOBA{suffix}",
        f"away_bp_R_SIERA{suffix}", f"away_bp_L_SIERA{suffix}",
        f"away_bp_R_Barrel_pct{suffix}", f"away_bp_L_Barrel_pct{suffix}",
        f"away_bp_R_GB_pct{suffix}", f"away_bp_R_FB_pct{suffix}",
        f"away_bp_R_LD_pct{suffix}", f"away_bp_R_PU_pct{suffix}",
        f"away_bp_L_GB_pct{suffix}", f"away_bp_L_FB_pct{suffix}",
        f"away_bp_L_LD_pct{suffix}", f"away_bp_L_PU_pct{suffix}",
    ]
    bp_feat_cols = [c for c in bp_feat_cols if c in games.columns]

    long_with_meta = lineups_long.merge(
        games[["game_id", "game_date", "home_id",
               "home_def_inf_R", "home_def_inf_L",
               "home_def_of_R",  "home_def_of_L",
               "away_def_inf_R", "away_def_inf_L",
               "away_def_of_R",  "away_def_of_L",
               *bp_feat_cols]],
        on="game_id", how="inner",
    )
    joined = join_features_asof(
        long_with_meta, cum_b, by="player_id", prefix="",
        feature_cols=["xwOBA", "PA_cum", *list(_FLIGHT_COLS)],
    )
    joined = _apply_pa_floor(
        joined, feature_cols=["xwOBA", *_FLIGHT_COLS],
        pa_col="PA_cum", min_pa=min_pa,
    )
    joined["venue_abbrev"] = joined["home_id"].map(MLBAM_TEAM_ID_TO_ABBREV)
    joined["batter_stand"] = joined["player_id"].map(batter_handedness)

    def _row_bp_matchup(row):
        side = row["side"]
        opp = "away" if side == "home" else "home"
        stand = row["batter_stand"]
        # Pick the opposing-pen pool of the matching handedness.
        # Switch / unknown defaults to R (majority handedness in pens).
        pool_hand = "L" if stand == "L" else "R"

        bp_xwoba = row.get(f"{opp}_bp_{pool_hand}_xwOBA{suffix}")
        bp_siera = row.get(f"{opp}_bp_{pool_hand}_SIERA{suffix}")
        bp_barrel = row.get(f"{opp}_bp_{pool_hand}_Barrel_pct{suffix}")
        bp_flight = tuple(
            float(row.get(f"{opp}_bp_{pool_hand}_{c}{suffix}", 0) or 0)
            for c in _FLIGHT_COLS
        )
        hitter_flight = tuple(
            float(row.get(c, 0) or 0) for c in _FLIGHT_COLS
        )
        venue_factors = hit_type_lookup.get(
            (row["venue_abbrev"], stand)
        )
        if stand == "R":
            inf_oaa = row.get(f"{opp}_def_inf_R", 0.0)
            of_oaa = row.get(f"{opp}_def_of_R", 0.0)
        elif stand == "L":
            inf_oaa = row.get(f"{opp}_def_inf_L", 0.0)
            of_oaa = row.get(f"{opp}_def_of_L", 0.0)
        else:
            inf_oaa = (row.get(f"{opp}_def_inf_R", 0.0)
                       + row.get(f"{opp}_def_inf_L", 0.0)) / 2
            of_oaa = (row.get(f"{opp}_def_of_R", 0.0)
                      + row.get(f"{opp}_def_of_L", 0.0)) / 2

        _, pf, d, _ = matchup_xwoba_adjustment(
            hitter_xwoba=float(row.get("xwOBA", 0.0) or 0.0),
            pitcher_flight=bp_flight,
            hitter_flight=hitter_flight,
            hit_type_pfs=venue_factors,
            infield_oaa=float(inf_oaa or 0.0),
            outfield_oaa=float(of_oaa or 0.0),
        )
        # Apply PF * D to the BP's xwOBA (handles park & defense).
        # SIERA and Barrel% pass through unmodified (they're skill stats
        # that don't change with park / defense the same way).
        bp_xwoba_adj = (
            float(bp_xwoba) * pf * d if pd.notna(bp_xwoba) else np.nan
        )
        return pd.Series({
            "bp_xwOBA_matchup_adj": bp_xwoba_adj,
            "bp_SIERA_lineup": float(bp_siera) if pd.notna(bp_siera) else np.nan,
            "bp_Barrel_pct_lineup": float(bp_barrel) if pd.notna(bp_barrel) else np.nan,
        })

    matchup = joined.apply(_row_bp_matchup, axis=1)
    joined = pd.concat([joined, matchup], axis=1)

    # Composite each BP feature over the lineup that faces it. The OPP
    # bullpen is what the LINEUP faces — so home lineup composites
    # produce features about the AWAY bullpen, and vice versa.
    def _compose(value_col: str, out_home_name: str, out_away_name: str):
        comp = _lineup_composite_long(joined, value_col)
        wide = (
            comp.unstack("side")
            .rename_axis(columns=None)
            .rename(columns={"home": out_away_name, "away": out_home_name})
            .reset_index()
        )
        return wide

    xwoba_wide = _compose(
        "bp_xwOBA_matchup_adj",
        out_home_name=f"home_bp_xwOBA{suffix}_matchup_adj",
        out_away_name=f"away_bp_xwOBA{suffix}_matchup_adj",
    )
    siera_wide = _compose(
        "bp_SIERA_lineup",
        out_home_name=f"home_bp_SIERA{suffix}_matchup",
        out_away_name=f"away_bp_SIERA{suffix}_matchup",
    )
    barrel_wide = _compose(
        "bp_Barrel_pct_lineup",
        out_home_name=f"home_bp_Barrel_pct{suffix}_matchup",
        out_away_name=f"away_bp_Barrel_pct{suffix}_matchup",
    )
    games = games.merge(xwoba_wide,  on="game_id", how="left")
    games = games.merge(siera_wide,  on="game_id", how="left")
    games = games.merge(barrel_wide, on="game_id", how="left")
    return games


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def _load_reference_inputs(
    year: int,
    *,
    statcast_years: tuple[int, ...],
    lineups_root: Path,
    park_factors_path: Path | None,
    oaa_root: Path | None,
    rolling_window_days: int,
) -> dict:
    """Load every reference table that the feature pipeline needs.

    Returns a dict so the same inputs can be re-used by ``build_training_features``
    (historical) and ``build_projected_features`` (today's slate). The
    expensive bits are statcast load + cumulative/rolling aggregates
    (≈10s); everything else is in-memory.
    """
    lineups_root = Path(lineups_root)
    lineups_wide = pd.read_parquet(lineups_root / f"lineups_{year}.parquet")
    lineups_long = pd.read_parquet(lineups_root / f"lineups_long_{year}.parquet")

    sc = load_statcast(statcast_years)
    cum_p = compute_rate_stats(build_cumulative(sc, group="pitcher"))
    cum_b = compute_rate_stats(build_cumulative(sc, group="batter"))
    rol_p = compute_rate_stats(build_rolling(sc, group="pitcher",
                                              window_days=rolling_window_days))
    rol_b = compute_rate_stats(build_rolling(sc, group="batter",
                                              window_days=rolling_window_days))

    hit_type_lookup = None
    batter_handedness = None
    oaa_R: dict = {}
    oaa_L: dict = {}
    if park_factors_path is not None and Path(park_factors_path).exists():
        hit_type_lookup = load_hit_type_park_factor_lookup(park_factors_path)
        hand_df = build_batter_handedness(sc)
        batter_handedness = dict(zip(hand_df["batter"].astype(int),
                                      hand_df["stand"]))
    if oaa_root is not None and (Path(oaa_root) / f"oaa_{year}.parquet").exists():
        oaa_df = load_oaa(year, root=oaa_root)
        games_by_pp = count_games_by_position(lineups_long)
        oaa_R = build_oaa_lookup(oaa_df, stand="R",
                                  games_by_player_position=games_by_pp)
        oaa_L = build_oaa_lookup(oaa_df, stand="L",
                                  games_by_player_position=games_by_pp)

    return {
        "sc": sc,
        "cum_p": cum_p, "cum_b": cum_b, "rol_p": rol_p, "rol_b": rol_b,
        "lineups_wide": lineups_wide, "lineups_long": lineups_long,
        "hit_type_lookup": hit_type_lookup,
        "batter_handedness": batter_handedness,
        "oaa_R": oaa_R, "oaa_L": oaa_L,
    }


def _compute_feature_columns(games: pd.DataFrame, ref: dict,
                              *, rolling_window_days: int = 30) -> pd.DataFrame:
    """Add every feature column the model uses to a ``games`` frame.

    ``games`` must already have: ``game_id``, ``game_date``, ``home_id``,
    ``away_id``, ``home_starter_id``, ``away_starter_id``, ``home_lineup``,
    ``away_lineup``, and optionally ``*_lineup_positions``.

    ``ref`` is whatever :func:`_load_reference_inputs` returned.

    For projected slates the lineups_long inside ``ref`` will not contain
    today's games — we build a side-cart projected long frame from the
    new ``games`` and concatenate it on for the batter / matchup steps.
    """
    # Build the per-game "long" view for the games we're computing. For
    # training data this is already in ref["lineups_long"]. For projected
    # data we synthesize it from the games_df itself.
    long_from_games = _explode_games_to_long(games)

    # Combine: use historical long for all games NOT in this batch, plus
    # our synthesized long for this batch. Lookups are keyed by player_id
    # and joined against cum_b / rol_b which span all years — so as long
    # as the batter has any statcast appearances, the join works.
    base_long = ref["lineups_long"]
    if "game_id" in base_long.columns:
        base_long = base_long[~base_long["game_id"].isin(games["game_id"].unique())]
    full_long = pd.concat([base_long, long_from_games], ignore_index=True)

    # --- Starter features ---
    games = _attach_pitcher_features(games, ref["cum_p"], side="home", suffix="")
    games = _attach_pitcher_features(games, ref["cum_p"], side="away", suffix="")
    games = _attach_pitcher_features(games, ref["rol_p"], side="home", suffix="_30d")
    games = _attach_pitcher_features(games, ref["rol_p"], side="away", suffix="_30d")
    for prefix in ("home_sp_", "away_sp_"):
        rate_cols = [f"{prefix}{c}" for c in PITCHER_FEATURE_COLS if c != "PA_cum"]
        games = _apply_pa_floor(games, rate_cols, pa_col=f"{prefix}PA_cum",
                                 min_pa=PITCHER_MIN_PA)
        rate_cols_30d = [f"{prefix}{c}_30d" for c in PITCHER_FEATURE_COLS if c != "PA_cum"]
        games = _apply_pa_floor(games, rate_cols_30d, pa_col=f"{prefix}PA_cum_30d",
                                 min_pa=PITCHER_MIN_PA_ROLLING)

    # --- Batter composites (park-neutral) ---
    games = _attach_batter_composites(games, full_long, ref["cum_b"],
                                       suffix="", min_pa=BATTER_MIN_PA)
    games = _attach_batter_composites(games, full_long, ref["rol_b"],
                                       suffix="_30d", min_pa=BATTER_MIN_PA_ROLLING)

    # --- Team defense + matchup adjustments ---
    games = _team_defense_columns(games, ref["oaa_R"], ref["oaa_L"])
    if ref["hit_type_lookup"] is not None and ref["batter_handedness"] is not None:
        games = _attach_matchup_adjustments(
            games, full_long, ref["cum_b"],
            ref["hit_type_lookup"], ref["batter_handedness"],
            suffix="", min_pa=BATTER_MIN_PA,
        )
        games = _attach_matchup_adjustments(
            games, full_long, ref["rol_b"],
            ref["hit_type_lookup"], ref["batter_handedness"],
            suffix="_30d", min_pa=BATTER_MIN_PA_ROLLING,
        )

    # --- Bullpen composites ---
    # Make sure the pool computation includes any (team, date) pairs in
    # this batch that aren't already in lineups_wide (e.g. tonight's
    # projected slate — today's appearances don't exist yet).
    extras = (
        list(zip(games["home_id"].astype("int64"),
                 pd.to_datetime(games["game_date"])))
        + list(zip(games["away_id"].astype("int64"),
                   pd.to_datetime(games["game_date"])))
    )
    bp = build_bullpen_features(ref["sc"], ref["lineups_wide"],
                                 rolling_window_days=rolling_window_days,
                                 extra_team_dates=extras)
    games = _attach_bp_raw_composites(games, bp["cum_composites"], suffix="")
    games = _attach_bp_raw_composites(games, bp["rol_composites"], suffix="_30d")
    if ref["hit_type_lookup"] is not None and ref["batter_handedness"] is not None:
        games = _attach_bullpen_matchup_adjustments(
            games, full_long, ref["cum_b"],
            ref["hit_type_lookup"], ref["batter_handedness"],
            suffix="", min_pa=BATTER_MIN_PA,
        )
        games = _attach_bullpen_matchup_adjustments(
            games, full_long, ref["rol_b"],
            ref["hit_type_lookup"], ref["batter_handedness"],
            suffix="_30d", min_pa=BATTER_MIN_PA_ROLLING,
        )
    return games


def _explode_games_to_long(games: pd.DataFrame) -> pd.DataFrame:
    """Replicate ``lineup_loader.explode_lineups`` directly on a games frame.

    We use this when the lineups_long parquet doesn't yet contain rows
    for our games (e.g. projected slates for tonight). Output schema
    matches ``lineups_long``: ``game_id, side, slot, player_id, position``.
    """
    pieces = []
    for side in ("home", "away"):
        lineup_col = f"{side}_lineup"
        pos_col = f"{side}_lineup_positions"
        if lineup_col not in games.columns:
            continue
        cols = ["game_id", lineup_col]
        if pos_col in games.columns:
            cols.append(pos_col)
        sub = games[cols].copy()
        if pos_col in sub.columns:
            sub_l = sub[["game_id", lineup_col]].explode(lineup_col, ignore_index=True)
            sub_p = sub[["game_id", pos_col]].explode(pos_col, ignore_index=True)
            sub = sub_l.assign(position=sub_p[pos_col])
        else:
            sub = sub.explode(lineup_col, ignore_index=True)
            sub["position"] = None
        sub["slot"] = sub.groupby("game_id").cumcount() + 1
        sub = sub.rename(columns={lineup_col: "player_id"})
        sub["side"] = side
        pieces.append(sub[["game_id", "side", "slot", "player_id", "position"]])
    if not pieces:
        return pd.DataFrame(columns=["game_id", "side", "slot", "player_id", "position"])
    out = pd.concat(pieces, ignore_index=True)
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    return out


def build_training_features(
    year: int,
    *,
    statcast_years: tuple[int, ...] | None = None,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    lineups_root: Path | str = LINEUPS_ROOT,
    park_factors_path: Path | str | None = DEFAULT_PARK_FACTORS_PATH,
    oaa_root: Path | str | None = DEFAULT_OAA_ROOT,
    rolling_window_days: int = 30,
) -> pd.DataFrame:
    """End-to-end builder for one season.

    Produces, for every game:
      * Season-cumulative pitcher and batter features (park-neutral).
      * Rolling-N-day pitcher and batter features (park-neutral).
      * **Matchup-adjusted xwOBA** for both offense and opposing starter,
        in both cumulative and rolling flavors. The adjustment is:
        per-hitter blended-flight × per-flight handedness-keyed park
        factor × per-handedness opposing-team OAA defense factor.

    If ``park_factors_path`` is None, the matchup-adjusted columns are
    skipped. If ``oaa_root`` is None, defense factor falls back to 1.0
    (only park is applied).
    """
    statcast_years = statcast_years or (year,)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    lineups_root = Path(lineups_root)

    # --- 1. Load labels + lineups (positions included) ---
    games = load_outcomes(year)
    ref = _load_reference_inputs(
        year=year,
        statcast_years=statcast_years,
        lineups_root=lineups_root,
        park_factors_path=Path(park_factors_path) if park_factors_path else None,
        oaa_root=Path(oaa_root) if oaa_root else None,
        rolling_window_days=rolling_window_days,
    )

    lineup_keep = [
        "game_id", "home_starter_id", "away_starter_id",
        "home_lineup", "away_lineup",
        "home_pitchers_used", "away_pitchers_used",
    ]
    for c in ("home_lineup_positions", "away_lineup_positions"):
        if c in ref["lineups_wide"].columns:
            lineup_keep.append(c)
    games = games.merge(ref["lineups_wide"][lineup_keep], on="game_id", how="inner")
    games = games.dropna(subset=["home_starter_id", "away_starter_id"]).reset_index(drop=True)

    # --- 2-8. Compute every feature column ---
    games = _compute_feature_columns(games, ref, rolling_window_days=rolling_window_days)

    # --- 9. Final shape ---
    games["season_year"] = year
    out = output_root / f"training_{year}.parquet"
    games.to_parquet(out, index=False)
    return games


def build_projected_features(
    target_date: date | str,
    year: int | None = None,
    *,
    statcast_years: tuple[int, ...] | None = None,
    schedule_root: Path | str = Path("data/raw/schedule"),
    schedule_sport: str = "baseball_mlb",
    lineups_root: Path | str = LINEUPS_ROOT,
    park_factors_path: Path | str | None = DEFAULT_PARK_FACTORS_PATH,
    oaa_root: Path | str | None = DEFAULT_OAA_ROOT,
    rolling_window_days: int = 30,
    output_root: Path | str | None = Path("data/features/projected"),
    use_actual_lineups: bool = True,
    filter_active_roster: bool = True,
) -> pd.DataFrame:
    """Build features for the upcoming slate using projected lineups.

    Pulls the persisted schedule JSON for ``target_date`` (written by
    ``src.ingest.fetch_schedule``). For each game, if both teams have
    published their lineup card to MLB-StatsAPI (typically ~3 hours
    before first pitch), uses the **actual** lineup; otherwise falls
    back to the modal **projected** lineup. Per-side ``lineup_source``
    columns (``'actual'`` vs ``'projected'``) track which is which.

    Runs the same feature pipeline ``build_training_features`` uses on
    the result. Scores are NaN. Returns a DataFrame whose columns
    superset matches ``training_<year>.parquet`` (minus score/win labels).

    If ``output_root`` is given, also writes
    ``data/features/projected/projected_<YYYY-MM-DD>.parquet``.
    """
    from src.ingest.fetch_schedule import load_schedule_for_date, pitcher_hand_lookup
    from src.features.lineup_projection import (
        project_lineups_for_schedule, fetch_published_lineup,
    )

    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    if year is None:
        year = target_date.year
    statcast_years = statcast_years or (year,)

    # 1. Schedule + probable pitchers
    schedule_df = load_schedule_for_date(target_date,
                                          local_root=schedule_root,
                                          sport=schedule_sport)
    if schedule_df.empty:
        return pd.DataFrame()

    # 2. Project lineups for both sides of every game (IL-filtered).
    features_path = Path(f"data/features/training_{year}.parquet")
    proj_map = project_lineups_for_schedule(
        schedule_df,
        year=year,
        lineups_root=Path(lineups_root),
        features_path=features_path,
        pitcher_hand_map=pitcher_hand_lookup(years=(year,)),
        filter_active_roster=filter_active_roster,
    )
    if not proj_map:
        return pd.DataFrame()

    # 3. Override layer: if MLB has posted the lineup card already, use
    #    those actuals instead of our modal projection.
    actual_map: dict[int, "PublishedLineups | None"] = {}
    if use_actual_lineups:
        n_actual = 0
        for gid in proj_map:
            pub = fetch_published_lineup(gid)
            actual_map[gid] = pub
            if pub is not None:
                n_actual += 1
        logger.info("[predict] published lineups available for %d / %d games",
                    n_actual, len(proj_map))

    # 4. Assemble a games frame with the same shape build_training_features uses.
    rows: list[dict] = []
    for _, r in schedule_df.iterrows():
        gid = int(r["game_id"])
        if gid not in proj_map:
            continue
        h_proj = proj_map[gid]["home"]
        a_proj = proj_map[gid]["away"]
        pub = actual_map.get(gid)

        # Per-side source resolution. (We only swap a side in when the
        # whole pub is non-None — partial posts are rare and treating
        # one side as actual + one as projected risks lineup-position
        # collisions.)
        if pub is not None:
            home_lineup = pub.home_lineup
            away_lineup = pub.away_lineup
            home_pos = pub.home_positions
            away_pos = pub.away_positions
            home_sp = pub.home_starter_id if pub.home_starter_id is not None \
                      else (int(r["home_probable_pitcher_id"])
                            if pd.notna(r.get("home_probable_pitcher_id")) else None)
            away_sp = pub.away_starter_id if pub.away_starter_id is not None \
                      else (int(r["away_probable_pitcher_id"])
                            if pd.notna(r.get("away_probable_pitcher_id")) else None)
            home_source = "actual"
            away_source = "actual"
        else:
            home_lineup = h_proj.player_ids
            away_lineup = a_proj.player_ids
            home_pos = h_proj.positions
            away_pos = a_proj.positions
            home_sp = (int(r["home_probable_pitcher_id"])
                       if pd.notna(r.get("home_probable_pitcher_id")) else None)
            away_sp = (int(r["away_probable_pitcher_id"])
                       if pd.notna(r.get("away_probable_pitcher_id")) else None)
            home_source = "projected"
            away_source = "projected"

        rows.append({
            "game_id": gid,
            "game_date": pd.Timestamp(r["game_date"]).to_pydatetime(),
            "game_datetime": r["game_datetime"],
            "game_type": r.get("game_type") or "R",
            "status": r.get("status") or "Scheduled",
            "home_id": int(r["home_id"]),
            "home_name": r["home_name"],
            "away_id": int(r["away_id"]),
            "away_name": r["away_name"],
            "home_score": pd.NA,
            "away_score": pd.NA,
            "home_starter_id": home_sp if home_sp is not None else pd.NA,
            "away_starter_id": away_sp if away_sp is not None else pd.NA,
            "home_lineup": home_lineup,
            "away_lineup": away_lineup,
            "home_lineup_positions": home_pos,
            "away_lineup_positions": away_pos,
            "home_pitchers_used": [home_sp] if home_sp is not None else [],
            "away_pitchers_used": [away_sp] if away_sp is not None else [],
            "venue_id": r.get("venue_id"),
            "venue_name": r.get("venue_name"),
            "doubleheader": r.get("doubleheader"),
            "game_num": r.get("game_num"),
            "home_projected_pool_size": h_proj.pool_size,
            "away_projected_pool_size": a_proj.pool_size,
            "home_projected_platoon": h_proj.used_platoon_split,
            "away_projected_platoon": a_proj.used_platoon_split,
            "home_lineup_source": home_source,
            "away_lineup_source": away_source,
        })
    games = pd.DataFrame(rows)
    games = games.dropna(subset=["home_starter_id", "away_starter_id"]).reset_index(drop=True)
    if games.empty:
        return games

    # 4. Load reference inputs + compute features
    ref = _load_reference_inputs(
        year=year,
        statcast_years=statcast_years,
        lineups_root=lineups_root,
        park_factors_path=Path(park_factors_path) if park_factors_path else None,
        oaa_root=Path(oaa_root) if oaa_root else None,
        rolling_window_days=rolling_window_days,
    )
    games = _compute_feature_columns(games, ref, rolling_window_days=rolling_window_days)
    games["season_year"] = year
    games["is_projected"] = True

    if output_root is not None:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        out = output_root / f"projected_{target_date.isoformat()}.parquet"
        games.to_parquet(out, index=False)
    return games
