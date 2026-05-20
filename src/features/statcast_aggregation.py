"""Aggregate raw Statcast pitch-level data into per-player PA-level counts.

Statcast (pybaseball ``statcast`` / ``statcast_pitcher`` / ``statcast_batter``)
returns one row per *pitch*. The ``events`` column is only populated on the
pitch that ends a plate appearance, and ``bb_type`` is only populated on
batted balls. These helpers do the filter + groupby once so downstream
sabermetric calculations get clean per-player counts.

The output of ``aggregate_pa_counts`` is shaped to feed directly into the
formulas in ``src/features/sabermetrics.py``:

    >>> agg = aggregate_pa_counts(statcast_df, by="pitcher")
    >>> agg["SIERA"] = siera(agg.SO, agg.BB, agg.GB, agg.FB, agg.PU, agg.PA)

This module has no pybaseball dependency — it operates on a generic
DataFrame, so it's equally usable on a Statcast pull, a saved Parquet of
historical pitches, or a synthetic test fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Statcast `events` values that count as a plate appearance.
PA_ENDING_EVENTS: frozenset[str] = frozenset(
    {
        "single", "double", "triple", "home_run",
        "walk", "intent_walk",
        "strikeout", "strikeout_double_play",
        "field_out", "force_out", "grounded_into_double_play",
        "double_play", "triple_play",
        "sac_fly", "sac_bunt", "sac_fly_double_play",
        "hit_by_pitch",
        "field_error", "fielders_choice", "fielders_choice_out",
        "catcher_interf",
    }
)

# Strikeout-like and walk-like events.
STRIKEOUT_EVENTS: frozenset[str] = frozenset({"strikeout", "strikeout_double_play"})
WALK_EVENTS: frozenset[str] = frozenset({"walk", "intent_walk"})

# Hit events (used for AB / batting average computations downstream).
HIT_EVENTS: frozenset[str] = frozenset({"single", "double", "triple", "home_run"})


def pa_end_pitches(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per plate appearance.

    Filters Statcast pitches to those whose ``events`` is in
    ``PA_ENDING_EVENTS`` — i.e. only the last pitch of each PA, excluding
    base-running events (pickoffs, steals, wild pitches, etc.).
    """
    return statcast_df[statcast_df["events"].isin(PA_ENDING_EVENTS)].copy()


def aggregate_pa_counts(
    statcast_df: pd.DataFrame,
    by: str = "pitcher",
) -> pd.DataFrame:
    """Group PAs by ``by`` and return event counts per group.

    Parameters
    ----------
    statcast_df
        Raw Statcast DataFrame (one row per pitch).
    by
        Column to group on. ``'pitcher'`` (default) or ``'batter'`` are the
        common choices; both are MLBAM player IDs.

    Returns
    -------
    DataFrame with columns:
        <by>, PA, SO, BB, HBP, HR, GB, FB, PU, LD, BIP

    where BIP = balls in play (GB + FB + PU + LD).
    """
    pa = pa_end_pitches(statcast_df)
    pa = pa.assign(
        is_so=pa["events"].isin(STRIKEOUT_EVENTS),
        is_bb=pa["events"].isin(WALK_EVENTS),
        is_hbp=(pa["events"] == "hit_by_pitch"),
        is_hr=(pa["events"] == "home_run"),
        is_gb=(pa["bb_type"] == "ground_ball"),
        is_fb=(pa["bb_type"] == "fly_ball"),
        is_pu=(pa["bb_type"] == "popup"),
        is_ld=(pa["bb_type"] == "line_drive"),
    )
    agg = (
        pa.groupby(by, dropna=True)
        .agg(
            PA=("events", "count"),
            SO=("is_so", "sum"),
            BB=("is_bb", "sum"),
            HBP=("is_hbp", "sum"),
            HR=("is_hr", "sum"),
            GB=("is_gb", "sum"),
            FB=("is_fb", "sum"),
            PU=("is_pu", "sum"),
            LD=("is_ld", "sum"),
        )
        .reset_index()
    )
    agg["BIP"] = agg["GB"] + agg["FB"] + agg["PU"] + agg["LD"]
    return agg


def add_rate_stats(agg: pd.DataFrame) -> pd.DataFrame:
    """Append common rate stats (K%, BB%, HR%, GB%, FB%, LD%, PU%) to a counts frame.

    Zero-denominator groups get NaN rates (rather than infinity).
    """
    out = agg.copy()
    pa = out["PA"].replace(0, np.nan)
    bip = out["BIP"].replace(0, np.nan)
    out["K_pct"] = out["SO"] / pa
    out["BB_pct"] = out["BB"] / pa
    out["HR_pct"] = out["HR"] / pa
    out["GB_pct"] = out["GB"] / bip
    out["FB_pct"] = out["FB"] / bip
    out["LD_pct"] = out["LD"] / bip
    out["PU_pct"] = out["PU"] / bip
    return out


def attach_player_names(
    agg: pd.DataFrame,
    id_col: str = "pitcher",
    name_col: str = "player_name",
) -> pd.DataFrame:
    """Join MLBAM ``id_col`` -> player name onto an aggregate.

    Uses ``pybaseball.playerid_reverse_lookup`` so it's a one-liner from the
    notebook side. Imported lazily so this module stays usable without
    pybaseball when you don't need names.
    """
    import pybaseball as pb

    ids = sorted(agg[id_col].dropna().astype(int).unique().tolist())
    if not ids:
        out = agg.copy()
        out[name_col] = pd.NA
        return out

    lookup = pb.playerid_reverse_lookup(ids, key_type="mlbam")
    lookup[name_col] = (
        lookup["name_first"].str.title() + " " + lookup["name_last"].str.title()
    )
    out = agg.merge(
        lookup[["key_mlbam", name_col]],
        left_on=id_col,
        right_on="key_mlbam",
        how="left",
    ).drop(columns=["key_mlbam"])
    return out
