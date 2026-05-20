"""Point-in-time cumulative + rolling-window aggregates from raw Statcast.

For training a no-lookahead model we need, for every (player, calendar_date)
pair, a feature vector summarizing the player's performance **strictly
before** that date. This module builds two flavors of that index and exposes:

* ``build_cumulative(statcast_df, group='pitcher')`` — daily-bucketed
  season-to-date running totals per ``(player, season)``.
* ``build_rolling(statcast_df, group, window_days=30)`` — last-N-day
  rolling sums per player. Captures current form vs season-cumulative
  which smooths team-to-team variance too aggressively.
* ``compute_rate_stats(cumulative)`` — append SIERA, K%, BB%, HR%, GB%,
  FB%, LD%, Barrel%, **true xwOBA**, and realized **wOBA** on top of
  either cumulative or rolling counts.
* ``lookup_asof(cumulative, player_id, as_of_date)`` — single-player query.
* ``join_features_asof(games, cumulative, by, ...)`` — bulk join via
  ``pd.merge_asof`` (``direction='backward'`` + ``allow_exact_matches=False``).

**xwOBA note**: prior versions of this module computed ``xwOBA`` as
``sum(woba_value) / sum(woba_denom)``, which is actually *realized* wOBA.
True Statcast xwOBA uses ``estimated_woba_using_speedangle`` for balls in
play (a league-wide calibrated EV+LA -> xwOBA lookup) and ``woba_value``
for non-BIP events (walks, HBP, K). We now compute both:

* ``xwOBA`` = ``XWOBA_NUM_cum / WOBA_DEN_cum``  (true xwOBA, park-neutral)
* ``wOBA``  = ``WOBA_NUM_cum  / WOBA_DEN_cum``  (realized wOBA, embeds park)

Park-adjustment of xwOBA for a specific game (multiplying by the venue's
``index_xwobacon`` for the hitter's handedness) happens later in
``build_features.py`` — not here — so the cumulative feature stays
park-neutral and can be re-adjusted for any game.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.sabermetrics import siera
from src.features.statcast_aggregation import (
    PA_ENDING_EVENTS, STRIKEOUT_EVENTS, WALK_EVENTS,
)

# Daily-bucket count columns. Pre-cumsum names.
# Hit-type counts (1B/2B/3B/HR) drive per-hitter park-factor adjustment in
# ``build_features``: each hitter's personalized park multiplier is a
# wOBA-weighted average of their hit-type mix × Savant's hit-type park
# factors for the venue × handedness.
_DAILY_COUNT_COLS: tuple[str, ...] = (
    "PA", "SO", "BB", "HBP",
    "_1B", "_2B", "_3B", "HR",
    "GB", "FB", "PU", "LD",
    "BBE", "BARREL",
    "WOBA_NUM", "XWOBA_NUM", "WOBA_DEN",
)

# After cumsum we keep the raw name + add `_cum` for the running total.
_CUM_SUFFIX = "_cum"


def load_statcast(years: Iterable[int],
                  root: Path | str = Path("data/raw/statcast")) -> pd.DataFrame:
    """Concatenate one or more season parquets from ``fetch_statcast_history``."""
    root = Path(root)
    frames = []
    for y in years:
        p = root / f"statcast_{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing. Run: python -m src.ingest.fetch_statcast_history --year {y}"
            )
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _daily_counts(statcast: pd.DataFrame, group: str) -> pd.DataFrame:
    """Per-(player, game_date) event counts. One row per player per day."""
    if group not in {"pitcher", "batter"}:
        raise ValueError(f"group must be 'pitcher' or 'batter', got {group!r}")

    pa = statcast[statcast["events"].isin(PA_ENDING_EVENTS)].copy()

    # True xwOBA numerator per PA-ending pitch:
    #   * BIP (bb_type not null): use estimated_woba_using_speedangle
    #     (Savant's league-wide EV+LA -> xwOBA lookup). Fall back to
    #     woba_value if speedangle estimate is missing.
    #   * Non-BIP (BB, HBP, K): use woba_value directly (these contribute
    #     the same to xwOBA as to realized wOBA — they're park-neutral).
    is_bip = pa["bb_type"].notna()
    xwoba_per_pa = np.where(
        is_bip,
        pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"]).astype(float),
        pa["woba_value"].fillna(0.0).astype(float),
    )

    pa = pa.assign(
        is_so=pa["events"].isin(STRIKEOUT_EVENTS),
        is_bb=pa["events"].isin(WALK_EVENTS),
        is_hbp=(pa["events"] == "hit_by_pitch"),
        is_1b=(pa["events"] == "single"),
        is_2b=(pa["events"] == "double"),
        is_3b=(pa["events"] == "triple"),
        is_hr=(pa["events"] == "home_run"),
        is_gb=(pa["bb_type"] == "ground_ball"),
        is_fb=(pa["bb_type"] == "fly_ball"),
        is_pu=(pa["bb_type"] == "popup"),
        is_ld=(pa["bb_type"] == "line_drive"),
        is_bbe=pa["bb_type"].notna(),
        is_barrel=(pa["launch_speed_angle"] == 6),
        woba_num=pa["woba_value"].fillna(0.0).astype(float),
        xwoba_num=xwoba_per_pa,
        woba_den=pa["woba_denom"].fillna(0.0).astype(float),
    )

    daily = (
        pa.groupby([group, "game_year", "game_date"], as_index=False)
        .agg(
            PA=("events", "count"),
            SO=("is_so", "sum"),
            BB=("is_bb", "sum"),
            HBP=("is_hbp", "sum"),
            _1B=("is_1b", "sum"),
            _2B=("is_2b", "sum"),
            _3B=("is_3b", "sum"),
            HR=("is_hr", "sum"),
            GB=("is_gb", "sum"),
            FB=("is_fb", "sum"),
            PU=("is_pu", "sum"),
            LD=("is_ld", "sum"),
            BBE=("is_bbe", "sum"),
            BARREL=("is_barrel", "sum"),
            WOBA_NUM=("woba_num", "sum"),
            XWOBA_NUM=("xwoba_num", "sum"),
            WOBA_DEN=("woba_den", "sum"),
        )
    )
    return daily.rename(columns={group: "player_id"})


def build_cumulative(statcast: pd.DataFrame,
                     group: str = "pitcher") -> pd.DataFrame:
    """Per-(player, season) running totals, one row per player-day-with-events.

    Columns:
        player_id, game_year, game_date,
        PA_cum, SO_cum, BB_cum, HBP_cum, HR_cum,
        GB_cum, FB_cum, PU_cum, LD_cum,
        BBE_cum, BARREL_cum, WOBA_NUM_cum, WOBA_DEN_cum

    Each row is the running total **through end of that game_date**.
    Reset to 0 at each season boundary.
    """
    daily = _daily_counts(statcast, group=group)
    daily = daily.sort_values(["player_id", "game_year", "game_date"])
    cum = (
        daily.groupby(["player_id", "game_year"])[list(_DAILY_COUNT_COLS)].cumsum()
    )
    cum.columns = [f"{c}{_CUM_SUFFIX}" for c in cum.columns]
    out = pd.concat(
        [daily[["player_id", "game_year", "game_date"]].reset_index(drop=True),
         cum.reset_index(drop=True)],
        axis=1,
    )
    return out


def _safe_div(num, den):
    """Element-wise safe division. NaN where denominator is zero."""
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    out = np.where(d > 0, n / np.where(d > 0, d, 1.0), np.nan)
    return out


def compute_rate_stats(cumulative: pd.DataFrame) -> pd.DataFrame:
    """Append SIERA + rate columns to a cumulative-counts frame.

    Adds:
        SIERA, K_pct, BB_pct, HR_pct,
        GB_pct, FB_pct, LD_pct, PU_pct,
        Barrel_pct (per BBE), xwOBA (woba_num/woba_den).

    Rate denominators use cumulative totals, so values stabilize over the
    season the way fan-graphs / Savant leaderboards do.
    """
    df = cumulative.copy()
    pa = df["PA_cum"]
    bip = df["GB_cum"] + df["FB_cum"] + df["PU_cum"] + df["LD_cum"]
    bbe = df["BBE_cum"]
    den = df["WOBA_DEN_cum"]

    df["BIP_cum"] = bip
    df["K_pct"] = _safe_div(df["SO_cum"], pa)
    df["BB_pct"] = _safe_div(df["BB_cum"], pa)
    df["HR_pct"] = _safe_div(df["HR_cum"], pa)
    df["GB_pct"] = _safe_div(df["GB_cum"], bip)
    df["FB_pct"] = _safe_div(df["FB_cum"], bip)
    df["LD_pct"] = _safe_div(df["LD_cum"], bip)
    df["PU_pct"] = _safe_div(df["PU_cum"], bip)
    df["Barrel_pct"] = _safe_div(df["BARREL_cum"], bbe)
    # True xwOBA: estimated wOBA on contact (EV+LA driven, league-wide
    # calibrated, park-neutral) + realized wOBA on non-BIP events.
    df["xwOBA"] = _safe_div(df["XWOBA_NUM_cum"], den)
    # Realized wOBA — kept for diagnostics / comparison with xwOBA.
    df["wOBA"] = _safe_div(df["WOBA_NUM_cum"], den)

    df["SIERA"] = siera(
        df["SO_cum"], df["BB_cum"],
        df["GB_cum"], df["FB_cum"], df["PU_cum"],
        df["PA_cum"],
    )
    return df


def build_rolling(statcast: pd.DataFrame,
                  group: str = "pitcher",
                  window_days: int = 30) -> pd.DataFrame:
    """Per-player rolling N-day sums of every event count, one row per
    player-day-with-events.

    Output schema matches ``build_cumulative`` (``*_cum`` suffix on every
    count column) so ``compute_rate_stats`` works on either. The ``_cum``
    suffix is a bit of a lie here — these are window sums, not cumulative —
    but it lets the rest of the pipeline stay one code path.

    The rolling window is purely calendar-based: it includes every event
    in the last ``window_days`` days regardless of season boundary. In
    practice the long offseason gap means an April game's window only
    sees in-season events; no special handling needed.

    Parameters
    ----------
    statcast
        Output of ``load_statcast(years)``.
    group
        ``'pitcher'`` or ``'batter'``.
    window_days
        Window size in calendar days. 30 captures recent form; 14 is
        more reactive but noisier; 60 is smoother.
    """
    daily = _daily_counts(statcast, group=group)
    daily = daily.sort_values(["player_id", "game_date"])

    indexed = daily.set_index("game_date")
    rolled = (
        indexed.groupby("player_id")[list(_DAILY_COUNT_COLS)]
        .rolling(f"{window_days}D")
        .sum()
        .reset_index()
    )
    # game_year on the rolling row should be the year of the row's date
    # (matches the as-of-join expectation), not the earliest year in the
    # window. Re-derive from game_date.
    rolled["game_year"] = rolled["game_date"].dt.year.astype("int64")
    rolled = rolled.rename(columns={c: f"{c}{_CUM_SUFFIX}"
                                     for c in _DAILY_COUNT_COLS})
    front = ["player_id", "game_year", "game_date"]
    other = [c for c in rolled.columns if c not in front]
    return rolled[front + other].reset_index(drop=True)


def lookup_asof(cumulative: pd.DataFrame,
                player_id: int,
                as_of_date: str | pd.Timestamp,
                same_season_only: bool = True) -> pd.Series | None:
    """Return the cumulative row STRICTLY BEFORE ``as_of_date`` for ``player_id``.

    Returns ``None`` if the player has no recorded events before that date.

    If ``same_season_only`` (default), the returned row is restricted to
    events from the same calendar-year as ``as_of_date`` — this matches how
    season stats reset at the start of each year. Set False to fall back to
    any prior season if the current season has no events yet.
    """
    target = pd.Timestamp(as_of_date)
    history = cumulative[cumulative["player_id"] == player_id]
    if same_season_only:
        history = history[history["game_year"] == target.year]
    history = history[history["game_date"] < target]
    if history.empty:
        return None
    return history.iloc[-1]


def join_features_asof(
    games: pd.DataFrame,
    cumulative: pd.DataFrame,
    *,
    by: str,
    feature_cols: list[str] | None = None,
    prefix: str = "",
    game_date_col: str = "game_date",
    same_season_only: bool = True,
) -> pd.DataFrame:
    """Bulk-attach as-of-date features to a ``games`` DataFrame.

    Parameters
    ----------
    games
        DataFrame containing at least ``game_date_col`` and ``by``.
    cumulative
        Output of ``compute_rate_stats(build_cumulative(...))``.
    by
        Column in ``games`` whose values are MLBAM player IDs to match
        against ``cumulative.player_id``. Examples: ``'home_pitcher_id'``,
        ``'away_pitcher_id'``, ``'batter_id'``.
    feature_cols
        Which cumulative columns to attach. Defaults to all rate stats
        plus SIERA and a few raw cumulatives. Pass an explicit list for
        smaller joins.
    prefix
        Prefix prepended to every feature column in the output (e.g.
        ``'home_sp_'`` -> ``home_sp_SIERA``).
    same_season_only
        If True, restrict each match to the same season as the game date.
    """
    if feature_cols is None:
        feature_cols = [
            "SIERA", "K_pct", "BB_pct", "HR_pct",
            "GB_pct", "FB_pct", "LD_pct", "PU_pct",
            "Barrel_pct", "xwOBA",
            "PA_cum", "BBE_cum",
        ]
    missing_in_cum = [c for c in feature_cols if c not in cumulative.columns]
    if missing_in_cum:
        raise KeyError(f"feature_cols not in cumulative: {missing_in_cum}")

    games = games.copy()
    cum = cumulative.rename(columns={"player_id": by}).copy()

    # merge_asof is strict about dtype matches on both `by` columns and
    # the on-key. Normalize to consistent dtypes:
    #   - ID columns: plain int64 (absorbs Int64/nullable variants)
    #   - date columns: datetime64[ns] (absorbs s/us/ns resolution mismatch)
    games[by] = pd.to_numeric(games[by], errors="coerce").astype("Int64").astype("int64")
    cum[by] = pd.to_numeric(cum[by], errors="coerce").astype("Int64").astype("int64")
    games[game_date_col] = pd.to_datetime(games[game_date_col]).astype("datetime64[ns]")
    cum["game_date"] = pd.to_datetime(cum["game_date"]).astype("datetime64[ns]")

    games_sorted = games.sort_values(game_date_col).reset_index().rename(
        columns={"index": "_orig_idx"}
    )
    cum_sorted = cum.sort_values("game_date")

    if same_season_only:
        games_sorted["_year"] = games_sorted[game_date_col].dt.year.astype("int64")
        cum_sorted = cum_sorted.rename(columns={"game_year": "_year"})
        cum_sorted["_year"] = cum_sorted["_year"].astype("int64")
        by_keys = [by, "_year"]
    else:
        by_keys = [by]

    keep = ["game_date"] + by_keys + feature_cols
    cum_sorted = cum_sorted[keep]

    joined = pd.merge_asof(
        games_sorted,
        cum_sorted,
        by=by_keys,
        left_on=game_date_col,
        right_on="game_date",
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_cum_date"),
    )

    if prefix:
        rename_map = {c: f"{prefix}{c}" for c in feature_cols}
        joined = joined.rename(columns=rename_map)

    drop_cols = [c for c in ("_year", "game_date_cum_date") if c in joined.columns]
    if drop_cols:
        joined = joined.drop(columns=drop_cols)

    joined = joined.sort_values("_orig_idx").drop(columns="_orig_idx").reset_index(drop=True)
    return joined
