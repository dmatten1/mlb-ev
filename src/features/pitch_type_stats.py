"""Point-in-time pitch-type run values and mix from Statcast.

For each (player, date) we track cumulative pitch counts and
``delta_run_exp`` sums per ``pitch_type``, then derive:

* **pitch mix** — share of pitches (pitcher) or pitches seen (batter)
  by type.
* **pitch run value** — mean ``delta_run_exp`` per pitch × 100
  (runs per 100 pitches of that type).

Downstream ``matchup.expected_pitch_run_value_matchup`` blends pitcher
and hitter RV per type (50/50) weighted by the *pitcher's* mix.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Statcast pitch_type codes we materialize as columns (stable schema).
CANONICAL_PITCH_TYPES: tuple[str, ...] = (
    "FF", "SI", "FC", "SL", "CH", "CU", "FS", "ST", "KC", "SV",
    "KN", "CS", "SC", "EP", "PO", "FO", "UN",
)

MIN_PITCHES_PER_TYPE: int = 20
MIN_TOTAL_PITCHES_MIX: int = 80


def pitch_type_cumulative_column_names() -> list[str]:
    """Columns produced by ``merge_pitch_type_into_rates``."""
    cols: list[str] = []
    for pt in CANONICAL_PITCH_TYPES:
        cols.append(f"{pt}_n_cum")
        cols.append(f"{pt}_dre_cum")
    return cols


def _pitch_type_daily_long(statcast: pd.DataFrame, group: str) -> pd.DataFrame:
    if group not in {"pitcher", "batter"}:
        raise ValueError(f"group must be 'pitcher' or 'batter', got {group!r}")
    sub = statcast.loc[statcast["pitch_type"].notna()].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["player_id", "game_year", "game_date", "pitch_type", "n", "dre"]
        )
    sub["pitch_type"] = sub["pitch_type"].astype(str)
    sub["dre"] = pd.to_numeric(sub["delta_run_exp"], errors="coerce").fillna(0.0)
    daily = (
        sub.groupby([group, "game_year", "game_date", "pitch_type"], as_index=False)
        .agg(n=("pitch_type", "count"), dre=("dre", "sum"))
    )
    return daily.rename(columns={group: "player_id"})


def _pivot_pitch_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Wide daily frame: ``{PT}_n``, ``{PT}_dre`` per player-day."""
    idx = ["player_id", "game_year", "game_date"]
    if daily.empty:
        out = pd.DataFrame(columns=idx)
        for pt in CANONICAL_PITCH_TYPES:
            out[f"{pt}_n"] = pd.Series(dtype=float)
            out[f"{pt}_dre"] = pd.Series(dtype=float)
        return out
    wide = daily.pivot_table(
        index=idx, columns="pitch_type", values=["n", "dre"],
        aggfunc="sum", fill_value=0,
    )
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [f"{pt}_{stat}" for stat, pt in wide.columns]
    wide = wide.reset_index()
    for pt in CANONICAL_PITCH_TYPES:
        for suffix in ("_n", "_dre"):
            col = f"{pt}{suffix}"
            if col not in wide.columns:
                wide[col] = 0.0
    return wide


def build_pitch_type_cumulative(statcast: pd.DataFrame, group: str) -> pd.DataFrame:
    """Season-to-date running pitch-type counts and DRE sums."""
    daily = _pitch_type_daily_long(statcast, group)
    wide = _pivot_pitch_daily(daily)
    if wide.empty:
        return wide
    wide = wide.sort_values(["player_id", "game_year", "game_date"])
    count_cols = [f"{pt}_n" for pt in CANONICAL_PITCH_TYPES]
    dre_cols = [f"{pt}_dre" for pt in CANONICAL_PITCH_TYPES]
    cum_n = wide.groupby(["player_id", "game_year"])[count_cols].cumsum()
    cum_n.columns = [f"{pt}_n_cum" for pt in CANONICAL_PITCH_TYPES]
    cum_dre = wide.groupby(["player_id", "game_year"])[dre_cols].cumsum()
    cum_dre.columns = [f"{pt}_dre_cum" for pt in CANONICAL_PITCH_TYPES]
    wide = pd.concat(
        [wide[["player_id", "game_year", "game_date"]], cum_n, cum_dre],
        axis=1,
    )
    keep = ["player_id", "game_year", "game_date", *pitch_type_cumulative_column_names()]
    return wide[keep]


def build_pitch_type_rolling(
    statcast: pd.DataFrame,
    group: str,
    *,
    window_days: int = 30,
) -> pd.DataFrame:
    """Calendar rolling window — same column names as cumulative."""
    daily = _pitch_type_daily_long(statcast, group)
    wide = _pivot_pitch_daily(daily)
    if wide.empty:
        return wide
    wide = wide.sort_values(["player_id", "game_date"])
    count_cols = [f"{pt}_n" for pt in CANONICAL_PITCH_TYPES]
    dre_cols = [f"{pt}_dre" for pt in CANONICAL_PITCH_TYPES]
    all_cols = count_cols + dre_cols
    indexed = wide.set_index("game_date")
    rolled = (
        indexed.groupby("player_id")[all_cols]
        .rolling(f"{window_days}D")
        .sum()
        .reset_index()
    )
    rolled["game_year"] = rolled["game_date"].dt.year.astype("int64")
    rename = {}
    for pt in CANONICAL_PITCH_TYPES:
        rename[f"{pt}_n"] = f"{pt}_n_cum"
        rename[f"{pt}_dre"] = f"{pt}_dre_cum"
    rolled = rolled.rename(columns=rename)
    keep = ["player_id", "game_year", "game_date", *pitch_type_cumulative_column_names()]
    return rolled[keep].reset_index(drop=True)


def merge_pitch_type_into_rates(
    statcast: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    group: str,
    rolling_window_days: int | None = None,
) -> pd.DataFrame:
    """Left-merge pitch-type columns onto a cumulative or rolling rates frame."""
    if rolling_window_days is not None:
        pt = build_pitch_type_rolling(
            statcast, group, window_days=rolling_window_days,
        )
    else:
        pt = build_pitch_type_cumulative(statcast, group)
    if pt.empty:
        return rates
    return rates.merge(
        pt, on=["player_id", "game_year", "game_date"], how="left",
    )


def _cell_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return default
    return float(v)


def pitch_mix_from_row(
    row: pd.Series,
    *,
    prefix: str = "",
    suffix: str = "",
    min_total: int = MIN_TOTAL_PITCHES_MIX,
) -> dict[str, float]:
    """Pitch-type frequency shares from cumulative count columns."""
    counts: dict[str, float] = {}
    total = 0.0
    for pt in CANONICAL_PITCH_TYPES:
        n = _cell_float(row, f"{prefix}{pt}_n_cum{suffix}")
        if n > 0:
            counts[pt] = n
            total += n
    if total < min_total:
        return {}
    return {pt: n / total for pt, n in counts.items()}


def pitch_run_values_from_row(
    row: pd.Series,
    *,
    prefix: str = "",
    suffix: str = "",
    min_per_type: int = MIN_PITCHES_PER_TYPE,
) -> dict[str, float]:
    """Per pitch-type run value (runs per 100 pitches)."""
    out: dict[str, float] = {}
    for pt in CANONICAL_PITCH_TYPES:
        n = _cell_float(row, f"{prefix}{pt}_n_cum{suffix}")
        dre = _cell_float(row, f"{prefix}{pt}_dre_cum{suffix}")
        if n >= min_per_type:
            out[pt] = (dre / n) * 100.0
    return out
