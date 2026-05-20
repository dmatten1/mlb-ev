"""Load and tidy MLB outcomes from the raw JSON snapshots written by
``src.ingest.fetch_outcomes``.

The raw layout is one JSON file per *game date*:

    data/raw/outcomes/baseball_mlb/<year>/<YYYY-MM-DD>.json

Each file's top level is a metadata dict with a ``data`` field holding the
list of games. This module flattens that into a tidy DataFrame and
optionally writes a per-season parquet for fast downstream reads.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_RAW_ROOT = Path("data/raw/outcomes/baseball_mlb")
DEFAULT_ROLLUP_ROOT = Path("data/outcomes")

# Columns we surface in the tidy frame. The raw JSON has more fields; this
# is the subset the matchup model actually needs.
TIDY_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "game_datetime",
    "game_type",
    "status",
    "home_id", "home_name",
    "away_id", "away_name",
    "home_score", "away_score",
    "winning_team", "losing_team",
    "winning_pitcher", "losing_pitcher", "save_pitcher",
    "home_probable_pitcher", "away_probable_pitcher",
    "venue_id", "venue_name",
    "doubleheader", "game_num",
)

# MLB-StatsAPI game_type codes used in this codebase.
#   R = regular season    F = wild card (formerly first-round)
#   D = division series   L = league championship   W = world series
#   S = spring training   E = exhibition            A = all-star
REGULAR_SEASON_TYPES: frozenset[str] = frozenset({"R"})
POSTSEASON_TYPES: frozenset[str] = frozenset({"F", "D", "L", "W"})
COUNTABLE_GAME_TYPES: frozenset[str] = REGULAR_SEASON_TYPES | POSTSEASON_TYPES

# Status values where the score on the wire is the final score.
COUNTABLE_STATUSES: frozenset[str] = frozenset({"Final", "Completed Early"})


def _iter_raw_files(root: Path, year: int | None) -> Iterable[Path]:
    if year is None:
        return sorted(root.rglob("*.json"))
    return sorted((root / str(year)).rglob("*.json"))


def load_outcomes_raw(
    year: int | None = None,
    root: Path | str = DEFAULT_RAW_ROOT,
) -> pd.DataFrame:
    """Concatenate every game in every raw JSON under ``root`` (optionally
    filtered to ``year``) into one DataFrame.

    Includes ALL games — postponed, suspended, spring training, etc. The
    caller is responsible for filtering down to what they want via
    ``status``, ``game_type``, etc.
    """
    root = Path(root)
    rows: list[dict] = []
    for p in _iter_raw_files(root, year):
        try:
            doc = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        for g in doc.get("data", []):
            rows.append(g)
    if not rows:
        return pd.DataFrame(columns=list(TIDY_COLUMNS))

    df = pd.DataFrame(rows)
    # Stable column order; tolerate missing fields if the API drops some.
    keep = [c for c in TIDY_COLUMNS if c in df.columns]
    other = [c for c in df.columns if c not in TIDY_COLUMNS]
    return df[keep + other].copy()


def load_outcomes(
    year: int | None = None,
    root: Path | str = DEFAULT_RAW_ROOT,
    *,
    finals_only: bool = True,
    regular_and_postseason_only: bool = True,
) -> pd.DataFrame:
    """Cleaned outcomes ready for analysis / training labels.

    Default filters:
      - ``status`` in ``COUNTABLE_STATUSES`` (drops postponed, suspended, in-progress)
      - ``game_type`` in ``COUNTABLE_GAME_TYPES`` (R + F/D/L/W postseason rounds;
        drops spring, exhibition, all-star)

    Also adds:
      - ``home_win``       bool
      - ``run_diff``       int   (home_score - away_score)
      - ``total_runs``     int
      - ``is_postseason``  bool
    """
    df = load_outcomes_raw(year=year, root=root)
    if df.empty:
        return df

    if finals_only:
        df = df[df["status"].isin(COUNTABLE_STATUSES)].copy()
    if regular_and_postseason_only:
        df = df[df["game_type"].isin(COUNTABLE_GAME_TYPES)].copy()

    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["home_win"] = df["home_score"] > df["away_score"]
    df["run_diff"] = (df["home_score"] - df["away_score"]).astype("Int64")
    df["total_runs"] = (df["home_score"] + df["away_score"]).astype("Int64")
    df["is_postseason"] = df["game_type"].isin(POSTSEASON_TYPES)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Suspended games appear twice — once on the date they were paused
    # (status flips to "Final" on the original date when they finish on a
    # later one) and once on the resumption date. Keep the LATER row so
    # ``game_date`` reflects when the final pitch actually happened.
    df = (
        df.sort_values(["game_id", "game_date"])
        .drop_duplicates(subset=["game_id"], keep="last")
    )
    return df.reset_index(drop=True)


def rollup_to_parquet(
    year: int,
    *,
    raw_root: Path | str = DEFAULT_RAW_ROOT,
    output_root: Path | str = DEFAULT_ROLLUP_ROOT,
    finals_only: bool = True,
    regular_and_postseason_only: bool = True,
) -> Path:
    """Roll the year's raw JSON snapshots into a single tidy parquet."""
    df = load_outcomes(
        year=year,
        root=raw_root,
        finals_only=finals_only,
        regular_and_postseason_only=regular_and_postseason_only,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"outcomes_{year}.parquet"
    df.to_parquet(path, index=False)
    return path
