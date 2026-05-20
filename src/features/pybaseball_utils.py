"""Lightweight helpers for exploring pybaseball data.

Functions:
  enable_cache()        Turn on pybaseball's HTTP cache so repeat queries are free.
  save_raw(df, name)    Persist a DataFrame as Parquet under data/raw/pybaseball/.
  load_raw(name)        Read it back; resolves the latest snapshot when given a prefix.
  list_raw(prefix='')   Glob saved pulls.
  show(df, n=5)         Pretty preview of a DataFrame in a notebook.

Design notes:
  * Parquet is used (not CSV/JSON) because pybaseball outputs are wide and we want
    typed, fast reloads. pyarrow is the engine.
  * Filenames include a UTC timestamp so successive pulls of the same query don't
    clobber each other. ``load_raw('pitching_stats_2025')`` returns the newest.
  * Caching is pybaseball's own (``~/.pybaseball``) — survives across notebook
    restarts so you don't re-hit FanGraphs while iterating.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PYBASEBALL_DIR = REPO_ROOT / "data" / "raw" / "pybaseball"


def enable_cache() -> None:
    """Enable pybaseball's local disk cache (idempotent).

    Silently no-ops if the cache directory is read-only (e.g. CI / sandbox).
    Cache misses just mean another upstream call; the rest of the code works.
    """
    try:
        from pybaseball import cache

        cache.enable()
    except (OSError, PermissionError):
        pass


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def save_raw(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame to ``data/raw/pybaseball/<name>_<utc>.parquet``."""
    RAW_PYBASEBALL_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_PYBASEBALL_DIR / f"{name}_{_utc_timestamp()}.parquet"
    df.to_parquet(target, index=False)
    return target


def list_raw(prefix: str = "") -> list[Path]:
    """List saved pulls, optionally filtered by filename prefix. Oldest first."""
    if not RAW_PYBASEBALL_DIR.exists():
        return []
    return sorted(RAW_PYBASEBALL_DIR.glob(f"{prefix}*.parquet"))


def load_raw(name_or_prefix: str) -> pd.DataFrame:
    """Load a saved pull.

    Accepts either:
      * an exact filename (``pitching_stats_2025_2026-05-12T16-30-00Z.parquet``)
      * a prefix (``pitching_stats_2025``) — returns the most recent match.
    """
    exact = RAW_PYBASEBALL_DIR / name_or_prefix
    if exact.exists():
        return pd.read_parquet(exact)
    matches = list_raw(name_or_prefix)
    if not matches:
        raise FileNotFoundError(f"No pull matching '{name_or_prefix}' under {RAW_PYBASEBALL_DIR}")
    return pd.read_parquet(matches[-1])


def show(df: pd.DataFrame, n: int = 5) -> Any:
    """Pretty preview: shape + dtypes + head, friendly for notebooks."""
    print(f"shape={df.shape}  columns={len(df.columns)}")
    return df.head(n)
