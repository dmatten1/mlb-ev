"""Pull full-season MLB Statcast pitch-level data for training/feature use.

``pybaseball.statcast(start, end)`` chunks under the hood and caches each
date locally. We wrap it once per season into a single column-pruned
parquet so downstream feature code can ``pd.read_parquet(...)`` and operate
without hitting Baseball Savant again.

The kept columns are everything we need for:

* SIERA / xFIP / FIP (events, bb_type)
* xwOBA (woba_value / woba_denom, estimated_woba_using_speedangle)
* Barrel% (launch_speed_angle == 6)
* Pitch run values (delta_run_exp + pitch_type)
* Lineup / matchup metadata (game_pk, batter, pitcher, stand, p_throws)

CLI:
    python -m src.ingest.fetch_statcast_history --years 2024 2025
    python -m src.ingest.fetch_statcast_history --year 2024 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "raw" / "statcast"

# Approximate MLB season windows. We pull regular-season + postseason in
# one go; Statcast covers both. Spring training is excluded.
SEASON_RANGES: dict[int, tuple[str, str]] = {
    # 2022 had a lockout-delayed Opening Day (April 7).
    2022: ("2022-04-07", "2022-11-05"),
    # 2023: pitch clock + shift restrictions + bigger bases introduced.
    # Regime change vs. 2022 — league run-scoring ticked up materially.
    2023: ("2023-03-30", "2023-11-01"),
    2024: ("2024-03-20", "2024-11-01"),
    2025: ("2025-03-18", "2025-11-05"),
    2026: ("2026-03-18", "2026-11-15"),  # extend as the season progresses
}

# Columns we keep from the raw pitch-level Statcast pull. ~24 columns
# vs. the native ~90+ keeps the parquet under ~150 MB per season.
KEEP_COLS: tuple[str, ...] = (
    "game_pk", "game_date", "game_year", "game_type",
    "pitcher", "batter",
    "stand", "p_throws",
    "events", "bb_type", "description",
    "pitch_type", "pitch_name",
    "delta_run_exp",
    "woba_value", "woba_denom",
    "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "home_team", "away_team",
    "balls", "strikes",
)

logger = logging.getLogger("ingest.fetch_statcast_history")


def pull_season(year: int, *, verbose: bool = False):
    """Pull one full season of Statcast pitches via pybaseball."""
    import pybaseball as pb

    if year not in SEASON_RANGES:
        raise ValueError(
            f"No season range configured for {year}. "
            f"Known years: {sorted(SEASON_RANGES.keys())}. "
            f"Add an entry to SEASON_RANGES."
        )
    start, end = SEASON_RANGES[year]
    logger.info("Pulling Statcast for %d: %s -> %s", year, start, end)
    df = pb.statcast(start, end, verbose=verbose)

    # Column prune (tolerate any columns that have vanished upstream).
    available = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        logger.warning("Missing %d expected columns: %s", len(missing), missing)
    return df[available].copy()


def parquet_path(year: int, root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return root / f"statcast_{year}.parquet"


def save_season(year: int, *, output_root: Path = DEFAULT_OUTPUT_ROOT,
                force: bool = False, verbose: bool = False) -> Path:
    """Pull and save one season; skip the pull if the parquet exists and
    ``force=False``.
    """
    out = parquet_path(year, output_root)
    if out.exists() and not force:
        logger.info("Skipping %d — parquet already exists at %s. Pass --force to repull.", year, out)
        return out
    df = pull_season(year, verbose=verbose)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows, %.1f MB)",
                out, len(df), out.stat().st_size / 1e6)
    return out


def update_season_incremental(year: int, *,
                              output_root: Path = DEFAULT_OUTPUT_ROOT,
                              end_date: str | None = None,
                              verbose: bool = False) -> Path:
    """Append any *new* days to an existing season parquet without re-pulling
    the whole thing.

    Reads the current parquet, finds the most recent ``game_date``, pulls
    statcast from the next day through ``end_date`` (default today), and
    appends + dedupes by ``(game_pk, batter, pitcher, balls, strikes,
    description)``.

    If no parquet exists yet, falls back to ``save_season`` (full pull).
    Designed to be safe to run daily: yields zero rows if the season is
    already up to date.
    """
    import pandas as pd
    import pybaseball as pb
    from datetime import date, timedelta

    out = parquet_path(year, output_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        logger.info("No existing parquet for %d — doing full season pull.", year)
        return save_season(year, output_root=output_root, force=True, verbose=verbose)

    end_str = end_date or date.today().isoformat()
    existing = pd.read_parquet(out)
    if existing.empty:
        return save_season(year, output_root=output_root, force=True, verbose=verbose)

    last_date = pd.to_datetime(existing["game_date"]).max().date()
    start_date = last_date + timedelta(days=1)
    if start_date > date.fromisoformat(end_str):
        logger.info("Statcast %d is already current through %s — nothing to fetch.",
                    year, last_date)
        return out
    logger.info("Incremental statcast pull for %d: %s -> %s",
                year, start_date, end_str)
    new_df = pb.statcast(start_date.isoformat(), end_str, verbose=verbose)
    if new_df is None or new_df.empty:
        logger.info("No new statcast rows in range %s..%s for %d.",
                    start_date, end_str, year)
        return out
    available = [c for c in KEEP_COLS if c in new_df.columns]
    new_df = new_df[available].copy()

    combined = pd.concat([existing, new_df], ignore_index=True)
    dedup_keys = [c for c in
                  ("game_pk", "batter", "pitcher", "balls", "strikes", "description")
                  if c in combined.columns]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    after = len(combined)
    combined.to_parquet(out, index=False)
    logger.info(
        "Updated %s: %d existing + %d new -> %d total (%d dupes dropped, %.1f MB)",
        out, len(existing), len(new_df), after, before - after,
        out.stat().st_size / 1e6,
    )
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull full-season Statcast history.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--year", type=int, help="Single season.")
    grp.add_argument("--years", type=int, nargs="+",
                     help="Multiple seasons (e.g. --years 2024 2025).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--force", action="store_true",
                   help="Repull even if the parquet already exists.")
    p.add_argument("--verbose", action="store_true",
                   help="Print pybaseball's per-day progress bar.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.year:
        years = [args.year]
    elif args.years:
        years = args.years
    else:
        logger.error("Must pass --year or --years.")
        return 2

    for y in years:
        save_season(y, output_root=args.output_root,
                    force=args.force, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
