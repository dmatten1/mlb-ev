"""Pull MLB-StatsAPI boxscore data for historical games.

For each gamePk, ``statsapi.boxscore_data`` returns a dict with:

* ``home.battingOrder``  / ``away.battingOrder``  — list of 9 MLBAM IDs in batting order
* ``home.pitchers``      / ``away.pitchers``      — list of MLBAM IDs in appearance order
                                                    (index 0 is the starter)
* ``home.players``       / ``away.players``       — id -> name/position/stats dict
* ``home.bullpen``       / ``away.bullpen``       — pre-game bullpen roster

We persist the raw dict per game so downstream flattening logic in
``src.features.lineup_loader`` can be iterated without re-hitting the API.
Layout:

    data/raw/boxscores/<year>/<gamePk>.json

CLI:
    # Backfill every game in outcomes_2024.parquet + outcomes_2025.parquet
    python -m src.ingest.fetch_boxscores --years 2024 2025

    # One game
    python -m src.ingest.fetch_boxscores --game-id 776318

    # Re-fetch (override existing files)
    python -m src.ingest.fetch_boxscores --year 2024 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "raw" / "boxscores"

logger = logging.getLogger("ingest.fetch_boxscores")


def fetch_boxscore(game_id: int) -> dict:
    """Wrap ``statsapi.boxscore_data`` so callers don't import statsapi directly."""
    import statsapi

    return statsapi.boxscore_data(game_id)


def boxscore_path(game_id: int, year: int,
                  root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return root / str(year) / f"{game_id}.json"


def save_boxscore(game_id: int, year: int, *,
                  output_root: Path = DEFAULT_OUTPUT_ROOT,
                  force: bool = False) -> tuple[Path, bool]:
    """Fetch one game's boxscore and persist it.

    Returns ``(path, wrote_new)``. ``wrote_new=False`` means the file
    already existed and ``force=False``.
    """
    out = boxscore_path(game_id, year, output_root)
    if out.exists() and not force:
        return out, False
    doc = fetch_boxscore(game_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, default=str)
    return out, True


def _game_ids_for_year(year: int) -> list[int]:
    """Pull the list of gamePks to backfill from the outcomes parquet."""
    from src.features.outcomes_loader import load_outcomes

    df = load_outcomes(year)
    if df.empty:
        return []
    return df["game_id"].astype("Int64").dropna().astype(int).unique().tolist()


def backfill_year(year: int, *, output_root: Path = DEFAULT_OUTPUT_ROOT,
                  force: bool = False, sleep_s: float = 0.0,
                  log_every: int = 100) -> dict[str, int]:
    """Pull boxscores for every game in ``outcomes_{year}.parquet``."""
    game_ids = _game_ids_for_year(year)
    if not game_ids:
        logger.warning(
            "No games found for %d. Run the outcomes backfill first:\n"
            "    python -m src.ingest.fetch_outcomes --start %d-03-01 --end %d-11-15",
            year, year, year,
        )
        return {"requested": 0, "fetched": 0, "skipped": 0, "errors": 0}

    logger.info("Backfilling %d boxscores for %d -> %s",
                len(game_ids), year, output_root)
    started = time.monotonic()
    fetched = skipped = errors = 0
    for i, gid in enumerate(game_ids, start=1):
        try:
            _path, wrote = save_boxscore(gid, year, output_root=output_root, force=force)
            if wrote:
                fetched += 1
            else:
                skipped += 1
        except Exception as e:  # noqa: BLE001 — keep going on a single bad game
            errors += 1
            logger.warning("game_id=%d failed: %s", gid, e)
        if log_every and i % log_every == 0:
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(game_ids) - i) / rate if rate > 0 else 0
            logger.info(
                "  %5d/%d  fetched=%d skipped=%d errors=%d  "
                "rate=%.1f/s  eta=%.0fs",
                i, len(game_ids), fetched, skipped, errors, rate, remaining,
            )
        if sleep_s:
            time.sleep(sleep_s)
    elapsed = time.monotonic() - started
    logger.info(
        "Year %d done in %.0fs | fetched=%d skipped=%d errors=%d",
        year, elapsed, fetched, skipped, errors,
    )
    return {
        "requested": len(game_ids),
        "fetched": fetched,
        "skipped": skipped,
        "errors": errors,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill MLB boxscores per game.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--year", type=int, help="Single season.")
    grp.add_argument("--years", type=int, nargs="+", help="Multiple seasons.")
    grp.add_argument("--game-id", type=int,
                     help="One specific gamePk (writes to data/raw/boxscores/<year-from-game>/).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--force", action="store_true",
                   help="Re-fetch even when a file exists.")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Optional sleep (seconds) between calls.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.game_id:
        # Pull just one game; infer year from the boxscore itself.
        doc = fetch_boxscore(args.game_id)
        year_str = (doc.get("gameId") or "").split("/", 1)[0]
        try:
            year = int(year_str)
        except ValueError:
            logger.error("Could not infer year from gameId=%r", doc.get("gameId"))
            return 1
        out = boxscore_path(args.game_id, year, args.output_root)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, default=str)
        logger.info("Wrote %s", out)
        return 0

    if args.year:
        years = [args.year]
    elif args.years:
        years = args.years
    else:
        logger.error("Pass --year, --years, or --game-id.")
        return 2

    overall = {"requested": 0, "fetched": 0, "skipped": 0, "errors": 0}
    for y in years:
        summary = backfill_year(
            y, output_root=args.output_root,
            force=args.force, sleep_s=args.sleep,
        )
        for k in overall:
            overall[k] += summary[k]
    logger.info("All years done | %s", overall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
