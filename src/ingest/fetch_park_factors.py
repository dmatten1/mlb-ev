"""Pull handedness-split Statcast park factors from Baseball Savant.

Savant publishes its park factors at:
    https://baseballsavant.mlb.com/leaderboard/statcast-park-factors

There's no public API — the page embeds the leaderboard as an inline
``var data = [...]`` JSON literal. We fetch the HTML, extract that array
with a regex, and serialize to parquet.

What we pull (per ``year`` query, ``rolling`` window, ``batSide``):

* ``index_xwobacon`` — xwOBA on contact. This is the multiplier we apply
  to the BIP portion of a hitter's xwOBA, since true xwOBA from Statcast
  is calibrated league-wide (EV+LA -> xwOBA) and therefore park-neutral
  on contact.
* ``index_runs`` — composite run factor. Useful for whole-game runs
  adjustment if we ever want to apply park to the model's runs output.
* ``index_hr / 2b / 3b`` — hit-type factors. Kept for future use if we
  decompose xwOBA into hit-type shares.

Output: ``data/park_factors/park_factors.parquet`` — one row per
``(year, year_range, bat_side, venue_id)`` with all the index columns.

CLI:
    python -m src.ingest.fetch_park_factors --year 2024 --rolling 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_ROOT = Path("data/park_factors")
SAVANT_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=year&year={year}&rolling={rolling}"
    "&stat=index_wOBA&batSide={bat_side}&parks=mlb"
)

# Index columns we keep from each pull. xwobacon is the one we'll apply.
INDEX_COLS: tuple[str, ...] = (
    "index_xwobacon", "index_runs",
    "index_hr", "index_2b", "index_3b", "index_1b",
    "index_woba", "index_wobacon", "index_bb", "index_so",
)
META_COLS: tuple[str, ...] = (
    "venue_id", "venue_name", "main_team_id", "name_display_club",
    "n_pa", "year_range",
)

logger = logging.getLogger("ingest.fetch_park_factors")


def _fetch_one(year: int, bat_side: str, rolling: int) -> list[dict]:
    """Hit Savant and pull the embedded ``var data = [...]`` blob."""
    url = SAVANT_URL.format(year=year, rolling=rolling, bat_side=bat_side)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode()
    m = re.search(r"var\s+data\s*=\s*(\[\{.*?\}\]);", html, re.DOTALL)
    if not m:
        raise RuntimeError(
            f"Could not find embedded data array for year={year} side={bat_side}. "
            "Savant page structure may have changed."
        )
    return json.loads(m.group(1))


def fetch_park_factors(year: int, *, rolling: int = 3) -> pd.DataFrame:
    """Pull both R and L tables for ``year`` and stack them.

    Returns a tidy DataFrame with ``bat_side``, the index columns, and
    venue/team identifiers. Numeric index columns are coerced to int (Savant
    serves them as strings).
    """
    pieces = []
    for side in ("R", "L"):
        rows = _fetch_one(year, side, rolling)
        df = pd.DataFrame(rows)
        df["bat_side"] = side
        pieces.append(df)
        logger.info("Pulled %d rows for year=%d side=%s rolling=%d",
                    len(df), year, side, rolling)
    df = pd.concat(pieces, ignore_index=True)
    # Coerce numeric columns
    for c in (*INDEX_COLS, "n_pa", "venue_id", "main_team_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = ["bat_side", *META_COLS, *INDEX_COLS]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["fetched_year"] = year
    return df


def save_park_factors(year: int, *, rolling: int = 3,
                      output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    df = fetch_park_factors(year, rolling=rolling)
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f"park_factors_{year}_rolling{rolling}.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote %s  (%d rows, %d venues)",
                out, len(df), df["venue_id"].nunique())
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull Savant park factors.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--rolling", type=int, default=3,
                   help="Rolling window size in years (Savant default: 3).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    save_park_factors(args.year, rolling=args.rolling, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
