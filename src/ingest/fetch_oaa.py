"""Pull per-player Outs Above Average (OAA) from Baseball Savant via pybaseball.

Savant publishes OAA at the player×position×season grain. We pull every
non-catcher position (pybaseball doesn't support catcher OAA — that's
catcher framing/blocking, which is a different metric anyway) and stash
the result as one parquet per season.

The handedness splits matter for the matchup engine: a SS with
``outs_above_average_rhh=+5`` and ``outs_above_average_lhh=-3`` should
apply different defensive penalties depending on which side of the
opposing lineup is at the plate. We keep both columns.

Position codes (pybaseball convention, mirrors MLB Stats API positionId):

    3 = 1B
    4 = 2B
    5 = 3B
    6 = SS
    7 = LF
    8 = CF
    9 = RF
    (2 = C — not supported; 1 = P — not OAA-relevant)

CLI:
    python -m src.ingest.fetch_oaa --year 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pybaseball as pb

DEFAULT_OUTPUT_ROOT = Path("data/oaa")

# Minimum fielding attempts for a player-position row to be retained.
# Players below this threshold are dropped from the parquet entirely
# and will be treated as league-average (0 OAA) by the lookup, which
# avoids letting a 4-play call-up swing a team's defense estimate.
# 50 attempts is roughly equivalent to:
#   * Infielders: ~12-15 games of regular play
#   * Outfielders: ~18-22 games of regular play
# (Infielders see more chances per game than outfielders, so the same
# attempt threshold corresponds to fewer games at IF and more at OF.)
# Tune via the CLI ``--min-att`` flag if needed.
DEFAULT_MIN_ATT: int = 50

POSITION_CODE_TO_ABBREV: dict[int, str] = {
    3: "1B", 4: "2B", 5: "3B", 6: "SS",
    7: "LF", 8: "CF", 9: "RF",
}

KEEP_COLS: tuple[str, ...] = (
    "player_id",
    "primary_pos_formatted",
    "display_team_name",
    "year",
    "outs_above_average",
    "outs_above_average_rhh",
    "outs_above_average_lhh",
    "fielding_runs_prevented",
    # diff_success_rate is "OAA / attempts" (rounded to whole %); kept so
    # we can derive attempts. Savant doesn't publish attempts directly.
    "diff_success_rate_formatted",
)

logger = logging.getLogger("ingest.fetch_oaa")


def fetch_oaa_one_position(year: int, pos_code: int,
                           *, min_att: int | str = DEFAULT_MIN_ATT) -> pd.DataFrame:
    """One ``(year, position)`` pull. ``min_att=50`` (default) drops the
    long tail of 4-play call-ups; pass ``min_att=1`` to keep every
    fielder or ``'q'`` for Savant's "qualified" definition.

    Adds a derived ``attempts`` column:
    ``attempts = OAA / (diff_success_rate / 100)`` — Savant doesn't
    publish fielding attempts directly but they're recoverable from the
    success-rate diff. Rounded to whole %, so the value is approximate
    (~5-10% relative error for typical players). For rows where the
    formatted diff is "0%" (OAA was within 0.5%-of-attempts of expected),
    the derived value is NaN and we fall back to the per-position median
    at lookup time.
    """
    df = pb.statcast_outs_above_average(year, pos_code, min_att=min_att)
    df = df.copy()
    df["position_abbrev"] = POSITION_CODE_TO_ABBREV[pos_code]
    df["position_code"] = pos_code
    df["attempts"] = _derive_attempts(df)
    keep = [c for c in (*KEEP_COLS, "position_abbrev", "position_code", "attempts")
            if c in df.columns]
    return df[keep]


def _derive_attempts(df: pd.DataFrame) -> pd.Series:
    """``attempts = OAA / (diff_success_rate / 100)`` element-wise.

    Returns NaN whenever either OAA or diff is zero (we can't divide):
    these rows fall back to the per-position median at lookup time.
    """
    diff_str = df["diff_success_rate_formatted"].astype(str).str.rstrip("%")
    diff_pct = pd.to_numeric(diff_str, errors="coerce") / 100.0
    oaa = pd.to_numeric(df["outs_above_average"], errors="coerce")
    # Only derive when BOTH OAA and diff are nonzero — otherwise the
    # division is either 0/x = 0 (degenerate "league-average" player)
    # or x/0 = inf.
    mask = (oaa.abs() > 0) & (diff_pct.abs() > 0)
    attempts = (oaa.abs() / diff_pct.abs()).where(mask, other=pd.NA)
    return attempts.round().astype("Int64")


def fetch_oaa_year(year: int, *,
                   min_att: int | str = DEFAULT_MIN_ATT,
                   sleep_sec: float = 0.5) -> pd.DataFrame:
    """All non-catcher positions for one year, concatenated."""
    pieces = []
    for code in POSITION_CODE_TO_ABBREV.keys():
        try:
            piece = fetch_oaa_one_position(year, code, min_att=min_att)
            pieces.append(piece)
            logger.info("Pulled %d rows for year=%d position=%s (code %d)",
                        len(piece), year, POSITION_CODE_TO_ABBREV[code], code)
        except Exception as exc:  # pybaseball flakes; report and continue
            logger.warning("Skipped year=%d position=%d: %s", year, code, exc)
        time.sleep(sleep_sec)
    if not pieces:
        raise RuntimeError(f"No OAA data pulled for {year}")
    df = pd.concat(pieces, ignore_index=True)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    return df


def save_oaa(year: int, *,
             min_att: int | str = DEFAULT_MIN_ATT,
             output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    df = fetch_oaa_year(year, min_att=min_att)
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f"oaa_{year}.parquet"
    df.to_parquet(out, index=False)
    logger.info("Wrote %s  (%d player-position rows)", out, len(df))
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull Savant per-player OAA.")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--min-att", default=str(DEFAULT_MIN_ATT),
                   help=f"Minimum fielding attempts ('q' for qualified, "
                        f"or an integer threshold; default {DEFAULT_MIN_ATT}).")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    raw = args.min_att
    min_att: int | str = raw if raw == "q" else int(raw)
    save_oaa(args.year, min_att=min_att, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
