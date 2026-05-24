"""Lightweight dashboard refresh: live scores only (no predict / odds).

Re-renders ``bet_dashboard.html`` from the bet log and fetches current
scores for pending rows via MLB-StatsAPI (free). Intended for a ~10-minute
EventBridge schedule on the inference Lambda while games are in progress.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.tracking.bet_log import filter_log_by_season, load_log
from src.tracking.dashboard import DEFAULT_DASHBOARD_SEASON_YEAR, DEFAULT_LOG_PATH, render

logger = logging.getLogger("pipeline.scores_refresh")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    full = load_log(args.log_path)
    log = filter_log_by_season(full, args.season_year)
    n_pending = int((log["outcome"].astype(str) == "pending").sum()) if not log.empty else 0
    if n_pending == 0:
        logger.info("No pending bets — skipping dashboard live-score refresh")
        return 0

    out = render(
        log_path=args.log_path,
        out_path=args.out_path,
        season_year=args.season_year,
    )
    logger.info("Live scores dashboard written to %s (%d pending bets)", out, n_pending)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh dashboard live scores only.")
    p.add_argument("--log-path", type=str, default=str(DEFAULT_LOG_PATH))
    p.add_argument("--out-path", type=str, default="")
    p.add_argument("--season-year", type=int, default=DEFAULT_DASHBOARD_SEASON_YEAR)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    if not args.out_path:
        from src.tracking.dashboard import DEFAULT_OUT

        args.out_path = str(DEFAULT_OUT)
    return args


if __name__ == "__main__":
    sys.exit(main())
