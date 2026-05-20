"""Lightweight recurring refresh for a live tracker + dashboard.

Runs every time you schedule it (see Makefile / Docker / launchd) and:

1. **Outcomes** — recent final scores (same lookback as full refresh).
2. **Schedule** — today’s games + short lookahead.
3. **Predict** — tonight’s slate + odds → ``data/predictions/<date>.*`` and
   ``bet_log.log_recommendations`` for the tracker.
4. **Track** — CLV + outcome reconciliation + static HTML dashboard.

**Skips** heavy steps (boxscores loop, statcast, OAA, lineups rollup, full
feature rebuild). Run a full :mod:`src.pipeline.daily_refresh` at least once
per day so training parquets stay current.

**Model cache:** loads ``data/models/runs_model_bullpen_cached.pkl`` when
present so hourly runs do not re-fit Ridge. A full ``daily_refresh`` retrains
by default and overwrites that pickle. Delete the cache if you change
``BULLPEN_FEATURE_COLS`` or training seasons.

CLI::

    python -m src.pipeline.live_refresh
    python -m src.pipeline.live_refresh --year 2026 --fail-fast

Environment: same as ``daily_refresh`` (AWS creds for S3 odds if used, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from src.pipeline.daily_refresh import (
    DEFAULT_PREDICTIONS_ROOT,
    run_refresh,
)

logger = logging.getLogger("pipeline.live_refresh")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Light refresh: outcomes + schedule + predict + dashboard "
        "(no statcast / full feature rebuild).",
    )
    p.add_argument("--year", type=int, default=date.today().year)
    p.add_argument("--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    logger.info(
        "Live refresh starting | year=%d (cached model, light data steps)",
        args.year,
    )
    result = run_refresh(
        args.year,
        do_outcomes=True,
        do_boxscores=False,
        do_statcast=False,
        do_oaa=False,
        do_lineups=False,
        do_features=False,
        do_schedule=True,
        do_predict=True,
        do_track=True,
        predictions_root=args.predictions_root,
        predict_use_cached_model=True,
        fail_fast=args.fail_fast,
    )
    n_failed = sum(1 for s in result.steps if not s.ok)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
