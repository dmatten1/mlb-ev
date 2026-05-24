"""Render today's model slate as markdown for IDE review.

The predict step already writes ``data/predictions/<date>.parquet`` on each
run. This module loads that file (local or S3) and regenerates the companion
``.md`` summary, including the **Full slate** table with run predictions for
every game — not just recommended bets.

Usage::

    python -m src.inference.view_slate
    python -m src.inference.view_slate --date 2026-05-22
    python -m src.inference.view_slate --from-s3
    python -m src.inference.view_slate --game "Yankees"   # filter one matchup
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger("inference.view_slate")

DEFAULT_PREDICTIONS_ROOT = Path("data/predictions")


def _resolve_parquet(
    target: date,
    *,
    predictions_root: Path,
    from_s3: bool,
) -> Path:
    predictions_root.mkdir(parents=True, exist_ok=True)
    pq = predictions_root / f"{target.isoformat()}.parquet"
    if pq.exists() and not from_s3:
        return pq

    bucket = os.environ.get("MLB_EV_S3_BUCKET") or os.environ.get("ODDS_S3_BUCKET")
    if not bucket:
        if pq.exists():
            return pq
        raise FileNotFoundError(
            f"No slate at {pq}. Run `make project` locally, or set "
            "MLB_EV_S3_BUCKET and pass --from-s3."
        )

    prefix = os.environ.get("MLB_EV_PIPELINE_PREFIX", "pipeline/data").strip("/")
    key = f"{prefix}/predictions/{target.isoformat()}.parquet"
    try:
        import boto3
    except ImportError as e:
        raise FileNotFoundError(
            f"No local slate at {pq} and boto3 not installed for S3 pull."
        ) from e

    region = os.environ.get("AWS_REGION", "us-east-1")
    logger.info("Downloading s3://%s/%s", bucket, key)
    boto3.client("s3", region_name=region).download_file(bucket, key, str(pq))
    return pq


def _filter_games(slate: pd.DataFrame, game_query: str | None) -> pd.DataFrame:
    if not game_query or slate.empty:
        return slate
    q = game_query.strip().lower()
    mask = (
        slate["home_name"].astype(str).str.lower().str.contains(q, regex=False)
        | slate["away_name"].astype(str).str.lower().str.contains(q, regex=False)
    )
    out = slate.loc[mask]
    if out.empty:
        names = sorted(
            set(slate["away_name"].astype(str)) | set(slate["home_name"].astype(str))
        )
        raise SystemExit(
            f"No game matching {game_query!r}. On slate: {', '.join(names[:12])}..."
        )
    return out


def render_markdown(
    slate: pd.DataFrame,
    md_path: Path,
    *,
    run_date: str,
    slate_label: str = "cloud slate",
) -> None:
    from src.pipeline.daily_refresh import _write_markdown_summary

    _write_markdown_summary(
        slate, md_path, run_date=run_date, slate_label=slate_label,
    )


def print_compact(slate: pd.DataFrame) -> None:
    """Stdout table for quick terminal / IDE peek."""
    cols = [
        "away_name", "away_runs_pred", "home_name", "home_runs_pred",
        "p_home", "recommended",
    ]
    cols = [c for c in cols if c in slate.columns]
    view = slate[cols].copy()
    if "recommended" in view.columns:
        view["recommended"] = view["recommended"].fillna("—")
    view = view.sort_values(
        ["away_runs_pred", "home_runs_pred"] if "away_runs_pred" in view.columns else cols[0],
        ascending=False,
    )
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(view.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="View model run predictions for today's slate (markdown + optional filter).",
    )
    p.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Slate date (default: today)",
    )
    p.add_argument(
        "--predictions-root",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
    )
    p.add_argument(
        "--from-s3",
        action="store_true",
        help="Pull parquet from S3 if missing or stale locally",
    )
    p.add_argument(
        "--game",
        default="",
        help="Filter to one matchup (substring match on team name)",
    )
    p.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print compact table only; skip writing .md",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(levelname)s %(name)s | %(message)s",
    )

    target = date.fromisoformat(args.date)
    pq_path = _resolve_parquet(
        target,
        predictions_root=args.predictions_root,
        from_s3=args.from_s3,
    )
    slate = pd.read_parquet(pq_path)
    if args.game:
        slate = _filter_games(slate, args.game)

    if args.stdout_only:
        print_compact(slate)
        return 0

    md_path = args.predictions_root / f"{target.isoformat()}.md"
    render_markdown(
        slate, md_path,
        run_date=target.isoformat(),
        slate_label=f"view_slate ({pq_path.name})",
    )
    print(f"Wrote {md_path}")
    print(f"  {len(slate)} games — open in IDE for full run predictions table")
    no_bet = slate[~slate["recommended"].isin(["home", "away"])] if "recommended" in slate.columns else slate
    if "recommended" in slate.columns and not no_bet.empty:
        print(f"  {len(no_bet)} games with no bet (see 'Full slate' section)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
