"""Fetch MLB game outcomes (final scores, lineups metadata) from MLB-StatsAPI.

MLB-StatsAPI is the official statsapi.mlb.com wrapper — unlike pybaseball /
FanGraphs, it's a documented JSON endpoint and not at risk of being blocked.
We ingest one JSON record per *game date* so the file shape mirrors the odds
pipeline (date-partitioned raw JSON, one file per "capture").

Destinations:
  - Local filesystem (default): writes ``data/raw/outcomes/...``.
  - S3: when ``OUTCOMES_S3_BUCKET`` is set, uploads to that bucket under the
    same relative key.

CLI:
    # yesterday (nightly cron use case)
    python -m src.ingest.fetch_outcomes

    # single explicit date
    python -m src.ingest.fetch_outcomes --date 2025-08-14

    # range (backfill)
    python -m src.ingest.fetch_outcomes --start 2024-03-28 --end 2025-10-31

The module is also imported by ``outcomes_lambda_handler.handler``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # Lambda package omits python-dotenv.
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

DEFAULT_SPORT = "baseball_mlb"
DEFAULT_S3_PREFIX = "raw/outcomes"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "raw" / "outcomes"

logger = logging.getLogger("ingest.fetch_outcomes")


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_outcomes_for_date(target_date: date) -> list[dict[str, Any]]:
    """Return MLB-StatsAPI's ``schedule()`` payload for a single date.

    Empty list for off-season / no-game days. Status filtering (Final vs
    Postponed) happens downstream — we persist the raw response so the
    parquet rollup can re-derive whatever subset it needs.
    """
    import statsapi  # imported lazily so non-ingest code paths don't pay the import cost

    iso = target_date.isoformat()
    return statsapi.schedule(start_date=iso, end_date=iso, sportId=1)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_outcomes_record(
    games: list[dict[str, Any]],
    *,
    target_date: date,
    sport: str = DEFAULT_SPORT,
) -> dict[str, Any]:
    """Wrap a day's games with capture metadata."""
    return {
        "fetched_at_utc": _utc_timestamp(),
        "game_date": target_date.isoformat(),
        "sport": sport,
        "game_count": len(games),
        "status_counts": _count_statuses(games),
        "data": games,
    }


def _count_statuses(games: list[dict[str, Any]]) -> dict[str, int]:
    """Tally games by ``status`` for at-a-glance auditing."""
    out: dict[str, int] = {}
    for g in games:
        s = g.get("status") or "Unknown"
        out[s] = out.get(s, 0) + 1
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def outcomes_relative_key(target_date: date, *, sport: str = DEFAULT_SPORT) -> str:
    """Relative path: ``<sport>/<year>/<YYYY-MM-DD>.json``."""
    return f"{sport}/{target_date.year}/{target_date.isoformat()}.json"


def write_local_outcomes(
    record: dict[str, Any],
    *,
    output_root: Path,
    target_date: date,
    sport: str = DEFAULT_SPORT,
) -> Path:
    rel = outcomes_relative_key(target_date, sport=sport)
    target = output_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    return target


def upload_s3_outcomes(
    record: dict[str, Any],
    *,
    bucket: str,
    prefix: str,
    target_date: date,
    sport: str = DEFAULT_SPORT,
) -> str:
    """Upload one day's outcomes to S3; return s3:// URI."""
    import boto3  # lazy: local runs don't need boto3 installed

    rel = outcomes_relative_key(target_date, sport=sport)
    key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
    body = json.dumps(record, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_outcomes_ingest(
    target_date: date,
    *,
    sport: str = DEFAULT_SPORT,
    local_root: Path | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str = DEFAULT_S3_PREFIX,
    skip_empty: bool = True,
) -> dict[str, Any]:
    """Pull, wrap, and persist one date's worth of game outcomes.

    Returns a summary dict (game_count, destinations, status_counts).
    """
    games = fetch_outcomes_for_date(target_date)
    record = build_outcomes_record(games, target_date=target_date, sport=sport)

    destinations: list[str] = []
    if skip_empty and not games:
        logger.debug("%s: no games, skipping write", target_date)
    else:
        if s3_bucket:
            uri = upload_s3_outcomes(
                record,
                bucket=s3_bucket,
                prefix=s3_prefix,
                target_date=target_date,
                sport=sport,
            )
            destinations.append(uri)
        if local_root is not None:
            path = write_local_outcomes(
                record,
                output_root=local_root,
                target_date=target_date,
                sport=sport,
            )
            destinations.append(str(path))

    return {
        "game_date": target_date.isoformat(),
        "game_count": record["game_count"],
        "status_counts": record["status_counts"],
        "destinations": destinations,
    }


def run_backfill(
    start_date: date,
    end_date: date,
    *,
    sport: str = DEFAULT_SPORT,
    local_root: Path | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str = DEFAULT_S3_PREFIX,
    skip_empty: bool = True,
) -> dict[str, Any]:
    """Pull outcomes for every date in ``[start_date, end_date]`` inclusive.

    Returns a summary covering the entire range.
    """
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} < start_date {start_date}")

    cur = start_date
    total_games = 0
    days_written = 0
    days_skipped = 0
    days_processed = 0
    while cur <= end_date:
        summary = run_outcomes_ingest(
            cur,
            sport=sport,
            local_root=local_root,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            skip_empty=skip_empty,
        )
        total_games += summary["game_count"]
        days_processed += 1
        if summary["destinations"]:
            days_written += 1
        else:
            days_skipped += 1
        if days_processed % 30 == 0 or summary["game_count"] > 0:
            logger.info(
                "%s | games=%2d  cum_games=%5d  written=%4d  skipped=%4d",
                cur, summary["game_count"], total_games, days_written, days_skipped,
            )
        cur += timedelta(days=1)

    return {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "days_processed": days_processed,
        "days_written": days_written,
        "days_skipped_empty": days_skipped,
        "total_games": total_games,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _yesterday_utc() -> date:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB game outcomes.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--date",
        help="Single ISO date (YYYY-MM-DD). Default: yesterday (UTC).",
    )
    grp.add_argument(
        "--start",
        help="Start ISO date for a backfill (inclusive). Requires --end.",
    )
    parser.add_argument(
        "--end",
        help="End ISO date for a backfill (inclusive). Requires --start.",
    )
    parser.add_argument("--sport", default=os.getenv("OUTCOMES_SPORT", DEFAULT_SPORT))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("OUTCOMES_OUTPUT_DIR", str(DEFAULT_LOCAL_ROOT))),
        help="Local directory to write outcomes into. Pass '' to disable local writes.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("OUTCOMES_S3_BUCKET"),
        help="If set, also upload outcomes to this S3 bucket.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.getenv("OUTCOMES_S3_PREFIX", DEFAULT_S3_PREFIX),
    )
    parser.add_argument(
        "--keep-empty-days",
        action="store_true",
        help="Write empty JSON markers for days with no games (default: skip).",
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def _parse_iso(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    local_root: Path | None = args.output_dir if str(args.output_dir) else None
    skip_empty = not args.keep_empty_days

    if args.start or args.end:
        if not (args.start and args.end):
            logger.error("--start and --end must both be provided for a backfill.")
            return 2
        start = _parse_iso(args.start)
        end = _parse_iso(args.end)
        logger.info(
            "Backfilling outcomes | %s -> %s  local=%s  s3=%s",
            start, end, local_root, args.s3_bucket or "(none)",
        )
        result = run_backfill(
            start, end,
            sport=args.sport,
            local_root=local_root,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            skip_empty=skip_empty,
        )
        logger.info(
            "Backfill done | days=%d written=%d skipped=%d games=%d",
            result["days_processed"], result["days_written"],
            result["days_skipped_empty"], result["total_games"],
        )
        return 0

    target = _parse_iso(args.date) if args.date else _yesterday_utc()
    logger.info(
        "Fetching outcomes | date=%s local=%s s3=%s",
        target, local_root, args.s3_bucket or "(none)",
    )
    summary = run_outcomes_ingest(
        target,
        sport=args.sport,
        local_root=local_root,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        skip_empty=skip_empty,
    )
    logger.info(
        "Done | date=%s games=%d destinations=%s statuses=%s",
        summary["game_date"], summary["game_count"],
        summary["destinations"], summary["status_counts"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
