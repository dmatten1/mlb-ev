"""AWS Lambda entry point for the odds ingestion job.

Configure via environment variables on the Lambda function:
  ODDS_API_KEY       (required) The Odds API key.
  ODDS_S3_BUCKET     (required) Destination bucket for snapshots.
  ODDS_S3_PREFIX     (optional) Defaults to ``raw/odds``.
  ODDS_API_SPORT     (optional) Defaults to ``baseball_mlb``.
  ODDS_API_REGIONS   (optional) Defaults to ``us``.
  ODDS_API_MARKETS   (optional) Defaults to ``h2h``.
  ODDS_API_ODDS_FORMAT (optional) Defaults to ``american``.

The function is intended to be triggered by EventBridge Scheduler on a cron in
America/New_York. It returns a small summary dict that is captured in
CloudWatch Logs and visible in the EventBridge invocation history.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.ingest.fetch_odds import (
    DEFAULT_MARKETS,
    DEFAULT_ODDS_FORMAT,
    DEFAULT_REGIONS,
    DEFAULT_S3_PREFIX,
    DEFAULT_SPORT,
    run_snapshot,
)

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    api_key = os.environ["ODDS_API_KEY"]
    bucket = os.environ["ODDS_S3_BUCKET"]

    summary = run_snapshot(
        api_key=api_key,
        sport=os.getenv("ODDS_API_SPORT", DEFAULT_SPORT),
        regions=os.getenv("ODDS_API_REGIONS", DEFAULT_REGIONS),
        markets=os.getenv("ODDS_API_MARKETS", DEFAULT_MARKETS),
        odds_format=os.getenv("ODDS_API_ODDS_FORMAT", DEFAULT_ODDS_FORMAT),
        local_root=None,
        s3_bucket=bucket,
        s3_prefix=os.getenv("ODDS_S3_PREFIX", DEFAULT_S3_PREFIX),
    )

    logger.info(
        "snapshot saved games=%s remaining=%s destinations=%s",
        summary["game_count"],
        summary["requests_remaining"],
        summary["destinations"],
    )
    return summary
