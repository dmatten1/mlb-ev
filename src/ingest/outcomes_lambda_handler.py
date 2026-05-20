"""AWS Lambda entry point for the nightly outcomes ingestion job.

Configure via environment variables on the Lambda function:
  OUTCOMES_S3_BUCKET   (required) Destination bucket for outcomes.
  OUTCOMES_S3_PREFIX   (optional) Defaults to ``raw/outcomes``.
  OUTCOMES_SPORT       (optional) Defaults to ``baseball_mlb``.

Trigger: EventBridge Scheduler nightly at ~01:00 America/New_York. The
function ingests outcomes for "yesterday in UTC" — i.e. whatever date is
fully complete at the time of the cron. The schedule's cron expression
handles the timezone; the code uses UTC for safety.

Optionally accepts an explicit ``event["date"]`` (YYYY-MM-DD) for backfill
or replay use cases triggered manually from the Lambda console.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.ingest.fetch_outcomes import (
    DEFAULT_S3_PREFIX,
    DEFAULT_SPORT,
    run_outcomes_ingest,
)

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def _resolve_target_date(event: dict[str, Any] | None) -> date:
    if event and event.get("date"):
        return datetime.strptime(event["date"], "%Y-%m-%d").date()
    return (datetime.now(timezone.utc) - timedelta(days=1)).date()


def handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    bucket = os.environ["OUTCOMES_S3_BUCKET"]
    target = _resolve_target_date(event)

    summary = run_outcomes_ingest(
        target,
        sport=os.getenv("OUTCOMES_SPORT", DEFAULT_SPORT),
        local_root=None,
        s3_bucket=bucket,
        s3_prefix=os.getenv("OUTCOMES_S3_PREFIX", DEFAULT_S3_PREFIX),
        skip_empty=True,
    )

    logger.info(
        "outcomes ingested | date=%s games=%s destinations=%s statuses=%s",
        summary["game_date"],
        summary["game_count"],
        summary["destinations"],
        summary["status_counts"],
    )
    return summary
