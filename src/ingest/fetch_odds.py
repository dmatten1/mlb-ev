"""Fetch a moneyline odds snapshot from The Odds API and persist it as raw JSON.

Destinations:
  - Local filesystem (default): writes to ``data/raw/odds/...``.
  - S3: when ``ODDS_S3_BUCKET`` is set, uploads to that bucket under the same
    relative key.

Run locally:
    python -m src.ingest.fetch_odds

After a successful snapshot, optionally run ``src.pipeline.live_refresh``
(predict + bet log + dashboard)::

    MLB_EV_LIVE_AFTER_ODDS=1 python -m src.ingest.fetch_odds
    python -m src.ingest.fetch_odds --live-after

On minimal images (scheduler container) live refresh is skipped if imports fail.

Run in Docker:
    docker compose run --rm ingest

Used as a library from Lambda via ``src.ingest.lambda_handler.handler``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # Lambda package omits python-dotenv.
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "baseball_mlb"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h"
DEFAULT_ODDS_FORMAT = "american"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_ROOT = REPO_ROOT / "data" / "raw" / "odds"
DEFAULT_S3_PREFIX = "raw/odds"

logger = logging.getLogger("ingest.fetch_odds")


def fetch_odds_snapshot(
    api_key: str,
    *,
    sport: str = DEFAULT_SPORT,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
    odds_format: str = DEFAULT_ODDS_FORMAT,
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Call The Odds API once and return (payload, response_headers)."""
    url = f"{ODDS_API_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": "iso",
    }
    resp = requests.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json(), dict(resp.headers)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def build_snapshot_record(
    payload: list[dict[str, Any]],
    headers: dict[str, str],
    *,
    sport: str,
    regions: str,
    markets: str,
    odds_format: str,
) -> dict[str, Any]:
    """Wrap the raw odds payload with capture metadata."""
    return {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sport": sport,
        "regions": regions,
        "markets": markets,
        "odds_format": odds_format,
        "requests_used": headers.get("x-requests-used"),
        "requests_remaining": headers.get("x-requests-remaining"),
        "last_request_cost": headers.get("x-requests-last"),
        "game_count": len(payload),
        "data": payload,
    }


def snapshot_relative_key(
    *, sport: str, markets: str, timestamp: str | None = None
) -> str:
    """Path of a snapshot relative to its destination root (local dir or S3 prefix)."""
    ts = timestamp or _utc_timestamp()
    date_partition = ts[:10]
    return f"{sport}/{markets}/{date_partition}/{ts}.json"


def write_local_snapshot(
    record: dict[str, Any],
    *,
    output_root: Path,
    sport: str,
    markets: str,
    timestamp: str | None = None,
) -> Path:
    rel = snapshot_relative_key(sport=sport, markets=markets, timestamp=timestamp)
    target = output_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return target


def upload_s3_snapshot(
    record: dict[str, Any],
    *,
    bucket: str,
    prefix: str,
    sport: str,
    markets: str,
    timestamp: str | None = None,
) -> str:
    """Upload a snapshot record to S3 and return the s3:// URI."""
    import boto3  # Imported lazily so local runs don't need boto3 installed.

    rel = snapshot_relative_key(sport=sport, markets=markets, timestamp=timestamp)
    key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
    body = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")

    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def run_snapshot(
    *,
    api_key: str,
    sport: str,
    regions: str,
    markets: str,
    odds_format: str,
    local_root: Path | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str = DEFAULT_S3_PREFIX,
) -> dict[str, Any]:
    """Fetch one snapshot and persist it. Returns a summary dict."""
    payload, headers = fetch_odds_snapshot(
        api_key,
        sport=sport,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
    )
    record = build_snapshot_record(
        payload,
        headers,
        sport=sport,
        regions=regions,
        markets=markets,
        odds_format=odds_format,
    )

    timestamp = _utc_timestamp()
    destinations: list[str] = []

    if s3_bucket:
        uri = upload_s3_snapshot(
            record,
            bucket=s3_bucket,
            prefix=s3_prefix,
            sport=sport,
            markets=markets,
            timestamp=timestamp,
        )
        destinations.append(uri)

    if local_root is not None:
        path = write_local_snapshot(
            record,
            output_root=local_root,
            sport=sport,
            markets=markets,
            timestamp=timestamp,
        )
        destinations.append(str(path))

    summary = {
        "game_count": len(payload),
        "requests_used": headers.get("x-requests-used"),
        "requests_remaining": headers.get("x-requests-remaining"),
        "last_request_cost": headers.get("x-requests-last"),
        "destinations": destinations,
        "timestamp": timestamp,
    }
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB moneyline odds snapshot.")
    parser.add_argument("--sport", default=os.getenv("ODDS_API_SPORT", DEFAULT_SPORT))
    parser.add_argument("--regions", default=os.getenv("ODDS_API_REGIONS", DEFAULT_REGIONS))
    parser.add_argument("--markets", default=os.getenv("ODDS_API_MARKETS", DEFAULT_MARKETS))
    parser.add_argument(
        "--odds-format",
        default=os.getenv("ODDS_API_ODDS_FORMAT", DEFAULT_ODDS_FORMAT),
        choices=["american", "decimal"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("ODDS_OUTPUT_DIR", str(DEFAULT_LOCAL_ROOT))),
        help="Local directory to write snapshots into. Set to '' to disable local writes.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.getenv("ODDS_S3_BUCKET"),
        help="If set, also upload the snapshot to this S3 bucket.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.getenv("ODDS_S3_PREFIX", DEFAULT_S3_PREFIX),
    )
    parser.add_argument(
        "--live-after",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="After success, run live_refresh (predict + tracker + dashboard). "
        "Default: only if MLB_EV_LIVE_AFTER_ODDS is 1/true/yes/on.",
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def _should_run_live_after(args: argparse.Namespace) -> bool:
    """``--live-after`` / ``--no-live-after`` override env ``MLB_EV_LIVE_AFTER_ODDS``."""
    flag = getattr(args, "live_after", None)
    if flag is True:
        return True
    if flag is False:
        return False
    v = os.getenv("MLB_EV_LIVE_AFTER_ODDS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _run_live_refresh_after_odds() -> int:
    """Return exit code from ``src.pipeline.live_refresh`` or ``0`` if skipped."""
    try:
        from src.pipeline import live_refresh
    except ImportError as e:
        logger.warning("Skipping live_refresh after odds (missing deps): %s", e)
        return 0
    logger.info("Starting live_refresh after odds snapshot")
    try:
        return live_refresh.main([])
    except Exception:  # noqa: BLE001 — log full traceback; caller sets exit code
        logger.exception("live_refresh failed after odds snapshot")
        return 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        logger.error("ODDS_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2

    local_root: Path | None = args.output_dir if str(args.output_dir) else None

    logger.info(
        "Requesting odds | sport=%s regions=%s markets=%s format=%s s3_bucket=%s",
        args.sport,
        args.regions,
        args.markets,
        args.odds_format,
        args.s3_bucket or "(none)",
    )

    try:
        summary = run_snapshot(
            api_key=api_key,
            sport=args.sport,
            regions=args.regions,
            markets=args.markets,
            odds_format=args.odds_format,
            local_root=local_root,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
        )
    except requests.HTTPError as e:
        body = e.response.text[:500] if e.response is not None else ""
        status = e.response.status_code if e.response is not None else "?"
        logger.error("Odds API HTTP %s: %s", status, body)
        return 1
    except requests.RequestException as e:
        logger.error("Odds API request failed: %s", e)
        return 1

    logger.info(
        "Snapshot saved | games=%d used=%s remaining=%s last_cost=%s destinations=%s",
        summary["game_count"],
        summary["requests_used"],
        summary["requests_remaining"],
        summary["last_request_cost"],
        summary["destinations"],
    )
    if _should_run_live_after(args):
        return _run_live_refresh_after_odds()
    return 0


if __name__ == "__main__":
    sys.exit(main())
