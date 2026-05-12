"""Fetch a moneyline odds snapshot from The Odds API and persist it as raw JSON.

The snapshot is stored exactly as returned by the provider, wrapped with capture
metadata (timestamp, rate-limit headers, request parameters) so downstream
feature engineering can reconstruct point-in-time market state without
guessing.

Run locally:
    python -m src.ingest.fetch_odds

Run in Docker:
    docker compose run --rm ingest
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
from dotenv import load_dotenv

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "baseball_mlb"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h"
DEFAULT_ODDS_FORMAT = "american"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "raw" / "odds"

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


def write_snapshot(
    payload: list[dict[str, Any]],
    headers: dict[str, str],
    *,
    output_root: Path,
    sport: str,
    regions: str,
    markets: str,
    odds_format: str,
) -> Path:
    """Persist a wrapped snapshot to disk and return the path."""
    timestamp = _utc_timestamp()
    date_partition = timestamp[:10]
    target_dir = output_root / sport / markets / date_partition
    target_dir.mkdir(parents=True, exist_ok=True)

    record = {
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

    target = target_dir / f"{timestamp}.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return target


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
        default=Path(os.getenv("ODDS_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT))),
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


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

    logger.info(
        "Requesting odds | sport=%s regions=%s markets=%s format=%s",
        args.sport,
        args.regions,
        args.markets,
        args.odds_format,
    )

    try:
        payload, headers = fetch_odds_snapshot(
            api_key,
            sport=args.sport,
            regions=args.regions,
            markets=args.markets,
            odds_format=args.odds_format,
        )
    except requests.HTTPError as e:
        body = e.response.text[:500] if e.response is not None else ""
        status = e.response.status_code if e.response is not None else "?"
        logger.error("Odds API HTTP %s: %s", status, body)
        return 1
    except requests.RequestException as e:
        logger.error("Odds API request failed: %s", e)
        return 1

    path = write_snapshot(
        payload,
        headers,
        output_root=args.output_dir,
        sport=args.sport,
        regions=args.regions,
        markets=args.markets,
        odds_format=args.odds_format,
    )

    logger.info(
        "Snapshot saved | games=%d used=%s remaining=%s last_cost=%s path=%s",
        len(payload),
        headers.get("x-requests-used"),
        headers.get("x-requests-remaining"),
        headers.get("x-requests-last"),
        path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
