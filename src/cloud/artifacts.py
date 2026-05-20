"""Sync pipeline ``data/`` artifacts with S3 and publish the static dashboard.

Environment (inference Lambda)::

    MLB_EV_S3_BUCKET          — same bucket as odds (e.g. mlb-ev-dcm92)
    MLB_EV_PIPELINE_PREFIX    — prefix for parquets/model (default pipeline/data)
    DASHBOARD_S3_BUCKET       — bucket for public website (can equal MLB_EV_S3_BUCKET)
    DASHBOARD_S3_KEY          — object key (default index.html)
    MLB_EV_YEAR               — season year for lineup/statcast paths (default: today)
    MLB_EV_REPO_ROOT          — writable workspace (default /tmp/mlb-ev)

Odds Lambda (chain)::

    INFERENCE_LAMBDA_NAME     — if set, async-invoke after each snapshot
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import boto3

logger = logging.getLogger("cloud.artifacts")

DEFAULT_REPO_ROOT = Path("/tmp/mlb-ev")
DEFAULT_PIPELINE_PREFIX = "pipeline/data"


def repo_root() -> Path:
    return Path(os.environ.get("MLB_EV_REPO_ROOT", str(DEFAULT_REPO_ROOT)))


def pipeline_prefix() -> str:
    p = os.environ.get("MLB_EV_PIPELINE_PREFIX", DEFAULT_PIPELINE_PREFIX).strip("/")
    return p


def s3_bucket() -> str:
    b = os.environ.get("MLB_EV_S3_BUCKET") or os.environ.get("ODDS_S3_BUCKET")
    if not b:
        raise ValueError("MLB_EV_S3_BUCKET or ODDS_S3_BUCKET must be set")
    return b


def season_year() -> int:
    raw = os.environ.get("MLB_EV_YEAR")
    if raw:
        return int(raw)
    return date.today().year


def _s3():
    return boto3.client("s3")


def _download_key(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("s3 download s3://%s/%s -> %s", bucket, key, dest)
    _s3().download_file(bucket, key, str(dest))


def _upload_file(local: Path, bucket: str, key: str, *, content_type: str | None = None) -> str:
    logger.info("s3 upload %s -> s3://%s/%s", local, bucket, key)
    if content_type:
        _s3().upload_file(
            str(local), bucket, key,
            ExtraArgs={"ContentType": content_type},
        )
    else:
        _s3().upload_file(str(local), bucket, key)
    return f"s3://{bucket}/{key}"


def required_artifact_keys(year: int) -> list[str]:
    """S3 keys (under ``pipeline_prefix``) needed for ``live_refresh`` predict."""
    y = year
    return [
        f"features/training_2023.parquet",
        f"features/training_2024.parquet",
        f"features/training_{y}.parquet",
        "models/runs_model_bullpen_cached.pkl",
        f"lineups/lineups_long_{y}.parquet",
        f"lineups/lineups_{y}.parquet",
        f"oaa/oaa_{y}.parquet",
        "park_factors/park_factors_2024_rolling3.parquet",
        f"raw/statcast/statcast_{y}.parquet",
        "tracking/bet_log.parquet",
    ]


def optional_recent_prefixes() -> list[str]:
    """Prefixes to sync for schedule/outcomes continuity across cold starts."""
    return [
        "raw/schedule/baseball_mlb",
        "raw/outcomes/baseball_mlb",
        "predictions",
    ]


def download_pipeline_artifacts(root: Path | None = None) -> list[str]:
    """Pull required files from S3 into ``root/data``. Returns list of missing keys."""
    root = root or repo_root()
    bucket = s3_bucket()
    prefix = pipeline_prefix()
    missing: list[str] = []
    year = season_year()

    for rel in required_artifact_keys(year):
        key = f"{prefix}/{rel}"
        dest = root / "data" / rel
        try:
            _download_key(bucket, key, dest)
        except Exception as e:  # noqa: BLE001
            if rel.endswith("bet_log.parquet"):
                logger.info("no existing bet_log in S3 (starting fresh): %s", e)
                continue
            logger.warning("missing artifact %s: %s", key, e)
            missing.append(key)

    # Recent schedule/outcomes JSON (last 14 partition days best-effort)
    _sync_prefix_down(bucket, f"{prefix}/raw/schedule/baseball_mlb", root / "data/raw/schedule/baseball_mlb")
    _sync_prefix_down(bucket, f"{prefix}/raw/outcomes/baseball_mlb", root / "data/raw/outcomes/baseball_mlb")

    return missing


def _sync_prefix_down(bucket: str, prefix: str, local_dir: Path, *, max_keys: int = 500) -> None:
    """Download objects under ``prefix`` (paginated, capped)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = _s3().get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents") or []:
            if n >= max_keys:
                return
            key = obj["Key"]
            rel = key[len(prefix.rstrip("/") + "/") :]
            if not rel:
                continue
            dest = local_dir / rel
            try:
                _download_key(bucket, key, dest)
                n += 1
            except Exception as e:  # noqa: BLE001
                logger.debug("skip %s: %s", key, e)


def upload_pipeline_artifacts(root: Path | None = None) -> dict[str, str]:
    """Push mutable artifacts back to S3 after a run."""
    root = root or repo_root()
    bucket = s3_bucket()
    prefix = pipeline_prefix()
    uploaded: dict[str, str] = {}

    uploads: list[tuple[Path, str]] = [
        (root / "data/tracking/bet_log.parquet", f"{prefix}/tracking/bet_log.parquet"),
        (root / "data/tracking/bet_dashboard.html", f"{prefix}/tracking/bet_dashboard.html"),
    ]
    pred_dir = root / "data/predictions"
    if pred_dir.is_dir():
        for p in sorted(pred_dir.glob("*.parquet"))[-3:]:
            uploads.append((p, f"{prefix}/predictions/{p.name}"))

    for local, key in uploads:
        if not local.exists():
            continue
        ct = "text/html" if local.suffix == ".html" else None
        uploaded[key] = _upload_file(local, bucket, key, content_type=ct)

    _upload_tree(root / "data/raw/schedule/baseball_mlb", bucket, f"{prefix}/raw/schedule/baseball_mlb", days=7)
    _upload_tree(root / "data/raw/outcomes/baseball_mlb", bucket, f"{prefix}/raw/outcomes/baseball_mlb", days=7)

    return uploaded


def _upload_tree(local_root: Path, bucket: str, s3_prefix: str, *, days: int = 7) -> None:
    if not local_root.is_dir():
        return
    cutoff = date.today() - timedelta(days=days)
    for path in local_root.rglob("*.json"):
        try:
            part = path.parent.name
            if len(part) == 10 and part[4] == "-":
                if date.fromisoformat(part) < cutoff:
                    continue
        except ValueError:
            pass
        rel = path.relative_to(local_root)
        key = f"{s3_prefix.rstrip('/')}/{rel.as_posix()}"
        _upload_file(path, bucket, key)


def publish_dashboard(root: Path | None = None) -> str | None:
    """Copy dashboard HTML to the static-website bucket as ``index.html``."""
    root = root or repo_root()
    html_path = root / "data/tracking/bet_dashboard.html"
    if not html_path.exists():
        logger.warning("dashboard HTML missing at %s", html_path)
        return None

    bucket = os.environ.get("DASHBOARD_S3_BUCKET") or s3_bucket()
    key = os.environ.get("DASHBOARD_S3_KEY", "index.html")
    uri = _upload_file(html_path, bucket, key, content_type="text/html; charset=utf-8")
    logger.info("published dashboard to %s", uri)
    return uri


def install_workspace(root: Path | None = None) -> Path:
    """Point all pipeline modules at a writable repo root (``/tmp/mlb-ev`` in Lambda)."""
    root = (root or repo_root()).resolve()
    os.environ["MLB_EV_REPO_ROOT"] = str(root)
    (root / "data").mkdir(parents=True, exist_ok=True)

    import src.pipeline.daily_refresh as dr
    import src.ingest.fetch_outcomes as fo
    import src.ingest.fetch_schedule as fs
    import src.tracking.bet_log as bl
    import src.tracking.dashboard as dash

    dr.REPO_ROOT = root
    dr.DEFAULT_PREDICTIONS_ROOT = root / "data/predictions"
    dr.DEFAULT_RUNS_MODEL_CACHE = root / "data/models/runs_model_bullpen_cached.pkl"

    fo.REPO_ROOT = root
    fo.DEFAULT_LOCAL_ROOT = root / "data/raw/outcomes"

    fs.REPO_ROOT = root
    fs.DEFAULT_LOCAL_ROOT = root / "data/raw/schedule"
    fs.DEFAULT_STATCAST_ROOT = root / "data/raw/statcast"

    bl.DEFAULT_LOG_PATH = root / "data/tracking/bet_log.parquet"
    dash.DEFAULT_OUT = root / "data/tracking/bet_dashboard.html"

    logger.info("workspace installed at %s", root)
    return root


def invoke_inference_lambda(*, payload: dict | None = None) -> None:
    """Async-invoke the inference Lambda (used by odds handler)."""
    name = os.environ.get("INFERENCE_LAMBDA_NAME", "").strip()
    if not name:
        return
    import json

    client = boto3.client("lambda")
    body = payload or {"source": "odds"}
    logger.info("invoking inference Lambda %s (Event)", name)
    client.invoke(
        FunctionName=name,
        InvocationType="Event",
        Payload=json.dumps(body).encode("utf-8"),
    )
