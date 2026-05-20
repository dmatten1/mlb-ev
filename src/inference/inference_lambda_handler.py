"""AWS Lambda (container image) entry: light refresh + S3 artifacts + dashboard.

Configure environment variables (see :mod:`src.cloud.artifacts`).

Typical flow: Odds zip Lambda saves snapshot → invokes this function →
downloads ``pipeline/data`` artifacts from S3 → runs outcomes/schedule/predict/track
→ uploads bet log + HTML → publishes ``index.html`` to the dashboard bucket.

Memory: 2048–3008 MB recommended. Timeout: 900 s (15 min) for cold starts + feature build.
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Any

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


def handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    from pathlib import Path

    from src.cloud.artifacts import (
        download_pipeline_artifacts,
        install_workspace,
        publish_dashboard,
        season_year,
        upload_pipeline_artifacts,
    )

    root = install_workspace(Path(os.environ.get("MLB_EV_REPO_ROOT", "/tmp/mlb-ev")))
    os.chdir(root)

    try:
        missing = download_pipeline_artifacts(root)
        if missing:
            logger.warning(
                "missing %d S3 artifacts — predict may fail. Run "
                "infra/sync_artifacts_to_s3.sh after a local `make refresh`. "
                "Missing: %s",
                len(missing),
                missing[:5],
            )

        from src.pipeline.live_refresh import main as live_main

        year = season_year()
        os.environ.setdefault("MLB_EV_YEAR", str(year))
        rc = live_main(["--year", str(year)])

        uploaded = upload_pipeline_artifacts(root)
        dash_uri = publish_dashboard(root)

        body = {
            "status": "ok" if rc == 0 else "partial_failure",
            "exit_code": rc,
            "year": year,
            "workspace": str(root),
            "missing_artifacts": missing,
            "uploaded_keys": list(uploaded.keys()),
            "dashboard_uri": dash_uri,
        }
        if rc != 0:
            logger.error("live_refresh exited %d", rc)
        else:
            logger.info("cloud live_refresh complete: %s", body)
        return body
    except Exception:
        logger.exception("inference Lambda failed")
        raise
