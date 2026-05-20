#!/usr/bin/env bash
# One-time / periodic upload of local pipeline data artifacts to S3 for the
# inference Lambda. Run after `make refresh` when parquets and model cache are fresh.
#
# Usage:
#   export BUCKET=mlb-ev-dcm92
#   export YEAR=2026
#   bash infra/sync_artifacts_to_s3.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUCKET="${BUCKET:?set BUCKET}"
PREFIX="${MLB_EV_PIPELINE_PREFIX:-pipeline/data}"
YEAR="${YEAR:-$(date +%Y)}"
DEST="s3://${BUCKET}/${PREFIX}"

cd "$REPO_ROOT"

echo "==> Syncing required artifacts to ${DEST}/"
_upload() {
  local src="$1" dest="$2"
  if [[ ! -f "$src" ]]; then
    echo "SKIP (missing): $src"
    return 0
  fi
  aws s3 cp "$src" "$dest"
}
_upload "data/features/training_2023.parquet" "${DEST}/features/training_2023.parquet"
_upload "data/features/training_2024.parquet" "${DEST}/features/training_2024.parquet"
_upload "data/features/training_${YEAR}.parquet" "${DEST}/features/training_${YEAR}.parquet"
_upload "data/models/runs_model_bullpen_cached.pkl" "${DEST}/models/runs_model_bullpen_cached.pkl"
_upload "data/lineups/lineups_long_${YEAR}.parquet" "${DEST}/lineups/lineups_long_${YEAR}.parquet"
_upload "data/lineups/lineups_${YEAR}.parquet" "${DEST}/lineups/lineups_${YEAR}.parquet"
_upload "data/oaa/oaa_${YEAR}.parquet" "${DEST}/oaa/oaa_${YEAR}.parquet"
_upload "data/park_factors/park_factors_2024_rolling3.parquet" \
  "${DEST}/park_factors/park_factors_2024_rolling3.parquet"
_upload "data/raw/statcast/statcast_${YEAR}.parquet" "${DEST}/raw/statcast/statcast_${YEAR}.parquet"

if [[ -f data/tracking/bet_log.parquet ]]; then
  aws s3 cp data/tracking/bet_log.parquet "${DEST}/tracking/bet_log.parquet"
fi

echo "==> Optional: recent schedule/outcomes JSON"
aws s3 sync data/raw/schedule/baseball_mlb "${DEST}/raw/schedule/baseball_mlb" --exclude "*" --include "*.json"
aws s3 sync data/raw/outcomes/baseball_mlb "${DEST}/raw/outcomes/baseball_mlb" --exclude "*" --include "*.json"

echo "==> Done. Inference Lambda reads from ${DEST}/"
