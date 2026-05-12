#!/usr/bin/env bash
# Build a deployment ZIP for the odds-ingestion Lambda.
#
# Output: build/lambda.zip
#
# Targets Linux ARM64 (Lambda Graviton) — pure-Python deps so it builds cleanly
# on macOS Apple Silicon without needing Docker.
#
# Usage:
#   bash infra/build_lambda_zip.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
STAGE_DIR="${BUILD_DIR}/lambda"
ZIP_PATH="${BUILD_DIR}/lambda.zip"

echo "==> Cleaning ${STAGE_DIR}"
rm -rf "${STAGE_DIR}" "${ZIP_PATH}"
mkdir -p "${STAGE_DIR}"

echo "==> Installing Lambda deps into ${STAGE_DIR}"
python3 -m pip install \
  --platform manylinux2014_aarch64 \
  --target "${STAGE_DIR}" \
  --implementation cp \
  --python-version 3.12 \
  --only-binary=:all: \
  --upgrade \
  -r "${REPO_ROOT}/requirements-lambda.txt"

echo "==> Copying source"
mkdir -p "${STAGE_DIR}/src/ingest"
touch "${STAGE_DIR}/src/__init__.py" "${STAGE_DIR}/src/ingest/__init__.py"
cp "${REPO_ROOT}/src/ingest/fetch_odds.py" "${STAGE_DIR}/src/ingest/"
cp "${REPO_ROOT}/src/ingest/lambda_handler.py" "${STAGE_DIR}/src/ingest/"

echo "==> Zipping into ${ZIP_PATH}"
( cd "${STAGE_DIR}" && zip -qr "${ZIP_PATH}" . )

SIZE_KB=$(du -k "${ZIP_PATH}" | awk '{print $1}')
echo "==> Done: ${ZIP_PATH} (${SIZE_KB} KB)"
