#!/usr/bin/env bash
# Push ODDS_API_KEY from repo .env to the odds Lambda (merges other env vars).
#
# Usage:
#   export ODDS_FUNCTION=mlb-ev-ingest-odds
#   export AWS_REGION=us-east-1
#   bash infra/update_odds_api_key.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
ODDS_FUNCTION="${ODDS_FUNCTION:-mlb-ev-ingest-odds}"
REGION="${AWS_REGION:-us-east-1}"
BUILD_DIR="${REPO_ROOT}/build"
OUT="${BUILD_DIR}/odds_lambda_env.json"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: ${ENV_FILE} not found (need ODDS_API_KEY=...)" >&2
  exit 1
fi

ODDS_API_KEY="$(grep -E '^ODDS_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
if [[ -z "$ODDS_API_KEY" || ${#ODDS_API_KEY} -lt 10 ]]; then
  echo "ERROR: ODDS_API_KEY in .env looks invalid (too short)" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: install jq (brew install jq)" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
CURRENT_VARS=$(aws lambda get-function-configuration \
  --function-name "$ODDS_FUNCTION" \
  --region "$REGION" \
  --query 'Environment.Variables' \
  --output json)
if [[ "$CURRENT_VARS" == "null" || -z "$CURRENT_VARS" ]]; then
  CURRENT_VARS='{}'
fi

jq -n \
  --argjson vars "$CURRENT_VARS" \
  --arg key "$ODDS_API_KEY" \
  '{Variables: ($vars + {ODDS_API_KEY: $key})}' > "$OUT"

aws lambda update-function-configuration \
  --function-name "$ODDS_FUNCTION" \
  --environment "file://${OUT}" \
  --region "$REGION"
aws lambda wait function-updated-v2 \
  --function-name "$ODDS_FUNCTION" \
  --region "$REGION"

echo "Done. Updated ODDS_API_KEY on ${ODDS_FUNCTION} (len=${#ODDS_API_KEY})."
