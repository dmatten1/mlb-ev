#!/usr/bin/env bash
# cloud_deploy.md step 4: redeploy odds zip, set INFERENCE_LAMBDA_NAME, allow odds role
# to invoke the inference Lambda.
#
# Usage:
#   export BUCKET=mlb-ev-dcm92
#   export AWS_REGION=us-east-1
#   bash infra/chain_odds_lambda.sh

set -euo pipefail

BUCKET="${BUCKET:?export BUCKET=your-s3-bucket}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ODDS_FUNCTION="${ODDS_FUNCTION:-mlb-ev-ingest-odds}"
INFERENCE_FUNCTION="${INFERENCE_FUNCTION:-mlb-ev-inference}"
ODDS_ROLE="${ODDS_ROLE:-mlb-ev-ingest-lambda-role}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
INVOKE_POLICY="${BUILD_DIR}/odds_invoke_inference_policy.json"
ENV_FILE="${BUILD_DIR}/odds_lambda_env.json"

mkdir -p "$BUILD_DIR"

INFERENCE_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${INFERENCE_FUNCTION}"

echo "==> Build odds Lambda zip (includes src/cloud for chaining)"
bash "${REPO_ROOT}/infra/build_lambda_zip.sh"

echo "==> Deploy zip to ${ODDS_FUNCTION}"
aws lambda update-function-code \
  --function-name "$ODDS_FUNCTION" \
  --zip-file "fileb://${REPO_ROOT}/build/lambda.zip" \
  --region "$REGION"
aws lambda wait function-updated-v2 \
  --function-name "$ODDS_FUNCTION" \
  --region "$REGION"

echo "==> Merge INFERENCE_LAMBDA_NAME into odds Lambda environment"
CURRENT_VARS=$(aws lambda get-function-configuration \
  --function-name "$ODDS_FUNCTION" \
  --region "$REGION" \
  --query 'Environment.Variables' \
  --output json)
if [[ "$CURRENT_VARS" == "null" || -z "$CURRENT_VARS" ]]; then
  CURRENT_VARS='{}'
fi
# Requires jq
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: install jq (brew install jq) to merge Lambda env vars safely." >&2
  exit 1
fi
jq -n \
  --argjson vars "$CURRENT_VARS" \
  --arg bucket "$BUCKET" \
  --arg infer "$INFERENCE_FUNCTION" \
  '{Variables: ($vars + {
    ODDS_S3_BUCKET: $bucket,
    INFERENCE_LAMBDA_NAME: $infer,
    LOG_LEVEL: ($vars.LOG_LEVEL // "INFO")
  })}' > "$ENV_FILE"
aws lambda update-function-configuration \
  --function-name "$ODDS_FUNCTION" \
  --environment "file://${ENV_FILE}" \
  --region "$REGION"
aws lambda wait function-updated-v2 \
  --function-name "$ODDS_FUNCTION" \
  --region "$REGION"

echo "==> IAM: allow ${ODDS_ROLE} to invoke ${INFERENCE_FUNCTION}"
cat > "$INVOKE_POLICY" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeInference",
      "Effect": "Allow",
      "Action": "lambda:InvokeFunction",
      "Resource": "${INFERENCE_ARN}"
    }
  ]
}
POLICY
aws iam put-role-policy \
  --role-name "$ODDS_ROLE" \
  --policy-name mlb-ev-invoke-inference \
  --policy-document "file://${INVOKE_POLICY}"

echo ""
echo "Done."
echo "  Odds Lambda: ${ODDS_FUNCTION}"
echo "  Will async-invoke: ${INFERENCE_ARN}"
echo "  Test: aws lambda invoke --function-name ${ODDS_FUNCTION} --region ${REGION} /tmp/odds-out.json"
