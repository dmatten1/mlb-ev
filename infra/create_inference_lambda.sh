#!/usr/bin/env bash
# Create IAM role + container Lambda for mlb-ev inference (cloud_deploy.md step 3).
#
# Usage:
#   export BUCKET=mlb-ev-dcm92
#   export AWS_REGION=us-east-1
#   export IMAGE_URI=966801367854.dkr.ecr.us-east-1.amazonaws.com/mlb-ev-inference:latest
#   bash infra/create_inference_lambda.sh

set -euo pipefail

BUCKET="${BUCKET:?export BUCKET=your-s3-bucket}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
FUNCTION_NAME="${FUNCTION_NAME:-mlb-ev-inference}"
ROLE_NAME="${ROLE_NAME:-mlb-ev-inference-lambda-role}"
IMAGE_URI="${IMAGE_URI:?export IMAGE_URI=ACCOUNT.dkr.ecr.REGION.amazonaws.com/mlb-ev-inference:latest}"
YEAR="${MLB_EV_YEAR:-2026}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_FILE="${REPO_ROOT}/build/inference_lambda_policy.json"
ENV_FILE="${REPO_ROOT}/build/inference_lambda_env.json"
TRUST_FILE="${REPO_ROOT}/build/inference_lambda_trust.json"

mkdir -p "${REPO_ROOT}/build"

cat > "$TRUST_FILE" <<'TRUST'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "lambda.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
TRUST

cat > "$POLICY_FILE" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PipelineS3",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ]
    },
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*"
    }
  ]
}
POLICY

cat > "$ENV_FILE" <<ENV
{
  "Variables": {
    "MLB_EV_S3_BUCKET": "${BUCKET}",
    "MLB_EV_PIPELINE_PREFIX": "pipeline/data",
    "DASHBOARD_S3_BUCKET": "${BUCKET}",
    "DASHBOARD_S3_KEY": "index.html",
    "MLB_EV_YEAR": "${YEAR}",
    "ODDS_S3_BUCKET": "${BUCKET}",
    "LOG_LEVEL": "INFO"
  }
}
ENV

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo "==> IAM role ${ROLE_NAME}"
if aws iam get-role --role-name "$ROLE_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "    role already exists"
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "file://${TRUST_FILE}"
  echo "    created role; waiting for IAM propagation..."
  sleep 10
fi

echo "==> IAM inline policy"
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name mlb-ev-inference-policy \
  --policy-document "file://${POLICY_FILE}"

echo "==> Lambda ${FUNCTION_NAME}"
if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" >/dev/null 2>&1; then
  echo "    function exists — updating image + config"
  aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$REGION"
  aws lambda wait function-updated-v2 \
    --function-name "$FUNCTION_NAME" \
    --region "$REGION"
  aws lambda update-function-configuration \
    --function-name "$FUNCTION_NAME" \
    --role "$ROLE_ARN" \
    --timeout 900 \
    --memory-size 3008 \
    --environment "file://${ENV_FILE}" \
    --region "$REGION"
else
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code "ImageUri=${IMAGE_URI}" \
    --role "$ROLE_ARN" \
    --timeout 900 \
    --memory-size 3008 \
    --architectures arm64 \
    --environment "file://${ENV_FILE}" \
    --region "$REGION"
fi

echo ""
echo "Done. Function: ${FUNCTION_NAME} (${REGION})"
echo "Role ARN: ${ROLE_ARN}"
echo "Next: set INFERENCE_LAMBDA_NAME=${FUNCTION_NAME} on the odds Lambda (cloud_deploy.md step 4)."
