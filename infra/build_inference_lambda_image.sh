#!/usr/bin/env bash
# Build and push the inference Lambda container image to ECR.
#
# Prerequisites: AWS CLI, Docker, logged in to ECR.
#
# Usage:
#   export AWS_REGION=us-east-1
#   export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
#   export ECR_REPO=mlb-ev-inference
#   bash infra/build_inference_lambda_image.sh
#
# Then create/update the Lambda function from the image URI printed at the end.

set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not running." >&2
  echo "  Open Docker Desktop on macOS and wait until 'Docker is running', then retry." >&2
  echo "  Verify with: docker info" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REPO="${ECR_REPO:-mlb-ev-inference}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo "==> ECR login"
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Ensure repository ${ECR_REPO}"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" 2>/dev/null || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION"

# Lambda requires a single Docker V2 manifest for linux/arm64. Docker Desktop's
# default BuildKit attestations (provenance/SBOM) produce an OCI index Lambda rejects:
#   "image manifest, config or layer media type ... is not supported"
echo "==> docker build (linux/arm64, no attestations)"
export DOCKER_BUILDKIT=1
docker build \
  --platform linux/arm64 \
  --provenance=false \
  --sbom=false \
  -f "${REPO_ROOT}/infra/lambda_inference.Dockerfile" \
  -t "$URI" \
  "${REPO_ROOT}"

echo "==> docker push ${URI}"
docker push "$URI"

echo ""
echo "Image URI for Lambda:"
echo "  ${URI}"
