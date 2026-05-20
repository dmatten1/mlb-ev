#!/usr/bin/env bash
# Enable S3 static website hosting for the bet dashboard (cloud_deploy.md step 6).
#
# Usage:
#   export BUCKET=mlb-ev-dcm92
#   export AWS_REGION=us-east-1
#   bash infra/setup_dashboard_website.sh
#
# Optional: upload local dashboard first
#   aws s3 cp data/tracking/bet_dashboard.html s3://$BUCKET/index.html

set -euo pipefail

BUCKET="${BUCKET:?export BUCKET=your-s3-bucket}"
REGION="${AWS_REGION:-us-east-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_FILE="${REPO_ROOT}/build/dashboard_website_policy.json"
WEBSITE_FILE="${REPO_ROOT}/build/s3_website.json"

mkdir -p "${REPO_ROOT}/build"

echo "==> Bucket ${BUCKET} (${REGION})"
if ! aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" 2>/dev/null; then
  echo "ERROR: bucket ${BUCKET} not found or no access" >&2
  exit 1
fi

echo "==> Allow public bucket policy (required for website endpoint)"
aws s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false" \
  --region "$REGION"

echo "==> Static website hosting (index.html)"
cat > "$WEBSITE_FILE" <<'WEB'
{
  "IndexDocument": { "Suffix": "index.html" },
  "ErrorDocument": { "Key": "index.html" }
}
WEB
aws s3api put-bucket-website \
  --bucket "$BUCKET" \
  --website-configuration "file://${WEBSITE_FILE}" \
  --region "$REGION"

echo "==> Bucket policy: public read index.html only"
cat > "$POLICY_FILE" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadDashboard",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::${BUCKET}/index.html"
    }
  ]
}
POLICY
aws s3api put-bucket-policy \
  --bucket "$BUCKET" \
  --policy "file://${POLICY_FILE}" \
  --region "$REGION"

LOCAL_HTML="${REPO_ROOT}/data/tracking/bet_dashboard.html"
if [[ -f "$LOCAL_HTML" ]]; then
  echo "==> Upload local dashboard -> s3://${BUCKET}/index.html"
  aws s3 cp "$LOCAL_HTML" "s3://${BUCKET}/index.html" \
    --content-type "text/html; charset=utf-8" \
    --region "$REGION"
elif aws s3api head-object --bucket "$BUCKET" --key index.html --region "$REGION" 2>/dev/null; then
  echo "==> index.html already in bucket"
else
  echo "==> Placeholder index.html (inference Lambda will overwrite on next run)"
  cat > "${REPO_ROOT}/build/placeholder_index.html" <<'HTML'
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>MLB EV Tracker</title></head>
<body><p>Dashboard not generated yet. Wait for the next odds + inference run, or upload
<code>data/tracking/bet_dashboard.html</code> to <code>index.html</code>.</p></body></html>
HTML
  aws s3 cp "${REPO_ROOT}/build/placeholder_index.html" "s3://${BUCKET}/index.html" \
    --content-type "text/html; charset=utf-8" \
    --region "$REGION"
fi

WEBSITE_URL="http://${BUCKET}.s3-website-${REGION}.amazonaws.com/"
echo ""
echo "Website URL:"
echo "  ${WEBSITE_URL}"
echo ""
echo "Inference Lambda should have:"
echo "  DASHBOARD_S3_BUCKET=${BUCKET}"
echo "  DASHBOARD_S3_KEY=index.html"
