#!/usr/bin/env bash
# Email alerts when the cloud pipeline fails (CloudWatch → SNS).
#
# Creates:
#   * SNS topic + email subscription (you must click Confirm in email)
#   * Lambda error alarms (odds + inference)
#   * Log-based alarms (odds API failure, inference partial_failure)
#
# Usage:
#   export ALERT_EMAIL=you@example.com
#   export AWS_REGION=us-east-1
#   bash infra/setup_email_alerts.sh
#
# After setup, AWS sends a subscription confirmation — click the link or
# alerts will not arrive.

set -euo pipefail

ALERT_EMAIL="${ALERT_EMAIL:?export ALERT_EMAIL=you@example.com}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
TOPIC_NAME="${SNS_TOPIC_NAME:-mlb-ev-pipeline-alerts}"
ODDS_FUNCTION="${ODDS_FUNCTION:-mlb-ev-ingest-odds}"
INFERENCE_FUNCTION="${INFERENCE_FUNCTION:-mlb-ev-inference}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
SNS_POLICY="${BUILD_DIR}/sns_cloudwatch_policy.json"

mkdir -p "$BUILD_DIR"

echo "==> SNS topic ${TOPIC_NAME}"
TOPIC_ARN=$(aws sns create-topic \
  --name "$TOPIC_NAME" \
  --region "$REGION" \
  --query 'TopicArn' \
  --output text 2>/dev/null || \
  aws sns list-topics --region "$REGION" \
    --query "Topics[?contains(TopicArn, ':${TOPIC_NAME}')].TopicArn | [0]" \
    --output text)
echo "    ${TOPIC_ARN}"

echo "==> Email subscription ${ALERT_EMAIL}"
SUB_ARN=$(aws sns subscribe \
  --topic-arn "$TOPIC_ARN" \
  --protocol email \
  --notification-endpoint "$ALERT_EMAIL" \
  --region "$REGION" \
  --query 'SubscriptionArn' \
  --output text)
if [[ "$SUB_ARN" == "pending confirmation" ]]; then
  echo "    Pending — check ${ALERT_EMAIL} and confirm the subscription."
else
  echo "    ${SUB_ARN}"
fi

echo "==> Allow CloudWatch to publish to SNS"
cat > "$SNS_POLICY" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCloudWatchPublish",
      "Effect": "Allow",
      "Principal": { "Service": "cloudwatch.amazonaws.com" },
      "Action": "SNS:Publish",
      "Resource": "${TOPIC_ARN}",
      "Condition": {
        "StringEquals": { "AWS:SourceAccount": "${ACCOUNT_ID}" }
      }
    }
  ]
}
POLICY
aws sns set-topic-attributes \
  --topic-arn "$TOPIC_ARN" \
  --attribute-name Policy \
  --attribute-value "file://${SNS_POLICY}" \
  --region "$REGION"

_put_alarm() {
  local name="$1"
  shift
  aws cloudwatch put-metric-alarm \
    --alarm-name "$name" \
    --region "$REGION" \
    --alarm-actions "$TOPIC_ARN" \
    --ok-actions "$TOPIC_ARN" \
    --treat-missing-data notBreaching \
    "$@"
  echo "    alarm: ${name}"
}

echo "==> Lambda error alarms (any uncaught exception)"
_put_alarm "mlb-ev-odds-lambda-errors" \
  --alarm-description "mlb-ev-ingest-odds Lambda reported Errors (check CloudWatch logs for 401, etc.)" \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions "Name=FunctionName,Value=${ODDS_FUNCTION}" \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold

_put_alarm "mlb-ev-inference-lambda-errors" \
  --alarm-description "mlb-ev-inference Lambda reported Errors (predict/track may have failed)" \
  --namespace AWS/Lambda \
  --metric-name Errors \
  --dimensions "Name=FunctionName,Value=${INFERENCE_FUNCTION}" \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold

echo "==> Log metric filters (pipeline-specific failures)"
aws logs put-metric-filter \
  --log-group-name "/aws/lambda/${ODDS_FUNCTION}" \
  --filter-name mlb-ev-odds-snapshot-failed \
  --filter-pattern '?ERROR ?"odds snapshot failed" ?"401 Client Error" ?"status code 401"' \
  --metric-transformations \
    "metricName=OddsSnapshotFailed,metricNamespace=MLBEv/Pipeline,metricValue=1,defaultValue=0" \
  --region "$REGION"
_put_alarm "mlb-ev-odds-snapshot-failed" \
  --alarm-description "Odds pull failed (often invalid ODDS_API_KEY — run infra/update_odds_api_key.sh)" \
  --namespace MLBEv/Pipeline \
  --metric-name OddsSnapshotFailed \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold

aws logs put-metric-filter \
  --log-group-name "/aws/lambda/${INFERENCE_FUNCTION}" \
  --filter-name mlb-ev-inference-partial-failure \
  --filter-pattern '?ERROR ?"[predict] FAILED" ?"live_refresh exited" ?"[FAIL] track"' \
  --metric-transformations \
    "metricName=InferencePartialFailure,metricNamespace=MLBEv/Pipeline,metricValue=1,defaultValue=0" \
  --region "$REGION"
_put_alarm "mlb-ev-inference-partial-failure" \
  --alarm-description "Inference live_refresh failed or partial (dashboard may be stale)" \
  --namespace MLBEv/Pipeline \
  --metric-name InferencePartialFailure \
  --statistic Sum \
  --period 300 \
  --evaluation-periods 1 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold

echo ""
echo "Done."
echo "  Topic: ${TOPIC_ARN}"
echo "  Confirm the SNS email to ${ALERT_EMAIL} if you have not already."
echo ""
echo "View alarms:"
echo "  https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#alarmsV2:"
echo ""
echo "Test (optional — fires an alarm email after the next failed odds key):"
echo "  aws lambda invoke --function-name ${ODDS_FUNCTION} --region ${REGION} /tmp/test.json"
