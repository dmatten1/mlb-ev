#!/usr/bin/env bash
# EventBridge Scheduler: lightweight live-score dashboard refresh (10×/hour ET).
#
# Invokes mlb-ev-inference with mode=scores_refresh — downloads bet log from S3,
# hits MLB Stats API (free) for pending game scores, re-renders HTML, publishes
# index.html. Skips work when there are no pending bets.
#
# Cost: ~144 Lambda invocations/day × ~2–5 s each ≈ $0 on free tier.
#
# Usage:
#   export AWS_REGION=us-east-1
#   bash infra/setup_scores_refresh_schedule.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
GROUP="${SCHEDULER_GROUP:-mlb-ev}"
INFERENCE_FUNCTION="${INFERENCE_FUNCTION:-mlb-ev-inference}"
SCHEDULER_ROLE="${SCHEDULER_ROLE:-mlb-ev-scheduler-role}"
ODDS_FUNCTION="${ODDS_FUNCTION:-mlb-ev-ingest-odds}"
OUTCOMES_FUNCTION="${OUTCOMES_FUNCTION:-mlb-ev-ingest-outcomes}"
SCHEDULER_POLICY_NAME="${SCHEDULER_POLICY_NAME:-invoke-ingest-lambda}"

LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${INFERENCE_FUNCTION}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SCHEDULER_ROLE}"
SCHEDULE_NAME="${SCORES_SCHEDULE_NAME:-scores-10min-et}"
# Every 10 minutes, all day (America/New_York).
CRON="${SCORES_CRON:-cron(0/10 * * * ? *)}"

echo "==> Scheduler role may invoke odds, outcomes, and inference Lambdas"
POLICY_DOC=$(python3 - <<PY
import json
print(json.dumps({
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": "lambda:InvokeFunction",
        "Resource": [
            f"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${ODDS_FUNCTION}",
            f"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${OUTCOMES_FUNCTION}",
            f"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${INFERENCE_FUNCTION}",
        ],
    }],
}))
PY
)
aws iam put-role-policy \
  --role-name "$SCHEDULER_ROLE" \
  --policy-name "$SCHEDULER_POLICY_NAME" \
  --policy-document "$POLICY_DOC"

echo "==> Ensure scheduler group ${GROUP}"
aws scheduler get-schedule-group --name "$GROUP" --region "$REGION" >/dev/null 2>&1 || \
  aws scheduler create-schedule-group --name "$GROUP" --region "$REGION"

target=$(python3 - <<PY
import json
print(json.dumps({
    "Arn": "${LAMBDA_ARN}",
    "RoleArn": "${ROLE_ARN}",
    "Input": json.dumps({"mode": "scores_refresh", "source": "${SCHEDULE_NAME}"}),
}))
PY
)

if aws scheduler get-schedule --name "$SCHEDULE_NAME" --group-name "$GROUP" --region "$REGION" >/dev/null 2>&1; then
  echo "==> Update ${SCHEDULE_NAME}  ${CRON}"
  aws scheduler update-schedule \
    --name "$SCHEDULE_NAME" \
    --group-name "$GROUP" \
    --schedule-expression "$CRON" \
    --schedule-expression-timezone "America/New_York" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --state ENABLED \
    --target "$target" \
    --region "$REGION" \
    --output text >/dev/null
else
  echo "==> Create ${SCHEDULE_NAME}  ${CRON}"
  aws scheduler create-schedule \
    --name "$SCHEDULE_NAME" \
    --group-name "$GROUP" \
    --schedule-expression "$CRON" \
    --schedule-expression-timezone "America/New_York" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --state ENABLED \
    --target "$target" \
    --region "$REGION" \
    --output text >/dev/null
fi

echo ""
aws scheduler get-schedule \
  --name "$SCHEDULE_NAME" \
  --group-name "$GROUP" \
  --region "$REGION" \
  --query '{Name:Name,Schedule:ScheduleExpression,Timezone:ScheduleExpressionTimezone,Target:Target.Arn}' \
  --output table

echo ""
echo "Deploy inference Lambda code first if you changed dashboard/scores_refresh:"
echo "  bash infra/build_inference_lambda_image.sh && bash infra/create_inference_lambda.sh"
