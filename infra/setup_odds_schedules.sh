#!/usr/bin/env bash
# EventBridge Scheduler: odds ingest Lambda (9×/day, America/New_York).
#
# Fires at 9am, noon, then every 90 minutes from 1pm through 10pm ET
# (9 API calls/day). Each invoke chains async to mlb-ev-inference.
#
# Usage:
#   export AWS_REGION=us-east-1
#   bash infra/setup_odds_schedules.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
GROUP="${SCHEDULER_GROUP:-mlb-ev}"
ODDS_FUNCTION="${ODDS_FUNCTION:-mlb-ev-ingest-odds}"
SCHEDULER_ROLE="${SCHEDULER_ROLE:-mlb-ev-scheduler-role}"

LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${ODDS_FUNCTION}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${SCHEDULER_ROLE}"

# name_suffix|cron expression (minute hour dom month dow year)
SCHEDULES=(
  "0900-et|cron(0 9 * * ? *)"
  "1200-et|cron(0 12 * * ? *)"
  "1300-et|cron(0 13 * * ? *)"
  "1430-et|cron(30 14 * * ? *)"
  "1600-et|cron(0 16 * * ? *)"
  "1730-et|cron(30 17 * * ? *)"
  "1900-et|cron(0 19 * * ? *)"
  "2030-et|cron(30 20 * * ? *)"
  "2200-et|cron(0 22 * * ? *)"
)

# Retired 4×/day schedules (superseded by the 9× cadence above).
RETIRED=(
  odds-0000-et
  odds-1230-et
  odds-1800-et
  odds-2100-et
)

echo "==> Ensure scheduler group ${GROUP}"
aws scheduler get-schedule-group --name "$GROUP" --region "$REGION" >/dev/null 2>&1 || \
  aws scheduler create-schedule-group --name "$GROUP" --region "$REGION"

_upsert_schedule() {
  local name="$1"
  local cron="$2"
  local target
  target=$(python3 - <<PY
import json
print(json.dumps({
    "Arn": "${LAMBDA_ARN}",
    "RoleArn": "${ROLE_ARN}",
    "Input": json.dumps({"source": "${name}"}),
}))
PY
)
  if aws scheduler get-schedule --name "$name" --group-name "$GROUP" --region "$REGION" >/dev/null 2>&1; then
    echo "    update ${name}  ${cron}"
    aws scheduler update-schedule \
      --name "$name" \
      --group-name "$GROUP" \
      --schedule-expression "$cron" \
      --schedule-expression-timezone "America/New_York" \
      --flexible-time-window '{"Mode":"OFF"}' \
      --state ENABLED \
      --target "$target" \
      --region "$REGION" \
      --output text >/dev/null
  else
    echo "    create ${name}  ${cron}"
    aws scheduler create-schedule \
      --name "$name" \
      --group-name "$GROUP" \
      --schedule-expression "$cron" \
      --schedule-expression-timezone "America/New_York" \
      --flexible-time-window '{"Mode":"OFF"}' \
      --state ENABLED \
      --target "$target" \
      --region "$REGION" \
      --output text >/dev/null
  fi
}

echo "==> Upsert odds schedules (9×/day ET)"
for entry in "${SCHEDULES[@]}"; do
  suffix="${entry%%|*}"
  cron="${entry#*|}"
  _upsert_schedule "odds-${suffix}" "$cron"
done

echo "==> Remove retired schedules"
for name in "${RETIRED[@]}"; do
  if aws scheduler get-schedule --name "$name" --group-name "$GROUP" --region "$REGION" >/dev/null 2>&1; then
    echo "    delete ${name}"
    aws scheduler delete-schedule --name "$name" --group-name "$GROUP" --region "$REGION"
  fi
done

echo ""
echo "Active odds schedules:"
aws scheduler list-schedules \
  --group-name "$GROUP" \
  --region "$REGION" \
  --query 'Schedules[?starts_with(Name, `odds-`)].{Name:Name,State:State}' \
  --output table
