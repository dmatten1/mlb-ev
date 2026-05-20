# Deploying the outcomes Lambda

Mirrors the odds-ingestion Lambda you already deployed. Differences:

| Item | Odds | Outcomes |
|---|---|---|
| Source API | the-odds-api.com (paid key) | statsapi.mlb.com (no key) |
| Schedule | 4×/day in ET | 1×/day at 01:30 ET (post-game) |
| S3 prefix | `raw/odds/...` | `raw/outcomes/...` |
| Lambda fn | `mlb-ev-ingest-odds` | `mlb-ev-ingest-outcomes` |
| Deps in zip | requests | MLB-StatsAPI |

Reuse what you already have: same bucket, same logging permissions, same Graviton+Python 3.12 runtime. The only NEW pieces are the Lambda function, a tiny IAM policy delta, and one EventBridge schedule.

## 0. Env vars (re-export each shell session)

```bash
export BUCKET=mlb-ev-dcm92            # your existing odds bucket
export REGION=us-east-1               # whatever you used for odds
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export FUNCTION_NAME=mlb-ev-ingest-outcomes
export ROLE_NAME=mlb-ev-ingest-lambda-role  # reuse the role from odds
export SCHEDULER_ROLE_NAME=mlb-ev-scheduler-role  # reuse from odds
```

## 1. Extend the Lambda execution role to write `raw/outcomes/*`

The odds role's policy is scoped to `raw/odds/*`. Easiest fix: add a second resource. Pull the current policy, patch it, push back:

```bash
# Inspect what's currently attached
aws iam list-role-policies --role-name "$ROLE_NAME"
aws iam get-role-policy --role-name "$ROLE_NAME" \
  --policy-name mlb-ev-ingest-lambda-policy
```

If the existing policy looks like:

```json
{
  "Statement": [
    { "Effect": "Allow", "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::mlb-ev-dcm92/raw/odds/*" },
    { "Effect": "Allow", "Action": "logs:*",
      "Resource": "*" }
  ]
}
```

replace the `Resource` value with an array covering both prefixes:

```bash
cat > /tmp/lambda-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": [
        "arn:aws:s3:::${BUCKET}/raw/odds/*",
        "arn:aws:s3:::${BUCKET}/raw/outcomes/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
      "Resource": "*"
    }
  ]
}
EOF

aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name mlb-ev-ingest-lambda-policy \
  --policy-document file:///tmp/lambda-policy.json
```

## 2. Build the deployment ZIP

```bash
bash infra/build_outcomes_lambda_zip.sh
# -> build/lambda_outcomes.zip  (~300 KB)
```

## 3. Create the Lambda function

```bash
aws lambda create-function \
  --function-name "$FUNCTION_NAME" \
  --runtime python3.12 \
  --architectures arm64 \
  --role "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}" \
  --handler src.ingest.outcomes_lambda_handler.handler \
  --timeout 60 \
  --memory-size 256 \
  --zip-file fileb://build/lambda_outcomes.zip \
  --environment "Variables={OUTCOMES_S3_BUCKET=${BUCKET},OUTCOMES_S3_PREFIX=raw/outcomes,LOG_LEVEL=INFO}" \
  --region "$REGION"
```

(For subsequent code changes use `aws lambda update-function-code --function-name "$FUNCTION_NAME" --zip-file fileb://build/lambda_outcomes.zip`.)

## 4. Verify with a one-off invoke

```bash
# Default: ingest yesterday in UTC
aws lambda invoke \
  --function-name "$FUNCTION_NAME" \
  --cli-binary-format raw-in-base64-out \
  --payload '{}' \
  /tmp/out.json && cat /tmp/out.json

# Force a specific date (e.g. for a replay)
aws lambda invoke \
  --function-name "$FUNCTION_NAME" \
  --cli-binary-format raw-in-base64-out \
  --payload '{"date":"2025-09-15"}' \
  /tmp/out.json && cat /tmp/out.json

# Confirm the object landed
aws s3 ls "s3://${BUCKET}/raw/outcomes/baseball_mlb/2025/"
```

## 5. Schedule it nightly at 01:30 ET

```bash
aws scheduler create-schedule \
  --name outcomes-0130-et \
  --group-name mlb-ev \
  --schedule-expression "cron(30 1 * * ? *)" \
  --schedule-expression-timezone "America/New_York" \
  --flexible-time-window '{"Mode":"OFF"}' \
  --target "{
    \"Arn\":\"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}\",
    \"RoleArn\":\"arn:aws:iam::${ACCOUNT_ID}:role/${SCHEDULER_ROLE_NAME}\"
  }" \
  --region "$REGION"
```

The scheduler role already has `lambda:InvokeFunction` on this Lambda (the policy uses a wildcard `mlb-ev-ingest-*` in most setups). If invocation fails with a 403, extend the scheduler role's policy to include the new function's ARN.

## 6. (One-time) Upload the local 2024-25 backfill to S3

You ran the backfill locally already. Sync it up:

```bash
aws s3 sync data/raw/outcomes/ "s3://${BUCKET}/raw/outcomes/" \
  --exclude "*.DS_Store"

aws s3 ls "s3://${BUCKET}/raw/outcomes/baseball_mlb/2024/" | wc -l
aws s3 ls "s3://${BUCKET}/raw/outcomes/baseball_mlb/2025/" | wc -l
```

## Cost notes

- Lambda: 1 invoke/day × ~1 second × 256 MB = essentially free under the 1M-request/400k-GB-second monthly free tier.
- S3 PUTs: ~365 PUTs/year of ~5–10 KB each ≈ pennies/year.
- EventBridge Scheduler: free for the first 14M invocations/month.
- MLB-StatsAPI: free, no key required.

Total marginal cost of the outcomes pipeline: **<$0.01/month**.
