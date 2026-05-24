# Serverless cloud pipeline (FinOps-friendly)

Architecture:

```text
EventBridge (9×/day, America/New_York)
    → Lambda zip (odds ingest) → S3 raw/odds/...
    → async invoke Lambda container (inference)
        → download pipeline/data artifacts from S3
        → live_refresh (outcomes, schedule, predict, bet log, CLV, dashboard)
        → upload bet_log + HTML → S3
        → publish index.html to static website bucket
```

No always-on ECS/Fargate. You pay for Lambda seconds + pennies of S3.

## Prerequisites

- Existing **odds** Lambda (`mlb-ev-ingest-odds`) + S3 bucket (e.g. `mlb-ev-dcm92`)
- Local machine with Docker, AWS CLI, and a fresh **`make refresh`** artifact set
- **The Odds API** key in Lambda env (odds function)

## 1. Upload pipeline artifacts (one-time / after each `make refresh`)

From your laptop (where `data/` is current):

```bash
export BUCKET=mlb-ev-dcm92
export YEAR=2026
bash infra/sync_artifacts_to_s3.sh
```

This copies training parquets, model pickle, lineups, statcast, OAA, park factors, and optional bet log to `s3://$BUCKET/pipeline/data/`.

**Model pickle:** train with the same library versions as the inference container. A local venv on pandas 3 / sklearn 1.8 produces pickles that **fail to load** in Lambda (pandas 2.2 / sklearn 1.5). After rebuilding features:

```bash
bash infra/train_model_lambda_compat.sh
bash infra/sync_artifacts_to_s3.sh
```

## 2. Build & push inference container (ECR)

**Prerequisite — Docker must be running on your Mac.** The inference Lambda uses a **container image**; you build it locally (or on any machine with Docker) and push to ECR.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) if you have not already.
2. **Open Docker Desktop** and wait until it says **Docker is running** (menu-bar whale icon is steady, not “starting”).
3. Verify the daemon responds:

```bash
docker info
```

If you see `Cannot connect to the Docker daemon` or `docker.sock: no such file or directory`, Docker Desktop is **not** running — start it and retry.

Then build and push:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
chmod +x infra/build_inference_lambda_image.sh
bash infra/build_inference_lambda_image.sh
# Note the printed image URI
```

**No Docker on this laptop?** Use any other machine with Docker + AWS CLI (work PC, Linux VM, GitHub Actions self-hosted runner), clone the repo, run the same script, then continue with step 3 from your Mac. The image lives in **your** ECR account once pushed.

## 3. Create inference Lambda (container)

Use the setup script (avoids **zsh** parsing errors from `Variables={...}` on the CLI and fixes IAM JSON):

```bash
export BUCKET=mlb-ev-dcm92
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/mlb-ev-inference:latest"

chmod +x infra/create_inference_lambda.sh
bash infra/create_inference_lambda.sh
```

The role `mlb-ev-inference-lambda-role` may already exist from a partial run; the script updates the policy and creates or updates the function.

Handler is set in the Dockerfile: `src.inference.inference_lambda_handler.handler`.

**If you already saw errors:**

| Error | Cause |
|-------|--------|
| `zsh: missing delimiter for 'u' glob qualifier` | Pasting multi-line `--environment Variables={...}` — zsh treats `{` specially. Use the script above. |
| `MalformedPolicyDocument` | Bad JSON from a broken paste; re-run `put-role-policy` via the script. |
| `Function not found` on `update-function-code` | `create-function` never succeeded; the script calls `create-function` when missing. |
| `image manifest ... media type ... is not supported` | ECR image has BuildKit attestations or wrong arch. **Re-run step 2** (script disables provenance/SBOM), then step 3. |

## 4. Chain odds → inference

**What this step does (plain English):**

1. Redeploy the **odds** Lambda zip so it includes code that calls inference after each snapshot.
2. Set env var **`INFERENCE_LAMBDA_NAME=mlb-ev-inference`** on the odds function (your existing `ODDS_API_KEY` / `ODDS_S3_BUCKET` are left as-is).
3. Give the **odds Lambda’s IAM role** permission to **invoke** the inference Lambda. Without this, odds succeeds but the chain fails with **access denied** in CloudWatch.

The odds function and inference function use **different** IAM roles. Step 3 created `mlb-ev-inference-lambda-role`. Your odds function already uses **`mlb-ev-ingest-lambda-role`** (from when you first deployed odds). “Attach to the odds execution role” only means: add `lambda:InvokeFunction` on the inference ARN to **that** role — not a new role.

**One command** (needs `jq`: `brew install jq`):

```bash
export BUCKET=mlb-ev-dcm92
export AWS_REGION=us-east-1
bash infra/chain_odds_lambda.sh
```

That script rebuilds `build/lambda.zip`, updates `mlb-ev-ingest-odds`, merges env vars, and adds inline policy `mlb-ev-invoke-inference` on `mlb-ev-ingest-lambda-role`.

After each successful odds snapshot, the odds handler **async-invokes** inference (`InvocationType=Event`).

## 5. EventBridge schedule (9×/day Eastern)

Odds ingest runs **9 times/day** on **`mlb-ev-ingest-odds`** (each invoke chains async to inference):

| ET | Schedule name |
|----|----------------|
| 9:00 AM | `odds-0900-et` |
| 12:00 PM | `odds-1200-et` |
| 1:00 PM | `odds-1300-et` |
| 2:30 PM | `odds-1430-et` |
| 4:00 PM | `odds-1600-et` |
| 5:30 PM | `odds-1730-et` |
| 7:00 PM | `odds-1900-et` |
| 8:30 PM | `odds-2030-et` |
| 10:00 PM | `odds-2200-et` |

Install or refresh all schedules:

```bash
export AWS_REGION=us-east-1
bash infra/setup_odds_schedules.sh
```

Timezone: `America/New_York` (DST-aware). You do **not** need a separate rule for inference if the chain is configured.

### Live scores on pending bets (every 10 minutes)

The dashboard **Live** column shows current score + inning for pending rows (MLB Stats API — **free**, no key). A lightweight inference invoke re-renders HTML and publishes `index.html` every **10 minutes**; the page also auto-reloads every 10 minutes while bets are pending.

```bash
export AWS_REGION=us-east-1
bash infra/setup_scores_refresh_schedule.sh
```

Schedule: `scores-10min-et` → `mlb-ev-inference` with `{"mode":"scores_refresh"}`. Skips work when there are no pending bets (~2–5 s when it runs). **~144 invocations/day** — still essentially $0 on Lambda free tier.

Hover a **Live** cell for detail (e.g. “Top 7th · Live” or first-pitch time pre-game).

## 6. S3 static website (dashboard)

Automated setup (enables website hosting, public read on `index.html` only; does **not** overwrite an existing `index.html` unless `UPLOAD_DASHBOARD=1`):

```bash
export BUCKET=mlb-ev-dcm92
export AWS_REGION=us-east-1
bash infra/setup_dashboard_website.sh
```

**Your dashboard URL** (after the script runs):

`http://mlb-ev-dcm92.s3-website-us-east-1.amazonaws.com/`

The inference Lambda (step 3) should already have `DASHBOARD_S3_BUCKET` and `DASHBOARD_S3_KEY=index.html`; each successful inference run overwrites `index.html` in the bucket.

**Note:** Only `index.html` is world-readable; odds/outcomes under `raw/` stay private.

## 7. Verify

```bash
aws lambda invoke --function-name mlb-ev-ingest-odds --region "$REGION" /tmp/odds-out.json
# Wait ~2–5 min, check inference logs:
aws logs tail /aws/lambda/mlb-ev-inference --follow --region "$REGION"
```

Check S3 for updated `index.html` and `pipeline/data/tracking/bet_log.parquet`.

## Paper-trading semantics

The bet log uses **first-touch locks** — once a game is logged, later runs cannot rewrite price/side. See `infra/paper_trade.md`.

## Cost notes

- **9 odds API calls/day** + **9 full inference runs/day** + **~144 live-score refreshes/day** (scores-only path when pending bets exist).
- **S3** storage for parquets + website: typically &lt; $1/month at this scale.
- Re-upload artifacts with `sync_artifacts_to_s3.sh` after major local `make refresh` runs.

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `failed to connect to the docker API` / `docker.sock: no such file` | Start **Docker Desktop** and wait until running; run `docker info` |
| `zsh: missing delimiter for 'u' glob qualifier` (step 3) | Do not paste inline `Variables={...}`; run `bash infra/create_inference_lambda.sh` |
| `MalformedPolicyDocument` (step 3) | Re-run `bash infra/create_inference_lambda.sh` (writes valid JSON under `build/`) |
| `image manifest ... media type ... is not supported` (step 3) | Re-run **step 2** with current `build_inference_lambda_image.sh` (`--provenance=false --sbom=false`), then step 3 again |
| Inference times out | Increase timeout to 900s; memory 3008 MB |
| `missing artifacts` in logs | Run `sync_artifacts_to_s3.sh` |
| Predict 0 games | Schedule JSON missing — inference run fetches schedule via Stats API into `/tmp` then uploads |
| Odds works, no inference | Set `INFERENCE_LAMBDA_NAME`; check odds role `lambda:InvokeFunction` |
| Dashboard 403 | Bucket policy / public access block settings |
| No email alerts | Run `bash infra/setup_email_alerts.sh` with `ALERT_EMAIL=...`; confirm SNS subscription |

## 8. Email alerts (optional)

Get notified when odds or inference fails (401 API key, predict/track errors, Lambda crashes):

```bash
export ALERT_EMAIL=you@example.com
export AWS_REGION=us-east-1
bash infra/setup_email_alerts.sh
```

**Important:** AWS sends a **Confirm subscription** email — click the link or alerts are silently dropped.

Alarms created:

| Alarm | Fires when |
|-------|------------|
| `mlb-ev-odds-lambda-errors` | Uncaught exception in odds Lambda |
| `mlb-ev-odds-snapshot-failed` | Log line with 401 / `odds snapshot failed` |
| `mlb-ev-inference-lambda-errors` | Uncaught exception in inference Lambda |
| `mlb-ev-inference-partial-failure` | `live_refresh` partial failure or failed predict/track step |

After rotating your Odds API key locally, push it to Lambda:

```bash
bash infra/update_odds_api_key.sh
```
