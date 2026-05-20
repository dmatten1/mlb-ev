# Serverless cloud pipeline (FinOps-friendly)

Architecture:

```text
EventBridge (4×/day, America/Chicago)
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

## 5. EventBridge schedule (4×/day Central)

Keep your existing EventBridge rule on **`mlb-ev-ingest-odds`** only. Example cron (8:00 / 11:30 / 17:00 / 20:00 **US Central** — same instants as 9 / 12:30 / 18 / 21 Eastern):

```text
cron(0 8,11,17,20 * * ? *)
```

Timezone: `America/Chicago` in the scheduler target.

You do **not** need a separate rule for inference if the chain is configured.

## 6. S3 static website (dashboard)

Automated setup (enables website hosting, public read on `index.html` only, uploads local dashboard if present):

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

- **4 odds API calls/day** + **4 inference runs/day** ≈ 8 Lambda invocations (inference may run 1–5 min each).
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
