#!/usr/bin/env bash
# Train runs_model_bullpen_cached.pkl with the same deps as the inference Lambda
# (pandas 2.2 + sklearn 1.5). Local venvs on pandas 3 / sklearn 1.8 produce pickles
# that fail to load in Lambda (StringDtype / InconsistentVersionWarning).
#
# Usage:
#   bash infra/train_model_lambda_compat.sh
#   export BUCKET=mlb-ev-dcm92 && bash infra/sync_artifacts_to_s3.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker must be running." >&2
  exit 1
fi

echo "==> Training in Lambda-compatible image (pandas 2.2 / sklearn 1.5)"
docker run --rm --entrypoint python \
  -v "${REPO_ROOT}:/var/task" -w /var/task \
  public.ecr.aws/lambda/python:3.12 \
  -c "
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-refresh.txt'])
import pandas as pd
from pathlib import Path
from src.model.runs_model import BULLPEN_FEATURE_COLS, train_runs_model, save_runs_model

train = pd.concat([
    pd.read_parquet('data/features/training_2023.parquet'),
    pd.read_parquet('data/features/training_2024.parquet'),
], ignore_index=True)
rm = train_runs_model(train, BULLPEN_FEATURE_COLS)
out = Path('data/models/runs_model_bullpen_cached.pkl')
save_runs_model(rm, out)
print('saved', out, 'train_n=', rm.train_n, 'features=', len(rm.feature_cols))
"

echo "==> Verify pickle loads in same image"
docker run --rm --entrypoint python \
  -v "${REPO_ROOT}:/var/task" -w /var/task \
  public.ecr.aws/lambda/python:3.12 \
  -c "
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-refresh.txt'])
from src.model.runs_model import load_runs_model
rm = load_runs_model('data/models/runs_model_bullpen_cached.pkl')
print('load ok train_n=', rm.train_n)
"

echo "==> Done. Upload with: BUCKET=mlb-ev-dcm92 bash infra/sync_artifacts_to_s3.sh"
