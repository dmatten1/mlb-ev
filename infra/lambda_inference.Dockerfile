# AWS Lambda container image for inference + paper-trading tracker.
# Build:  bash infra/build_inference_lambda_image.sh
# Deploy: see infra/cloud_deploy.md

FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLB_EV_REPO_ROOT=/tmp/mlb-ev \
    TZ=America/Chicago

COPY requirements.txt requirements-refresh.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-refresh.txt

COPY src ${LAMBDA_TASK_ROOT}/src

CMD ["src.inference.inference_lambda_handler.handler"]
