#!/usr/bin/env bash
# Container entrypoint: warm the model cache, start the SGLang-Omni engine
# in the background, wait for readiness, then hand off to the RunPod
# serverless handler in the foreground.
set -euo pipefail

cd /workspace

echo "entrypoint: sourcing scripts/cache_config.sh"
# shellcheck disable=SC1091
source scripts/cache_config.sh

echo "entrypoint: running scripts/base_env_check.sh"
scripts/base_env_check.sh || echo "entrypoint: WARN base_env_check reported issues, continuing"

if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]]; then
    echo "entrypoint: ensuring model weights are cached"
    python3 scripts/download_model.py
fi

echo "entrypoint: starting SGLang-Omni engine in background"
scripts/start_engine.sh --background

echo "entrypoint: engine ready, starting RunPod handler"
exec python3 -u handler.py
