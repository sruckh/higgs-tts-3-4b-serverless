#!/usr/bin/env bash
# Stage 02 — Hugging Face cache environment overrides for higgs-tts3-runpod.
# Source this file (`source cache_config.sh`) before running download_model.py
# or launching the SGLang-Omni engine so all stages agree on cache paths.
set -euo pipefail

# --- Model identity ----------------------------------------------------
export MODEL_REPO_ID="${MODEL_REPO_ID:-bosonai/higgs-tts-3-4b}"
export MODEL_REVISION="${MODEL_REVISION:-main}"

# --- Cache location ------------------------------------------------------
# Prefer the RunPod Network Volume so weights persist across worker
# restarts and are shared by every worker on the endpoint. Falls back to a
# local image-baked path when no volume is mounted (BAKE_INTO_IMAGE=1) or
# when /runpod-volume is absent (e.g. local dev).
if [[ "${BAKE_INTO_IMAGE:-0}" == "1" ]]; then
    export HF_HOME="${HF_HOME:-/opt/hf-cache}"
elif [[ -d /runpod-volume ]]; then
    export HF_HOME="${HF_HOME:-/runpod-volume/huggingface-cache}"
else
    echo "WARN: /runpod-volume not found; using local fallback cache" >&2
    export HF_HOME="${HF_HOME:-/opt/hf-cache}"
fi

export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_ENABLE_HF_TRANSFER_ERROR_ON_MISSING=0

mkdir -p "${HF_HOME}"

echo "cache_config: MODEL_REPO_ID=${MODEL_REPO_ID}"
echo "cache_config: MODEL_REVISION=${MODEL_REVISION}"
echo "cache_config: HF_HOME=${HF_HOME}"
