#!/usr/bin/env bash
# Launch SGLang-Omni engine serving Higgs TTS 3 (4B) and block until the
# /health endpoint reports ready.
#
# Usage: ./start_engine.sh [--background]
#   --background   start the server detached and return once healthy
#                   (used by entrypoint.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull in cache paths / model repo ID, if not already sourced by the caller.
CACHE_CONFIG="${SCRIPT_DIR}/cache_config.sh"
if [[ -z "${MODEL_REPO_ID:-}" && -f "${CACHE_CONFIG}" ]]; then
    # shellcheck disable=SC1090
    source "${CACHE_CONFIG}"
fi

MODEL_PATH="${MODEL_REPO_ID:-bosonai/higgs-tts-3-4b}"
HOST="127.0.0.1"
PORT="8000"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
HEALTH_URL="http://${HOST}:${PORT}/health"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-2}"

RUN_BACKGROUND=0
if [[ "${1:-}" == "--background" ]]; then
    RUN_BACKGROUND=1
fi

echo "start_engine: launching sgl-omni serve --model-path ${MODEL_PATH} --port ${PORT} --host ${HOST} --tp ${TP_SIZE} --mem-fraction-static ${MEM_FRACTION_STATIC}"

launch_cmd=(sgl-omni serve
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --tp "${TP_SIZE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}")

if [[ "${RUN_BACKGROUND}" -eq 1 ]]; then
    "${launch_cmd[@]}" >/workspace/engine.log 2>&1 &
    ENGINE_PID=$!
    echo "start_engine: engine started in background, pid=${ENGINE_PID}, log=/workspace/engine.log"
else
    "${launch_cmd[@]}" &
    ENGINE_PID=$!
fi

# --- Readiness probe --------------------------------------------------------
echo "start_engine: waiting for ${HEALTH_URL} (timeout ${READY_TIMEOUT_SECONDS}s)..."
elapsed=0
until curl -fsS -o /dev/null -w '%{http_code}' "${HEALTH_URL}" 2>/dev/null | grep -q '^200$'; do
    if ! kill -0 "${ENGINE_PID}" 2>/dev/null; then
        echo "start_engine: engine process ${ENGINE_PID} exited unexpectedly" >&2
        exit 1
    fi
    if (( elapsed >= READY_TIMEOUT_SECONDS )); then
        echo "start_engine: timed out waiting for ${HEALTH_URL} after ${READY_TIMEOUT_SECONDS}s" >&2
        exit 1
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
    elapsed=$(( elapsed + POLL_INTERVAL_SECONDS ))
done

echo "start_engine: engine healthy after ~${elapsed}s (pid=${ENGINE_PID})"

# --- GPU memory / CUDA graph warm-up check ----------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
fi

if [[ "${RUN_BACKGROUND}" -eq 0 ]]; then
    wait "${ENGINE_PID}"
fi
