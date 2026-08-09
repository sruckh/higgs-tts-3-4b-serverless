#!/usr/bin/env bash
# Stage 01 — Validation script for CUDA / Python / audio-lib compatibility.
# Run inside the built base image to confirm the environment matches
# _config/env.json before proceeding to stage 02.
set -euo pipefail

REQUIRED_CUDA="12.4"
REQUIRED_PYTHON="3.12"
MIN_VRAM_GB=32

fail() { echo "FAIL: $1" >&2; exit 1; }
ok()   { echo "OK:   $1"; }

echo "== higgs-tts3-runpod :: base environment check =="

# --- Python version --------------------------------------------------------
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
if [[ "${PY_VERSION}" != "${REQUIRED_PYTHON}" ]]; then
    fail "python3 is ${PY_VERSION:-missing}, expected ${REQUIRED_PYTHON}"
fi
ok "python3 ${PY_VERSION}"

# --- uv availability ---------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    fail "uv not found on PATH"
fi
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

# --- CUDA toolkit / driver -----------------------------------------------
if command -v nvcc >/dev/null 2>&1; then
    CUDA_FOUND="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
elif command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_FOUND="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}')"
else
    fail "neither nvcc nor nvidia-smi found; cannot detect CUDA version"
fi
if [[ -z "${CUDA_FOUND}" ]]; then
    fail "unable to parse CUDA version"
fi
ok "CUDA ${CUDA_FOUND} detected (required >= ${REQUIRED_CUDA})"

# --- GPU VRAM ---------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)"
    VRAM_GB=$(( VRAM_MIB / 1024 ))
    if (( VRAM_GB < MIN_VRAM_GB )); then
        fail "GPU has ${VRAM_GB}GiB VRAM, need >= ${MIN_VRAM_GB}GiB"
    fi
    ok "GPU VRAM ${VRAM_GB}GiB (required >= ${MIN_VRAM_GB}GiB)"
else
    echo "WARN: nvidia-smi unavailable, skipping VRAM check (non-GPU host?)"
fi

# --- Audio system libraries -----------------------------------------------
for bin in ffmpeg; do
    command -v "${bin}" >/dev/null 2>&1 || fail "${bin} not found"
    ok "${bin} present"
done

ldconfig -p 2>/dev/null | grep -q libsndfile.so || fail "libsndfile1 not found"
ok "libsndfile1 present"

# --- Env vars ---------------------------------------------------------------
[[ "${PYTHONUNBUFFERED:-}" == "1" ]] || echo "WARN: PYTHONUNBUFFERED is not set to 1"
[[ -n "${HF_HOME:-}" ]] || echo "WARN: HF_HOME is not set"

echo "== base environment check complete =="
