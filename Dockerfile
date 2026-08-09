# higgs-tts3-runpod — RunPod Serverless worker for Higgs TTS 3 (4B).
# Built from the project root; .dockerignore excludes agent/IDE meta
# directories (.icm/, .claude/, etc.) and AGENTS.md/CLAUDE.md from the
# build context.
FROM lmsysorg/sglang-omni:dev

LABEL maintainer="higgs-tts3-runpod" \
      description="Higgs TTS 3 (4B) RunPod Serverless worker"

# --- System dependencies ----------------------------------------------
# ffmpeg / libsndfile1: audio decode/encode and reference-audio
# preprocessing for zero-shot voice cloning.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# RUNPOD_SKIP_GPU_CHECK / RUNPOD_SKIP_AUTO_SYSTEM_CHECKS: the RunPod SDK's
# own post-model-load fitness checks (GPU memory probe, CUDA init) can
# false-positive OOM on otherwise-healthy heavy-VRAM workers and mark them
# unhealthy; skip them.
#
# RUNPOD_INIT_TIMEOUT: RunPod marks a worker unhealthy — and kills it, often
# before any of our own logs ship — once cold start passes 7 minutes
# (https://docs.runpod.io/serverless/development/optimization, verified
# 2026-08-09). A cache-miss cold start here (network model download, then
# sgl-omni engine warm-up) can plausibly exceed that default, which produces
# exactly the "worker exited with exit code 1, no logs" symptom. 1200s gives
# headroom above ENGINE_READY_TIMEOUT_SECONDS's own 600s engine-health poll
# plus download time.
#
# HF_HUB_ENABLE_HF_TRANSFER=1 requires the `hf_transfer` package
# (requirements.txt): huggingface_hub raises an unhandled ValueError the
# instant this flag is set without it installed, which crashes the whole
# module-scope bootstrap with a fast, immediately-repeating "exit code 1" —
# confirmed as the actual cause of the 2026-08-09 crash loop. Do not set
# this flag without keeping the pin in requirements.txt.
ENV PATH="/root/.local/bin:${PATH}" \
    CUDA_VERSION=12.4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache \
    RUNPOD_SKIP_GPU_CHECK=true \
    RUNPOD_SKIP_AUTO_SYSTEM_CHECKS=true \
    RUNPOD_INIT_TIMEOUT=1200

RUN command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)

WORKDIR /workspace

# --- Python dependencies (cached layer, changes less often than source) ---
# --break-system-packages: the base image's /usr Python is PEP 668
# "externally managed" (Debian); this container only ever runs this app, so
# installing into system Python is intentional, not accidental.
COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

# --- Application source ------------------------------------------------
# .dockerignore strips dot-directories, AGENTS.md, CLAUDE.md, and other
# non-runtime markdown before this hits the daemon.
COPY . /workspace

RUN chmod +x /workspace/scripts/*.sh

EXPOSE 8000

# handler.py must be the container's process from the first instant: RunPod's
# setup-time validator and worker supervisor both need to see a live
# runpod-importing Python process almost immediately, which a separate bash
# orchestration script (waiting on model download + engine warm-up before
# ever exec'ing into Python) can't provide. handler.py does its own
# cold-start bootstrap at module scope before runpod.serverless.start().
#
# ENTRYPOINT [] resets any ENTRYPOINT the base image carries (unverified for
# lmsysorg/sglang-omni:dev) — inheriting one would turn CMD into arguments
# appended to it instead of the actual start command, which has separately
# caused this exact "worker exits, zero logs" symptom before.
ENTRYPOINT []
CMD ["python3", "-u", "/workspace/handler.py"]
