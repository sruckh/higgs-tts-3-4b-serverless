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

ENV PATH="/root/.local/bin:${PATH}" \
    CUDA_VERSION=12.4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache

RUN command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)

WORKDIR /workspace

# --- Python dependencies (cached layer, changes less often than source) ---
COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

# --- Application source ------------------------------------------------
# .dockerignore strips dot-directories, AGENTS.md, CLAUDE.md, and other
# non-runtime markdown before this hits the daemon.
COPY . /workspace

RUN chmod +x /workspace/entrypoint.sh /workspace/scripts/*.sh

EXPOSE 8000

ENTRYPOINT ["/workspace/entrypoint.sh"]
