#!/usr/bin/env python3
"""Stage 02 — Standalone downloader for Higgs TTS 3 (4B) weights.

Downloads `bosonai/higgs-tts-3-4b` from the Hugging Face Hub into a
persistent cache directory, verifies the expected artifact set is present,
and supports two deployment modes:

  * network-volume pre-warming: cache lives on /runpod-volume and is shared
    across workers (default; skips re-download if already populated).
  * container baking: cache lives inside the image (set BAKE_INTO_IMAGE=1)
    for cold-start-sensitive deployments without a network volume.

Usage:
    python3 download_model.py [--repo-id bosonai/higgs-tts-3-4b] [--revision main]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download_model")

DEFAULT_REPO_ID = "bosonai/higgs-tts-3-4b"
NETWORK_VOLUME_CACHE = "/runpod-volume/huggingface-cache"
LOCAL_BAKE_CACHE = "/opt/hf-cache"

# Files that must exist after a successful download for the model to be
# usable by the SGLang-Omni engine (stage 03).
REQUIRED_SUFFIXES = (
    ".safetensors",
    "config.json",
)
REQUIRED_TOKENIZER_HINTS = (
    "tokenizer",
    "codebook",
)


def resolve_cache_dir() -> str:
    if os.environ.get("BAKE_INTO_IMAGE") == "1":
        cache_dir = os.environ.get("HF_HOME", LOCAL_BAKE_CACHE)
        log.info("BAKE_INTO_IMAGE=1 -> baking weights into image cache at %s", cache_dir)
        return cache_dir

    cache_dir = os.environ.get("HF_HOME", NETWORK_VOLUME_CACHE)
    if os.path.isdir("/runpod-volume"):
        log.info("Network Volume detected -> using %s", cache_dir)
    else:
        log.warning(
            "/runpod-volume not mounted; falling back to local cache at %s "
            "(weights will NOT persist across worker restarts)",
            cache_dir,
        )
    return cache_dir


def verify_snapshot(local_dir: str) -> None:
    """Sanity-check safetensors, tokenizer, and config presence."""
    files = [str(p) for p in Path(local_dir).rglob("*") if p.is_file()]
    if not files:
        raise RuntimeError(f"No files found under {local_dir} after download")

    has_safetensors = any(f.endswith(".safetensors") for f in files)
    has_config = any(f.endswith("config.json") for f in files)
    has_tokenizer = any(
        any(hint in os.path.basename(f).lower() for hint in REQUIRED_TOKENIZER_HINTS)
        for f in files
    )

    missing = []
    if not has_safetensors:
        missing.append("*.safetensors weights")
    if not has_config:
        missing.append("config.json")
    if not has_tokenizer:
        missing.append("tokenizer / multi-codebook tokenizer files")

    if missing:
        raise RuntimeError(f"Integrity check failed, missing: {', '.join(missing)}")

    log.info("Integrity check passed: %d files, safetensors=%s config=%s tokenizer=%s",
              len(files), has_safetensors, has_config, has_tokenizer)


def hub_cache_dir(cache_dir: str) -> str:
    """The actual Hugging Face Hub snapshot cache lives under `<cache_dir>/hub`
    — both huggingface_hub's own HF_HOME-based resolution and RunPod's native
    model cache (`/runpod-volume/huggingface-cache/hub/models--org--name/...`)
    use this exact nesting. Passing `cache_dir` bare (without `/hub`) to
    snapshot_download writes to a path nothing else looks at."""
    return os.path.join(cache_dir, "hub")


def download(repo_id: str, revision: str, cache_dir: str) -> str:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        log.warning("HF_TOKEN not set; proceeding unauthenticated (fails for gated repos)")

    hub_dir = hub_cache_dir(cache_dir)
    os.makedirs(hub_dir, exist_ok=True)
    log.info("Downloading %s@%s into %s ...", repo_id, revision, hub_dir)

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=hub_dir,
        token=token,
        max_workers=8,
        resume_download=True,
    )
    log.info("Download complete: %s", local_dir)
    return local_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("MODEL_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--revision", default=os.environ.get("MODEL_REVISION", "main"))
    args = parser.parse_args()

    cache_dir = resolve_cache_dir()

    try:
        local_dir = download(args.repo_id, args.revision, cache_dir)
        verify_snapshot(local_dir)
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        log.error("Model download/verification failed: %s", exc)
        return 1

    print(local_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
