#!/usr/bin/env python3
"""RunPod Serverless handler for higgs-tts3-runpod.

Bridges RunPod job input (see `_config/API_SCHEMA.json`) to a local
SGLang-Omni engine's OpenAI-compatible `/v1/audio/speech` endpoint,
returning either a unary base64-encoded audio payload or a generator of SSE
audio chunks when `stream=True`.

RunPod needs to see a live `runpod`-importing Python process almost
immediately, or its setup-time validator and worker supervisor both treat
the container as broken. So — unlike an earlier version of this file —
nothing here waits behind a separate bash entrypoint: this module does its
own cold-start bootstrap (env check, model download, engine launch +
health poll) at import time, before `runpod.serverless.start()` is ever
called, mirroring RunPod's own "load heavy state at module scope" pattern
for long-cold-start workers.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import requests
import runpod

# Unbuffered stdout: log lines only help if they're flushed before a slow
# cold start gets torn down by RunPod's worker supervisor.
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import download_model as _download_model  # noqa: E402

from schema_validator import ValidationError, validate_engine_response, validate_job_input  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("handler")

ENGINE_HOST = "127.0.0.1"
ENGINE_PORT = 8000
ENGINE_BASE_URL = f"http://{ENGINE_HOST}:{ENGINE_PORT}"
SPEECH_ENDPOINT = f"{ENGINE_BASE_URL}/v1/audio/speech"
HEALTH_ENDPOINT = f"{ENGINE_BASE_URL}/health"
ENGINE_LOG_PATH = "/workspace/engine.log"

REQUEST_TIMEOUT_SECONDS = 120
STREAM_CHUNK_TIMEOUT_SECONDS = 60
ENGINE_READY_TIMEOUT_SECONDS = int(os.environ.get("ENGINE_READY_TIMEOUT_SECONDS", "600"))
ENGINE_POLL_INTERVAL_SECONDS = 2

# Reference clips arrive as base64 in the job input (callers upload them;
# they have no access to the RunPod Network Volume). Decode each one to a
# short-lived local file the engine can read, then delete it once the job
# finishes — reference audio never touches persistent storage.
REFERENCE_TMP_DIR = os.environ.get("REFERENCE_TMP_DIR", "/tmp/higgs-refs")

_engine_process: subprocess.Popen[bytes] | None = None


# --------------------------------------------------------------------------
# Cold-start bootstrap — runs once at module import, before
# runpod.serverless.start(). Everything here must print with flush=True.
# --------------------------------------------------------------------------


def _print_diagnostics() -> None:
    print("=" * 75, flush=True)
    print("=== higgs-tts3-runpod worker :: startup diagnostics ===", flush=True)
    print(f"[DEBUG] Python:          {sys.version.split()[0]}", flush=True)
    print(f"[DEBUG] runpod SDK:      {getattr(runpod, '__version__', 'unknown')}", flush=True)
    print(
        f"[DEBUG] MODEL_REPO_ID:   {os.environ.get('MODEL_REPO_ID', _download_model.DEFAULT_REPO_ID)}",
        flush=True,
    )
    print(
        f"[DEBUG] HF_TOKEN set:    {bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))}",
        flush=True,
    )
    print(f"[DEBUG] Network Volume:  {os.path.isdir('/runpod-volume')}", flush=True)
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpu_info = gpu.stdout.strip() or gpu.stderr.strip() or "no output"
    except (OSError, subprocess.SubprocessError) as exc:
        gpu_info = f"nvidia-smi unavailable ({exc})"
    print(f"[DEBUG] GPU:             {gpu_info}", flush=True)
    print("=" * 75, flush=True)


def _run_env_check() -> None:
    """Best-effort, non-fatal CUDA/audio-lib sanity check — never blocks
    startup, only informs the log."""
    script = Path(__file__).resolve().parent / "scripts" / "base_env_check.sh"
    try:
        result = subprocess.run(
            [str(script)], capture_output=True, text=True, timeout=30, check=False
        )
        print(result.stdout, flush=True)
        if result.returncode != 0:
            print(
                f"[BOOTSTRAP] base_env_check.sh reported issues (exit {result.returncode}), continuing",
                flush=True,
            )
            print(result.stderr, flush=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[BOOTSTRAP] base_env_check.sh failed to run: {exc}", flush=True)


def _resolve_cached_snapshot(cache_dir: str, repo_id: str) -> str | None:
    """Resolve an already-present local HF snapshot with no network access,
    following RunPod's cached-model layout (`<cache_dir>/hub/models--org--
    name/...`) and its documented resolution order: `refs/main` first, else
    the newest non-empty `snapshots/` directory. Returns None if nothing
    usable is cached yet — RunPod's own model-cache pre-staging and any
    prior worker's download both land here, so this is what lets a warm
    Network Volume (or RunPod's native cache) skip the network entirely."""
    if "/" not in repo_id:
        return None
    org, name = repo_id.split("/", 1)
    model_dir = Path(_download_model.hub_cache_dir(cache_dir)) / f"models--{org}--{name}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    ref_main = model_dir / "refs" / "main"
    if ref_main.is_file():
        candidate = snapshots_dir / ref_main.read_text().strip()
        if candidate.is_dir() and any(candidate.iterdir()):
            return str(candidate)

    non_empty = [p for p in sorted(snapshots_dir.iterdir()) if p.is_dir() and any(p.iterdir())]
    return str(non_empty[-1]) if non_empty else None


def _ensure_model_cached() -> None:
    if os.environ.get("SKIP_MODEL_DOWNLOAD") == "1":
        print("[BOOTSTRAP] SKIP_MODEL_DOWNLOAD=1, skipping model download", flush=True)
        return

    cache_dir = _download_model.resolve_cache_dir()
    repo_id = os.environ.get("MODEL_REPO_ID", _download_model.DEFAULT_REPO_ID)
    revision = os.environ.get("MODEL_REVISION", "main")

    cached_snapshot = _resolve_cached_snapshot(cache_dir, repo_id)
    if cached_snapshot is not None:
        # RunPod's native model cache (endpoint "Model" field) or a prior
        # worker's download already staged this — runtime download is a
        # documented anti-pattern when a cache hit is available, so skip it
        # entirely rather than re-verifying over the network.
        print(f"[BOOTSTRAP] cached snapshot found at {cached_snapshot} — skipping network download", flush=True)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return

    print(f"[BOOTSTRAP] no cached snapshot; downloading {repo_id}@{revision} to {cache_dir} ...", flush=True)
    local_dir = _download_model.download(repo_id, revision, cache_dir)
    _download_model.verify_snapshot(local_dir)
    print("[BOOTSTRAP] model weights verified", flush=True)


def _engine_healthy(timeout: float = 2.0) -> bool:
    try:
        return requests.get(HEALTH_ENDPOINT, timeout=timeout).status_code == 200
    except requests.exceptions.RequestException:
        return False


def _ensure_engine_running() -> None:
    """Launch `sgl-omni serve` as a child of this process and block until
    it's healthy. If an engine is already answering /health (e.g. started
    manually for local testing), skip launching a duplicate."""
    global _engine_process

    if _engine_healthy():
        print("[BOOTSTRAP] engine already healthy (started externally), skipping launch", flush=True)
        return

    model_path = os.environ.get("MODEL_REPO_ID", _download_model.DEFAULT_REPO_ID)
    # SGLang's default --mem-fraction-static (~0.88 of total VRAM) reserves
    # far more than the ~9.3GB bf16 weights need, sized for high concurrency.
    # That default can overshoot on real "32GB" cards, which often report a
    # bit under 32GiB (e.g. RTX 5090 reports 31.8GiB) — cap it explicitly so
    # this fits comfortably on a 32GB card. Lower for smaller GPUs, raise for
    # more KV-cache headroom / concurrency on bigger ones.
    mem_fraction_static = os.environ.get("MEM_FRACTION_STATIC", "0.75")
    print(
        f"[BOOTSTRAP] launching sgl-omni serve --model-path {model_path} "
        f"--port {ENGINE_PORT} --mem-fraction-static {mem_fraction_static} "
        f"(log: {ENGINE_LOG_PATH})",
        flush=True,
    )

    log_file = open(ENGINE_LOG_PATH, "wb")
    try:
        _engine_process = subprocess.Popen(
            [
                "sgl-omni",
                "serve",
                "--model-path",
                model_path,
                "--host",
                ENGINE_HOST,
                "--port",
                str(ENGINE_PORT),
                "--mem-fraction-static",
                mem_fraction_static,
                # No --tp: sgl-omni's HiggsTtsPipelineConfig (pydantic,
                # extra_forbidden) rejects an unrecognized "tp" field
                # outright — confirmed live 2026-08-10
                # (ValidationError: Extra inputs are not permitted). The
                # official Higgs TTS cookbook launch command never passes
                # --tp either. TP_SIZE is intentionally unused until a
                # correct flag/mechanism for this pipeline is confirmed.
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "'sgl-omni' executable not found on PATH — the sglang-omni CLI is not "
            "installed in this image; the Dockerfile assumes lmsysorg/sglang-omni:dev "
            "provides it and may need an explicit `pip install sglang-omni` step"
        ) from exc

    print(
        f"[BOOTSTRAP] waiting for {HEALTH_ENDPOINT} (timeout {ENGINE_READY_TIMEOUT_SECONDS}s)...",
        flush=True,
    )
    deadline = time.monotonic() + ENGINE_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _engine_process.poll() is not None:
            _dump_engine_log()
            raise RuntimeError(
                f"sgl-omni engine exited early with code {_engine_process.returncode}; see {ENGINE_LOG_PATH}"
            )
        if _engine_healthy():
            print("[BOOTSTRAP] engine healthy", flush=True)
            return
        time.sleep(ENGINE_POLL_INTERVAL_SECONDS)

    _dump_engine_log()
    raise RuntimeError(
        f"engine did not become healthy within {ENGINE_READY_TIMEOUT_SECONDS}s; see {ENGINE_LOG_PATH}"
    )


def _dump_engine_log(tail_bytes: int = 20_000) -> None:
    """Print sgl-omni's own log to our stdout before the worker dies.

    ENGINE_LOG_PATH lives on ephemeral container storage — it does not
    survive the worker (it is not on `/runpod-volume`), and RunPod's log
    pipeline only ever captures this process's stdout/stderr. Without this,
    a crashed engine's actual error is silently lost the instant the
    container exits, leaving only "exited early with code 1" behind.
    """
    print(f"[BOOTSTRAP] ---- begin {ENGINE_LOG_PATH} (last {tail_bytes} bytes) ----", flush=True)
    try:
        with open(ENGINE_LOG_PATH, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            content = f.read().decode(errors="replace")
        print(content, flush=True)
    except OSError as exc:
        print(f"[BOOTSTRAP] could not read {ENGINE_LOG_PATH}: {exc}", flush=True)
    print(f"[BOOTSTRAP] ---- end {ENGINE_LOG_PATH} ----", flush=True)


def _bootstrap() -> None:
    try:
        _print_diagnostics()
        _run_env_check()
        _ensure_model_cached()
        _ensure_engine_running()
    except BaseException:
        # A worker that dies here can be killed by RunPod's supervisor (or
        # exit) faster than its stdout/stderr get shipped, which is why
        # crash-loop investigations have repeatedly seen "exited with exit
        # code 1" and nothing else. Print the full traceback explicitly,
        # flush both streams, and hold briefly so the log pipeline has a
        # real chance to capture it before the process actually dies.
        import traceback

        print("[BOOTSTRAP] FATAL: unhandled exception during cold-start bootstrap", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(5)
        raise
    print("[BOOTSTRAP] worker is warm and ready for jobs", flush=True)


# --------------------------------------------------------------------------
# Per-job handling
# --------------------------------------------------------------------------


def _materialize_references(references: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Decode base64 reference audio into local temp files for the engine.

    Returns (engine_references, temp_paths); engine_references use
    `audio_path` since the engine reads local files, while temp_paths lets
    the caller clean up after the job completes.
    """
    os.makedirs(REFERENCE_TMP_DIR, exist_ok=True)
    engine_refs: list[dict[str, Any]] = []
    temp_paths: list[str] = []
    for ref in references:
        audio_bytes = base64.b64decode(ref["audio_base64"])
        suffix = f".{ref.get('audio_format', 'wav')}"
        fd, path = tempfile.mkstemp(prefix="ref-", suffix=suffix, dir=REFERENCE_TMP_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        temp_paths.append(path)
        engine_refs.append({"audio_path": path, "text": ref["text"]})
    return engine_refs, temp_paths


def _cleanup_temp_paths(paths: list[str]) -> None:
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _build_engine_payload(job_input: dict[str, Any], engine_references: list[dict[str, Any]]) -> dict[str, Any]:
    """Translate validated RunPod job input into an OpenAI speech request."""
    return {
        "model": job_input["model"],
        "input": job_input["input"],
        "voice": job_input.get("voice"),
        "references": engine_references,
        "response_format": job_input["response_format"],
        "speed": job_input["speed"],
        "temperature": job_input["temperature"],
        "top_k": job_input["top_k"],
        "stream": job_input["stream"],
    }


def _stream_audio_chunks(
    payload: dict[str, Any], temp_paths: list[str]
) -> Generator[dict[str, Any], None, None]:
    """Yield base64-encoded audio chunks as they arrive over SSE from the
    local engine. Each yielded dict matches RunPod's streaming output
    convention (`runpod.serverless.start` forwards generator yields as
    incremental job results)."""
    try:
        with requests.post(
            SPEECH_ENDPOINT,
            json=payload,
            stream=True,
            timeout=(10, STREAM_CHUNK_TIMEOUT_SECONDS),
        ) as resp:
            validate_engine_response(resp.status_code, resp.headers.get("content-type"))
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    # Raw audio bytes encoded as a bare base64 SSE payload.
                    yield {"audio_chunk": data, "done": False}
                    continue

                audio_b64 = event.get("audio") or event.get("audio_chunk")
                if audio_b64 is None and "audio_bytes" in event:
                    audio_b64 = base64.b64encode(bytes(event["audio_bytes"])).decode("ascii")
                yield {"audio_chunk": audio_b64, "done": bool(event.get("done", False))}
    except requests.exceptions.Timeout as exc:
        yield {"error": f"engine timeout during streaming: {exc}"}
    except requests.exceptions.ConnectionError as exc:
        yield {"error": f"engine connection error: {exc}"}
    except ValidationError as exc:
        yield {"error": str(exc)}
    finally:
        _cleanup_temp_paths(temp_paths)


def _unary_audio_response(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = requests.post(SPEECH_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return {"error": f"engine request timed out after {REQUEST_TIMEOUT_SECONDS}s"}
    except requests.exceptions.ConnectionError as exc:
        return {"error": f"engine connection error: {exc}"}

    try:
        validate_engine_response(resp.status_code, resp.headers.get("content-type"))
    except ValidationError as exc:
        return {"error": str(exc)}

    if resp.status_code == 503:
        return {"error": "engine reported VRAM out-of-memory or is still warming up"}

    audio_b64 = base64.b64encode(resp.content).decode("ascii")
    return {
        "audio_base64": audio_b64,
        "response_format": payload["response_format"],
        "sample_rate": 24000,
    }


def handler(job: dict[str, Any]):
    """RunPod serverless entrypoint. See `runpod.serverless.start`."""
    raw_input = job.get("input", {})

    try:
        job_input = validate_job_input(raw_input)
    except ValidationError as exc:
        log.warning("Invalid job input: %s", exc)
        return {"error": f"invalid input: {exc}"}

    try:
        engine_references, temp_paths = _materialize_references(job_input.get("references", []))
    except Exception as exc:  # noqa: BLE001 - surface as a clean job error, not a worker crash
        log.error("Failed to materialize reference audio: %s", exc)
        return {"error": f"failed to process reference audio: {exc}"}

    payload = _build_engine_payload(job_input, engine_references)

    if job_input["stream"]:
        return _stream_audio_chunks(payload, temp_paths)

    try:
        return _unary_audio_response(payload)
    finally:
        _cleanup_temp_paths(temp_paths)


# Run the cold-start bootstrap at import time — RunPod's platform (setup
# validation and the worker supervisor) needs to observe a live
# runpod-importing process almost immediately, so this cannot wait behind
# a separate shell entrypoint. Importing this module (e.g. from
# tests/test_local.py) triggers the same bootstrap; _ensure_engine_running
# skips launching a duplicate engine if one is already healthy.
_bootstrap()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
