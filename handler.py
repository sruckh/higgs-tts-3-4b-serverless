#!/usr/bin/env python3
"""Stage 04 — RunPod Serverless handler for higgs-tts3-runpod.

Bridges RunPod job input (see `_config/API_SCHEMA.json`) to the local
SGLang-Omni engine's OpenAI-compatible `/v1/audio/speech` endpoint started
by stage 03 (`start_engine.sh`), returning either a unary base64-encoded
audio payload or a generator of SSE audio chunks when `stream=True`.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from collections.abc import Generator
from typing import Any

import requests
import runpod

from schema_validator import ValidationError, validate_engine_response, validate_job_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("handler")

ENGINE_BASE_URL = "http://127.0.0.1:8000"
SPEECH_ENDPOINT = f"{ENGINE_BASE_URL}/v1/audio/speech"

REQUEST_TIMEOUT_SECONDS = 120
STREAM_CHUNK_TIMEOUT_SECONDS = 60

# Reference clips arrive as base64 in the job input (callers upload them;
# they have no access to the RunPod Network Volume). Decode each one to a
# short-lived local file the engine can read, then delete it once the job
# finishes — reference audio never touches persistent storage.
REFERENCE_TMP_DIR = os.environ.get("REFERENCE_TMP_DIR", "/tmp/higgs-refs")


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


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
