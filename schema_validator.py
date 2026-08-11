"""Stage 04 — Request/response validation helpers for the higgs-tts3-runpod
handler. Mirrors _config/API_SCHEMA.json (openai_speech_request /
runpod_job_input).
"""
from __future__ import annotations

import base64
import binascii
from typing import Any

DEFAULTS = {
    "model": "bosonai/higgs-tts-3-4b",
    "response_format": "wav",
    "speed": 1.0,
    "temperature": 0.8,
    "top_k": 50,
    "stream": False,
}

ALLOWED_RESPONSE_FORMATS = {"wav", "mp3", "opus", "pcm"}
ALLOWED_REFERENCE_AUDIO_FORMATS = {"wav", "mp3", "opus", "flac", "pcm"}
MAX_INPUT_CHARS = 10_000
# Reference clips arrive inline as base64 in the job payload — RunPod caps
# request bodies at 10MB for /run and 20MB for /runsync (verified against
# https://docs.runpod.io/serverless/workers/handler-functions#payload-limits
# 2026-08-09). Base64 inflates decoded bytes by ~4/3, so a 6MB decoded total
# becomes ~8MB encoded, leaving headroom under the stricter /run ceiling.
# 3MB per clip (~62s of 24kHz/16-bit/mono WAV) covers realistic voice-cloning
# references; the combined cap is what actually protects the request size.
MAX_REFERENCE_AUDIO_BYTES = 3 * 1024 * 1024
MAX_TOTAL_REFERENCE_AUDIO_BYTES = 6 * 1024 * 1024
MAX_REFERENCES = 4


class ValidationError(ValueError):
    """Raised when a job input fails schema validation."""


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"'{field}' must be a non-empty string")
    return value


def validate_reference(ref: dict[str, Any], index: int) -> tuple[dict[str, Any], int]:
    """Validate one zero-shot voice-cloning reference. Reference audio is
    uploaded by the caller as inline base64 — never a path on the RunPod
    Network Volume, which the caller has no access to. Returns the
    normalized reference plus its decoded byte size, so the caller can
    enforce a combined cap across all references in the job."""
    if not isinstance(ref, dict):
        raise ValidationError(f"references[{index}] must be an object")

    audio_b64 = ref.get("audio_base64")
    text = ref.get("text")
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        raise ValidationError(
            f"references[{index}].audio_base64 must be a non-empty base64-encoded audio string"
        )
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"references[{index}].text must be a non-empty string")

    try:
        decoded_size = len(base64.b64decode(audio_b64, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValidationError(f"references[{index}].audio_base64 is not valid base64: {exc}") from exc

    if decoded_size == 0:
        raise ValidationError(f"references[{index}].audio_base64 decodes to empty audio")
    if decoded_size > MAX_REFERENCE_AUDIO_BYTES:
        raise ValidationError(
            f"references[{index}].audio_base64 exceeds max size of {MAX_REFERENCE_AUDIO_BYTES} bytes"
        )

    audio_format = ref.get("audio_format", "wav")
    if audio_format not in ALLOWED_REFERENCE_AUDIO_FORMATS:
        raise ValidationError(
            f"references[{index}].audio_format must be one of {sorted(ALLOWED_REFERENCE_AUDIO_FORMATS)}"
        )

    return {"audio_base64": audio_b64, "text": text, "audio_format": audio_format}, decoded_size


def validate_job_input(raw_input: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a RunPod `job["input"]` payload.

    Accepts the OpenAI-style speech request fields described in
    `_config/API_SCHEMA.json` (`openai_speech_request`), applying defaults
    for optional fields and raising `ValidationError` on malformed input.
    """
    if not isinstance(raw_input, dict):
        raise ValidationError("job input must be a JSON object")

    text = _require_str(raw_input.get("input"), "input")
    if len(text) > MAX_INPUT_CHARS:
        raise ValidationError(f"'input' exceeds max length of {MAX_INPUT_CHARS} characters")

    normalized: dict[str, Any] = dict(DEFAULTS)
    normalized["input"] = text
    normalized["model"] = raw_input.get("model", DEFAULTS["model"])
    normalized["voice"] = raw_input.get("voice")

    references = raw_input.get("references", []) or []
    if not isinstance(references, list):
        raise ValidationError("'references' must be a list")
    if len(references) > MAX_REFERENCES:
        raise ValidationError(f"'references' supports at most {MAX_REFERENCES} entries")

    normalized_references: list[dict[str, Any]] = []
    total_reference_bytes = 0
    for i, r in enumerate(references):
        normalized_ref, decoded_size = validate_reference(r, i)
        total_reference_bytes += decoded_size
        if total_reference_bytes > MAX_TOTAL_REFERENCE_AUDIO_BYTES:
            raise ValidationError(
                f"combined 'references' audio exceeds max total of {MAX_TOTAL_REFERENCE_AUDIO_BYTES} bytes"
            )
        normalized_references.append(normalized_ref)
    normalized["references"] = normalized_references

    response_format = raw_input.get("response_format", DEFAULTS["response_format"])
    if response_format not in ALLOWED_RESPONSE_FORMATS:
        raise ValidationError(
            f"'response_format' must be one of {sorted(ALLOWED_RESPONSE_FORMATS)}, got {response_format!r}"
        )
    normalized["response_format"] = response_format

    for field, expected_type in (("speed", (int, float)), ("temperature", (int, float))):
        value = raw_input.get(field, DEFAULTS[field])
        if not isinstance(value, expected_type):
            raise ValidationError(f"'{field}' must be a number")
        normalized[field] = float(value)

    top_k = raw_input.get("top_k", DEFAULTS["top_k"])
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValidationError("'top_k' must be a positive integer")
    normalized["top_k"] = top_k

    stream = raw_input.get("stream", DEFAULTS["stream"])
    if not isinstance(stream, bool):
        raise ValidationError("'stream' must be a boolean")
    normalized["stream"] = stream

    if not normalized["voice"] and not normalized["references"]:
        raise ValidationError("either 'voice' or 'references' must be provided")

    return normalized


def validate_engine_response(status_code: int, content_type: str | None, body: str = "") -> None:
    """Sanity-check the local SGLang-Omni engine's HTTP response before
    relaying it back to the RunPod caller.

    `body` (the engine's own response text, truncated) is included verbatim
    in the raised error rather than replaced with a guessed generic
    message — a previous version hid the engine's actual validation error
    behind a fixed string ("invalid inline control tags or malformed
    payload") for every 400, which made a real payload-shape bug
    undiagnosable from the job's returned error alone (confirmed live
    2026-08-10).
    """
    if status_code == 200:
        return
    detail = f": {body[:2000]}" if body else ""
    if status_code == 400:
        raise ValidationError(f"engine rejected request (400){detail}")
    if status_code == 503:
        raise ValidationError(f"engine unavailable: VRAM out-of-memory or not yet warmed up{detail}")
    raise ValidationError(f"engine returned unexpected status {status_code} ({content_type}){detail}")
