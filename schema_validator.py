"""Stage 04 — Request/response validation helpers for the higgs-tts3-runpod
handler. Mirrors _config/API_SCHEMA.json (openai_speech_request /
runpod_job_input).
"""
from __future__ import annotations

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
MAX_INPUT_CHARS = 10_000


class ValidationError(ValueError):
    """Raised when a job input fails schema validation."""


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"'{field}' must be a non-empty string")
    return value


def validate_reference(ref: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(ref, dict):
        raise ValidationError(f"references[{index}] must be an object")
    audio_path = ref.get("audio_path")
    text = ref.get("text")
    if not isinstance(audio_path, str) or not audio_path.strip():
        raise ValidationError(f"references[{index}].audio_path must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"references[{index}].text must be a non-empty string")
    return {"audio_path": audio_path, "text": text}


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
    normalized["references"] = [validate_reference(r, i) for i, r in enumerate(references)]

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


def validate_engine_response(status_code: int, content_type: str | None) -> None:
    """Sanity-check the local SGLang-Omni engine's HTTP response before
    relaying it back to the RunPod caller."""
    if status_code == 200:
        return
    if status_code == 400:
        raise ValidationError("engine rejected request: invalid inline control tags or malformed payload")
    if status_code == 503:
        raise ValidationError("engine unavailable: VRAM out-of-memory or not yet warmed up")
    raise ValidationError(f"engine returned unexpected status {status_code} ({content_type})")
