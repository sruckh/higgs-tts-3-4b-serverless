<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="higgs-tts3-runpod — RunPod Serverless worker that deploys bosonai/higgs-tts-3-4b behind an OpenAI-compatible /v1/audio/speech endpoint">
</p>

<p align="center">
  <a href="https://huggingface.co/bosonai/higgs-tts-3-4b"><img alt="Model: bosonai/higgs-tts-3-4b" src="https://img.shields.io/badge/model-bosonai%2Fhiggs--tts--3--4b-38E1A6?style=flat-square&labelColor=0A0D10"></a>
  <a href="https://github.com/sgl-project/sglang-omni"><img alt="Engine: SGLang-Omni" src="https://img.shields.io/badge/engine-SGLang--Omni-38E1A6?style=flat-square&labelColor=0A0D10"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-8A939C?style=flat-square&labelColor=0A0D10">
  <img alt="CUDA 12.4" src="https://img.shields.io/badge/CUDA-12.4-8A939C?style=flat-square&labelColor=0A0D10">
</p>

A [RunPod Serverless](https://www.runpod.io/serverless-gpu) worker that serves **Higgs TTS 3 (4B)** — a 4B-parameter autoregressive text-to-speech model — behind an OpenAI-compatible `/v1/audio/speech` endpoint. [SGLang-Omni](https://github.com/sgl-project/sglang-omni) runs the inference engine inside the worker container; a thin Python `handler.py` bridges RunPod's job queue to it and returns synthesized audio, unary or streamed.

## What it does

- **Zero-shot voice cloning** — upload a reference audio clip (base64) + transcript inline, no fine-tuning and no pre-staged files required.
- **Inline delivery control** — emotion, style, prosody, and sound-effect tags mid-utterance (`<|emotion:elation|>`, `<|sfx:laughter|>Haha`).
- **Streaming or unary audio** — Server-Sent Events for low time-to-first-audio, or a single base64 payload.
- **Cold-start-aware caching** — model weights persist on a RunPod Network Volume so restarts don't re-download ~8GB of weights.

## Quickstart

```bash
# 1. Build and push the worker image
docker build -t registry/higgs-tts3-runpod:latest .
docker push registry/higgs-tts3-runpod:latest

# 2. Create the RunPod Serverless endpoint
#    deploy/runpod_template.json has the GPU, Network Volume, and env var
#    defaults — import it via the RunPod console or `runpodctl`.

# 3. Call it
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "input": "Hello from Higgs TTS 3, running on RunPod.",
          "voice": "default",
          "response_format": "wav"
        }
      }'
```

A gated `bosonai/higgs-tts-3-4b` requires `HF_TOKEN` set on the endpoint; see [Configuration](#configuration).

## How it works

<p align="center">
  <img src="./assets/readme/architecture.svg" width="100%" alt="Container lifecycle: entrypoint.sh runs a one-time cold start, then handler.py serves every job against the local SGLang-Omni engine">
</p>

Each worker runs a **one-time cold start** (`entrypoint.sh`): resolve the Hugging Face cache path, verify CUDA/GPU/audio libraries, fetch and verify model weights, then launch `sgl-omni serve` in the background and poll `/health` until it's ready. Once healthy, `handler.py` takes over in the foreground and, for **every job**, validates the RunPod payload, forwards it to the local engine over HTTP, and relays the audio response back through RunPod — as a single base64 payload or an SSE-style generator of chunks.

The engine only ever talks to `127.0.0.1:8000`; RunPod's job queue is the only external surface.

## API

`handler.py` accepts the same fields as OpenAI's `/v1/audio/speech`, wrapped in RunPod's job envelope:

```json
{
  "input": {
    "input": "Text to synthesize.",
    "voice": "default",
    "references": [
      { "audio_base64": "<base64-encoded audio bytes>", "text": "Reference transcript.", "audio_format": "wav" }
    ],
    "response_format": "wav",
    "speed": 1.0,
    "temperature": 0.8,
    "top_k": 50,
    "stream": false
  }
}
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `input` | string | — | required, ≤10,000 characters |
| `voice` | string | — | either `voice` or `references` is required |
| `references` | array | `[]` | up to 4 `{audio_base64, text, audio_format}` clips for zero-shot cloning |
| `response_format` | string | `"wav"` | one of `wav`, `mp3`, `opus`, `pcm` |
| `speed` | number | `1.0` | |
| `temperature` | number | `0.8` | |
| `top_k` | integer | `50` | |
| `stream` | boolean | `false` | SSE chunk stream instead of one payload |

Unary responses return `{"audio_base64": "...", "response_format": "wav", "sample_rate": 24000}`. Streaming responses yield `{"audio_chunk": "...", "done": false}` events, ending with `done: true`; both modes surface engine failures as `{"error": "..."}` (invalid input, VRAM exhaustion, or engine timeout).

### Voice cloning with an uploaded reference clip

`references[].audio_base64` is the raw audio itself, base64-encoded — callers upload it inline; the worker never reads it from the RunPod Network Volume (that volume only holds the cached model weights). Encode a local clip and inline it into the job payload:

```bash
REF_B64=$(base64 -w0 my_voice_sample.wav)

curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "input": "This will be spoken in the cloned voice.",
          "references": [
            { "audio_base64": "'"$REF_B64"'", "text": "Transcript of my_voice_sample.wav", "audio_format": "wav" }
          ]
        }
      }'
```

`handler.py` decodes each reference to a short-lived local temp file for the engine call and deletes it once the job finishes — reference clips are capped at 25MB decoded and 4 per request.

### Inline control tags

Delivery is controllable mid-sentence with `<|category:value|>` tags — emotion and style at the start of the utterance, sound effects and pauses inline:

```text
<|emotion:elation|>We shipped it! <|sfx:laughter|>Haha, finally.
<|style:whispering|>Keep it down<|prosody:pause|>they're still listening.
```

Supported categories: `emotion` (21 values, e.g. `elation`, `determination`, `sadness`), `style` (`singing`, `shouting`, `whispering`), `prosody` (speed/pitch/expressiveness + `pause` / `long_pause`), and `sfx` (`laughter`, `cough`, `sigh`, `sneeze`, and more, each paired with matching onomatopoeia).

## Performance

Reference concurrency sweep on an A100 80GB (`tests/benchmark_results.json`; regenerate against a live engine with `python3 tests/test_local.py --concurrency-sweep`):

| Concurrency | Throughput (req/s) | Mean latency | Real-time factor |
| ---: | ---: | ---: | ---: |
| 1 | 1.19 | 0.84s | 0.336 |
| 2 | 1.90 | 0.98s | 0.210 |
| 4 | 2.47 | 1.41s | 0.162 |
| 8 | 2.72 | 2.51s | 0.147 |

Lower real-time factor is better (< 1.0 means faster than real-time playback).

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_REPO_ID` | `bosonai/higgs-tts-3-4b` | Hugging Face repo to serve |
| `MODEL_REVISION` | `main` | Model revision/branch |
| `HF_TOKEN` | — | Required if the model repo is gated |
| `HF_HOME` | `/runpod-volume/huggingface-cache` | Weight cache path; point it at the mounted Network Volume |
| `BAKE_INTO_IMAGE` | `0` | Set to `1` to cache weights inside the image instead of a Network Volume |
| `SKIP_MODEL_DOWNLOAD` | `0` | Set to `1` if weights are already present at `HF_HOME` |
| `TP_SIZE` | `1` | Tensor-parallel degree passed to `sgl-omni serve` |

Hardware: NVIDIA GPU with **≥32GB VRAM** (A100 80GB, H100 80GB, or L40S 48GB), CUDA 12.4, Python 3.12.

## Project layout

```text
.
├── Dockerfile              # final worker image (base env + engine + handler)
├── entrypoint.sh           # cold start orchestration -> exec handler.py
├── handler.py               # RunPod serverless handler
├── schema_validator.py      # request/response validation
├── requirements.txt
├── scripts/
│   ├── base_env_check.sh    # CUDA / GPU / audio-lib sanity check
│   ├── cache_config.sh      # resolves HF_HOME (network volume vs. baked)
│   ├── download_model.py    # snapshot_download + integrity verification
│   └── start_engine.sh      # launches sgl-omni serve, polls /health
├── config/
│   └── engine_config.json   # engine runtime parameters
├── deploy/
│   └── runpod_template.json # RunPod Serverless endpoint template
└── tests/
    ├── test_local.py        # local handler simulation + concurrency sweep
    └── benchmark_results.json
```

## Local testing

`tests/test_local.py` calls `handler.handler()` directly — no RunPod queue needed — but it still expects a live engine on `127.0.0.1:8000`, so run it on a GPU host with the engine already started:

```bash
source scripts/cache_config.sh
scripts/start_engine.sh --background   # waits for /health before returning

python3 tests/test_local.py --concurrency-sweep
```

This exercises plain synthesis, zero-shot cloning, SSE streaming, and inline control tags, then writes fresh numbers to `tests/benchmark_results.json`.

## Limitations

- Requires a CUDA GPU with ≥32GB VRAM — there is no CPU fallback.
- Each worker runs exactly one local engine instance; scaling happens across workers, not within one.
- `bosonai/higgs-tts-3-4b` may be gated on Hugging Face — set `HF_TOKEN` before first deploy.
- Cold start time depends on whether the Network Volume already has cached weights; the first worker to boot on a fresh volume pays the full download.
- Reference clips travel inline as base64 in the request body — keep them short (a few seconds); large clips inflate payload size and job latency.

## License

No `LICENSE` file is included yet — add one before distributing this worker publicly.
