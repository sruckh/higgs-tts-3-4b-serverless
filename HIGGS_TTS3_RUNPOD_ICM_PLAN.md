# Higgs TTS 3.4B RunPod Serverless ICM Workspace Specification

## 1. Executive Summary & Higgs TTS 3 Technical Profile

### Executive Summary
This document defines the complete architecture, integration design, and stage-by-stage Interpretable Context Methodology (ICM) specification for deploying **Higgs TTS 3.4B** (`bosonai/higgs-tts-3-4b`) as an auto-scaling, low-latency text-to-speech inference worker on **RunPod Serverless**. 

The serving architecture leverages **SGLang-Omni** (`sgl-omni serve`) as the primary high-throughput inference backbone (with **vLLM-Omni** as a secondary option), wrapped inside a lightweight Python RunPod Serverless handler (`runpod.serverless.start`). Model weights and HuggingFace assets are cached on high-speed RunPod Network Volumes (`/runpod-volume/huggingface-cache`) to eliminate cold-start download penalties.

The entire development and deployment workflow is governed by a 5-layer **Interpretable Context Methodology (ICM)** workspace, enabling predictable, single-agent execution from environment bootstrapping to production health verification.

---

### Higgs TTS 3.4B Model Profile & Architecture

| Parameter / Feature | Specification |
| :--- | :--- |
| **Model Name** | `bosonai/higgs-tts-3-4b` |
| **Backbone Architecture** | ~4B parameter autoregressive decoder (36 layers, hidden_size=2560, GQA 32/8) |
| **Audio Tokenization** | Higgs Tokenizer: 8 codebooks × 1026 vocabulary size, delay pattern alignment |
| **Audio Format & Rate** | 24 kHz sample rate, 25 frames per second (40 ms / frame) |
| **Context Length** | 8,192 tokens max sequence length |
| **Primary Serving Engine** | SGLang-Omni (`docker pull lmsysorg/sglang-omni:dev`, command `sgl-omni serve`) |
| **Alternative Engine** | vLLM-Omni (`vllm-omni serve --omni --trust-remote-code`) |
| **Target Hardware** | NVIDIA CUDA GPUs with ≥32GB VRAM (e.g., A100-80GB, H100-80GB, L40S-48GB) |
| **API Format** | OpenAI-compatible `/v1/audio/speech` REST API supporting SSE streaming |
| **Control Capabilities** | Zero-shot voice cloning, inline emotion/style/prosody/sfx control tokens |

#### Advanced Control Token Capabilities
Higgs TTS 3 supports mid-utterance inline control tokens using `<|category:value|>` tag syntax:
1. **Delivery & Emotion Tags** (placed at start of input):
   - *Emotion*: `<|emotion:elation|>`, `<|emotion:amusement|>`, `<|emotion:enthusiasm|>`, `<|emotion:determination|>`, `<|emotion:pride|>`, `<|emotion:contentment|>`, `<|emotion:affection|>`, `<|emotion:relief|>`, `<|emotion:contemplation|>`, `<|emotion:confusion|>`, `<|emotion:surprise|>`, `<|emotion:awe|>`, `<|emotion:longing|>`, `<|emotion:arousal|>`, `<|emotion:anger|>`, `<|emotion:fear|>`, `<|emotion:disgust|>`, `<|emotion:bitterness|>`, `<|emotion:sadness|>`, `<|emotion:shame|>`, `<|emotion:helplessness|>`.
   - *Style*: `<|style:singing|>`, `<|style:shouting|>`, `<|style:whispering|>`.
   - *Prosody*: `<|prosody:speed_very_slow|>`, `<|prosody:speed_slow|>`, `<|prosody:speed_fast|>`, `<|prosody:speed_very_fast|>`, `<|prosody:pitch_low|>`, `<|prosody:pitch_high|>`, `<|prosody:expressive_high|>`, `<|prosody:expressive_low|>`.
2. **Positional Tags & Sound Effects** (inserted inline with matching onomatopoeia):
   - *Pauses*: `<|prosody:pause|>` (400-700ms), `<|prosody:long_pause|>` (700-1500ms).
   - *Sound Effects*: `<|sfx:cough|>Ahem`, `<|sfx:laughter|>Haha`, `<|sfx:crying|>Boohoo`, `<|sfx:screaming|>Ahh`, `<|sfx:burping|>Burp`, `<|sfx:humming|>Mmm`, `<|sfx:sigh|>Uh`, `<|sfx:sniff|>Sff`, `<|sfx:sneeze|>Achoo`.

---

## 2. RunPod Serverless Integration Blueprint

### Cold-Start & Network Volume Strategy
To achieve minimal time-to-first-audio (TTFA) and avoid downloading ~8GB of model weights during pod startup:
- **RunPod Network Volume Mounting**: Mount a persistent Network Volume at `/runpod-volume/`.
- **HuggingFace Cache Redirect**: Set environment variable `HF_HOME=/runpod-volume/huggingface-cache` and `TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache`.
- **Pre-warming**: Stage `bosonai/higgs-tts-3-4b` on the Network Volume during initialization (`hf download bosonai/higgs-tts-3-4b`).
- **Container Pre-baking**: Option to bake Python dependencies and SGLang-Omni wheels directly into the container image to eliminate runtime `pip install` delays.

### Engine Launch & Handler Architecture
The RunPod serverless handler acts as a proxy bridge between RunPod's job queue and the background SGLang-Omni HTTP daemon:

```
[ RunPod Job Queue ]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ RunPod Worker Container                                │
│                                                        │
│  ┌───────────────────────┐   HTTP POST /v1/audio/speech│
│  │ handler.py            │ ──────────────────────────┐ │
│  │ (runpod.serverless)   │                           │ │
│  └───────────────────────┘                           ▼ │
│                                            ┌─────────┴────────┐
│                                            │ SGLang-Omni      │
│                                            │ Engine           │
│                                            │ (Port 8000)      │
│                                            └──────────────────┘
└────────────────────────────────────────────────────────┘
```

1. **Background Engine Startup**: Container entrypoint initializes `sgl-omni serve --model-path bosonai/higgs-tts-3-4b --port 8000` in a background process and waits for `/health` readiness.
2. **Job Translation**: `handler.py` receives `job["input"]`, validates payload against OpenAI `/v1/audio/speech` JSON schema, and sends internal HTTP request to `http://127.0.0.1:8000/v1/audio/speech`.
3. **Response Handling**:
   - **Unary Mode**: Collects complete audio buffer, converts to base64 or uploads to temporary S3/RunPod storage URL, returns payload JSON.
   - **Streaming Mode**: Uses Server-Sent Events (SSE) streaming generator yielding base64 audio chunks back through RunPod's generator handler (`runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})`).

### Performance Benchmarks (H100 80GB GPU Baseline)
SGLang-Omni benchmark numbers on H100 GPU (bf16 precision, CUDA Graph enabled):

| Concurrency Limit | Throughput (req/s) | Mean Latency (ms) | Real-Time Factor (RTF) | Audio Output (s/s) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.62 req/s | 617 ms | 0.147 | 6.89 s/s |
| 2 | 2.70 req/s | 742 ms | 0.180 | 11.37 s/s |
| 4 | 5.45 req/s | 733 ms | 0.177 | 22.84 s/s |
| 8 | 8.91 req/s | 898 ms | 0.217 | 37.38 s/s |
| 16 | 14.74 req/s | 1079 ms | 0.262 | 61.84 s/s |

---

## 3. ICM Workspace Architecture & 5-Layer Specification

The Interpretable Context Methodology (ICM) workspace organizes all codebase files, configuration contracts, and stage execution parameters into a strict 5-layer folder hierarchy:

```
higgs-tts3-runpod/
├── IDENTITY.md                         # Layer 0: Workspace identity & core principles
├── CLAUDE.md                           # Layer 0: Claude agent operating instructions
├── CONTEXT.md                          # Layer 1: Workspace pipeline routing table
├── _config/                            # Layer 3: Environment variables & global rules
│   ├── env.json                        # Hardware specs & RunPod environment settings
│   └── API_SCHEMA.json                 # OpenAI & RunPod payload contracts
├── references/                         # Layer 3: Static documentation & API guides
│   ├── SGLANG_OMNI_GUIDE.md            # SGLang-Omni serving documentation
│   └── RUNPOD_SERVERLESS_BLUEPRINT.md  # RunPod handler & deployment contract
├── stages/                             # Layer 2: Numbered execution stages
│   ├── 01-environment-and-base-image/
│   │   ├── CONTEXT.md                  # Stage 01 contract
│   │   └── output/                     # Layer 4: Dockerfile & base image specs
│   ├── 02-model-download-and-caching/
│   │   ├── CONTEXT.md                  # Stage 02 contract
│   │   └── output/                     # Layer 4: HuggingFace cache scripts
│   ├── 03-sglang-omni-engine-setup/
│   │   ├── CONTEXT.md                  # Stage 03 contract
│   │   └── output/                     # Layer 4: Engine daemon launch script
│   ├── 04-runpod-handler-implementation/
│   │   ├── CONTEXT.md                  # Stage 04 contract
│   │   └── output/                     # Layer 4: handler.py implementation
│   ├── 05-local-testing-and-benchmarking/
│   │   ├── CONTEXT.md                  # Stage 05 contract
│   │   └── output/                     # Layer 4: Benchmark results & test logs
│   └── 06-docker-build-and-runpod-deployment/
│       ├── CONTEXT.md                  # Stage 06 contract
│       └── output/                     # Layer 4: Final deployment manifests & verification
└── output/                             # Layer 4: Global aggregated build artifacts
```

### Layer Rules & Boundaries
- **Layer 0 (`IDENTITY.md`, `CLAUDE.md`)**: Immutable principles and system identity.
- **Layer 1 (`CONTEXT.md`)**: Central routing table mapping stages 01 through 06. Must be kept in sync via `icm sync`.
- **Layer 2 (`stages/NN-*/CONTEXT.md`)**: Stage contracts defining Inputs, Process, and Outputs. Max 80 lines per contract file.
- **Layer 3 (`_config/`, `references/`)**: Shared static knowledge, API schemas, and environment configs.
- **Layer 4 (`output/`)**: Ephemeral runtime artifacts generated by scripts or stage execution.

---

## 4. Stage-by-Stage Implementation Instructions & Contracts

Below are the explicit Layer 2 stage contracts for all 6 pipeline stages.

---

### Stage 01: Environment & Base Image Preparation (`stages/01-environment-and-base-image/CONTEXT.md`)

```markdown
# Stage 01 Contract: Environment & Base Image Preparation

## Inputs
| Input Source | Description |
| :--- | :--- |
| `_config/env.json` | Hardware requirements (CUDA 12.4, Python 3.12, PyTorch 2.4+) |
| `references/SGLANG_OMNI_GUIDE.md` | SGLang-Omni base container image requirements |

## Process
1. Select `lmsysorg/sglang-omni:dev` or `nvidia/cuda:12.4.1-devel-ubuntu22.04` as parent Docker image.
2. Configure system package dependencies (ffmpeg, libsndfile1, git, git-lfs, curl).
3. Set up Python 3.12 virtual environment and install `uv` package manager.
4. Define standard environment variables (`HF_HOME`, `PYTHONUNBUFFERED=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`).

## Outputs
| Output Target | Description |
| :--- | :--- |
| `output/Dockerfile.base` | Base Dockerfile specification |
| `output/base_env_check.sh` | Validation script for CUDA and driver compatibility |
```

---

### Stage 02: Model Download & Cache Management (`stages/02-model-download-and-caching/CONTEXT.md`)

```markdown
# Stage 02 Contract: Model Download & Cache Management

## Inputs
| Input Source | Description |
| :--- | :--- |
| `_config/env.json` | Model repo (`bosonai/higgs-tts-3-4b`) & HF token settings |
| `stages/01-environment-and-base-image/output/Dockerfile.base` | Base image specification |

## Process
1. Write Python/Bash model pre-downloader using `huggingface_hub.snapshot_download`.
2. Configure persistent Network Volume cache directory paths (`/runpod-volume/huggingface-cache`).
3. Set fallback mechanisms for local container baking vs network volume pre-warming.
4. Verify safetensors integrity, multi-codebook tokenizer files, and config JSON.

## Outputs
| Output Target | Description |
| :--- | :--- |
| `output/download_model.py` | Standalone script for downloading Higgs TTS 3.4B weights |
| `output/cache_config.sh` | Shell script setting HF environment cache overrides |
```

---

### Stage 03: SGLang-Omni Engine Setup (`stages/03-sglang-omni-engine-setup/CONTEXT.md`)

```markdown
# Stage 03 Contract: SGLang-Omni Engine Setup

## Inputs
| Input Source | Description |
| :--- | :--- |
| `stages/02-model-download-and-caching/output/cache_config.sh` | Cache paths & model repo ID |
| `references/SGLANG_OMNI_GUIDE.md` | SGLang-Omni CLI arguments and flags |

## Process
1. Create `start_engine.sh` script to launch `sgl-omni serve`.
2. Configure launch flags: `--model-path bosonai/higgs-tts-3-4b`, `--port 8000`, `--host 127.0.0.1`, `--tp 1`.
3. Implement readiness probe looping on `http://127.0.0.1:8000/health` until HTTP 200 is returned.
4. Verify GPU memory allocation and CUDA graph warming for Higgs TTS backbone.

## Outputs
| Output Target | Description |
| :--- | :--- |
| `output/start_engine.sh` | Engine launcher and readiness check script |
| `output/engine_config.json` | Runtime parameters for SGLang-Omni engine |
```

---

### Stage 04: RunPod Handler Implementation (`stages/04-runpod-handler-implementation/CONTEXT.md`)

```markdown
# Stage 04 Contract: RunPod Handler Implementation

## Inputs
| Input Source | Description |
| :--- | :--- |
| `_config/API_SCHEMA.json` | OpenAI `/v1/audio/speech` & RunPod input schema |
| `stages/03-sglang-omni-engine-setup/output/engine_config.json` | Local engine HTTP endpoint specs |

## Process
1. Implement `handler.py` using `runpod.serverless.start()`.
2. Parse `job["input"]` for `input` (text), `voice` / `references`, `temperature`, `top_k`, `stream`.
3. Construct payload for `http://127.0.0.1:8000/v1/audio/speech`.
4. Implement generator function for Server-Sent Events (SSE) streaming yielding base64 audio chunks.
5. Add error handling for timeout, VRAM out-of-memory, and invalid inline control tags.

## Outputs
| Output Target | Description |
| :--- | :--- |
| `output/handler.py` | Complete RunPod serverless handler module |
| `output/schema_validator.py` | Request/response validation helper module |
```

---

### Stage 05: Local Testing & Benchmarking (`stages/05-local-testing-and-benchmarking/CONTEXT.md`)

```markdown
# Stage 05 Contract: Local Testing & Benchmarking

## Inputs
| Input Source | Description |
| :--- | :--- |
| `stages/04-runpod-handler-implementation/output/handler.py` | Serverless handler module |
| `references/SGLANG_OMNI_GUIDE.md` | Benchmark testing protocol |

## Process
1. Implement `test_local.py` simulating RunPod job execution locally.
2. Execute test suite covering:
   - Plain text synthesis.
   - Zero-shot voice cloning with reference audio + transcript.
   - Streaming SSE responses.
   - Inline control tokens (`<|emotion:elation|>`, `<|sfx:laughter|>Haha`).
3. Run concurrency benchmark (sweep 1, 2, 4, 8 concurrent requests) measuring RTF and latency.

## Outputs
| Output Target | Description |
| :--- | :--- |
| `output/test_local.py` | Local simulation and test suite script |
| `output/benchmark_results.json` | Measured latency, TTFA, and throughput metrics |
```

---

### Stage 06: Docker Build & RunPod Deployment (`stages/06-docker-build-and-runpod-deployment/CONTEXT.md`)

```markdown
# Stage 06 Contract: Docker Build & RunPod Deployment

## Inputs
| Input Source | Description |
| :--- | :--- |
| `stages/01-environment-and-base-image/output/Dockerfile.base` | Base Dockerfile |
| `stages/03-sglang-omni-engine-setup/output/start_engine.sh` | Engine startup script |
| `stages/04-runpod-handler-implementation/output/handler.py` | Production handler script |

## Process
1. Assemble final `Dockerfile` combining base environment, engine scripts, and RunPod handler.
2. Write entrypoint `entrypoint.sh` starting SGLang-Omni engine in background and executing `python -u handler.py`.
3. Provide Docker build and push commands (`docker build -t registry/higgs-tts3-runpod:latest .`).
4. Detail RunPod Serverless template creation (GPU selection: A100/H100/L40S, Network Volume mount: `/runpod-volume`, Container Disk: 20GB, Environment Variables).

## Outputs
| Output Target | Description |
| :--- | :--- |
| `output/Dockerfile` | Final container Dockerfile |
| `output/entrypoint.sh` | Container entrypoint script |
| `output/runpod_template.json` | RunPod Serverless template configuration |
```

---

## 5. Implementation Playbook & Audit Checklist

### Execution Playbook
To execute this workspace specification using the `icm` skill:

1. **Scaffold Workspace**:
   ```bash
   python3 /root/.claude/skills/icm/scripts/new .icm/higgs-tts3-runpod --domain higgs-tts3-runpod
   ```
2. **Add Numbered Stages**:
   ```bash
   python3 /root/.claude/skills/icm/scripts/stage .icm/higgs-tts3-runpod 01-environment-and-base-image
   python3 /root/.claude/skills/icm/scripts/stage .icm/higgs-tts3-runpod 02-model-download-and-caching
   python3 /root/.claude/skills/icm/scripts/stage .icm/higgs-tts3-runpod 03-sglang-omni-engine-setup
   python3 /root/.claude/skills/icm/scripts/stage .icm/higgs-tts3-runpod 04-runpod-handler-implementation
   python3 /root/.claude/skills/icm/scripts/stage .icm/higgs-tts3-runpod 05-local-testing-and-benchmarking
   python3 /root/.claude/skills/icm/scripts/stage .icm/higgs-tts3-runpod 06-docker-build-and-runpod-deployment
   ```
3. **Synchronize Routing Table**:
   ```bash
   python3 /root/.claude/skills/icm/scripts/sync .icm/higgs-tts3-runpod
   ```
4. **Run Stage Contracts**:
   For each stage `01` through `06`, display next contract using:
   ```bash
   python3 /root/.claude/skills/icm/scripts/run .icm/higgs-tts3-runpod --stage <NN>
   ```
5. **Audit Workspace Compliance**:
   Verify adherence to ICM rules (max 80 lines per contract, valid Inputs/Process/Outputs tables):
   ```bash
   python3 /root/.claude/skills/icm/scripts/audit .icm/higgs-tts3-runpod
   ```

### Audit Checklist
- [x] **5-Layer Architecture**: All 5 layers explicitly defined (`IDENTITY.md`, `CONTEXT.md`, `stages/`, `_config/`, `output/`).
- [x] **Stage Contracts**: Every stage contains valid `Inputs`, `Process`, and `Outputs` markdown tables.
- [x] **RunPod Blueprint Alignment**: Incorporates cold-start mitigation via Network Volume (`/runpod-volume/huggingface-cache`), background engine proxy pattern, and SSE streaming support.
- [x] **Higgs TTS 3 Model Profile**: Full specs covered (SGLang-Omni backend, ~4B params, 24kHz/25fps, 8-codebook tokenization, inline tags, zero-shot voice cloning).
