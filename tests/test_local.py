#!/usr/bin/env python3
"""Local simulation and test suite for the higgs-tts3-runpod handler. Runs
`handler.handler()` directly against jobs shaped like RunPod's
`runpod_job_input`, bypassing the RunPod queue so the SGLang-Omni engine
(already running locally via scripts/start_engine.sh) can be exercised
end-to-end without deploying.

Usage:
    python3 tests/test_local.py [--concurrency-sweep] [--out benchmark_results.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import struct
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handler as handler_module  # noqa: E402

SAMPLE_RATE = 24000


def _sample_reference_audio_base64(duration_seconds: float = 0.5, freq_hz: float = 220.0) -> str:
    """Synthesize a tiny sine-wave WAV clip and return it base64-encoded, the
    same shape a caller would upload as `references[].audio_base64` — no
    file on disk or on the RunPod Network Volume involved."""
    n_samples = int(SAMPLE_RATE * duration_seconds)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        frames = b"".join(
            struct.pack("<h", int(32767 * 0.2 * math.sin(2 * math.pi * freq_hz * i / SAMPLE_RATE)))
            for i in range(n_samples)
        )
        wav_file.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode("ascii")


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "plain_text_synthesis",
        "job": {"input": {"input": "Hello from the higgs TTS 3 test suite.", "voice": "default"}},
    },
    {
        "name": "zero_shot_voice_cloning",
        "job": {
            "input": {
                "input": "This sentence should be spoken in the cloned voice.",
                "references": [
                    {
                        "audio_base64": _sample_reference_audio_base64(),
                        "text": "Reference transcript sample.",
                        "audio_format": "wav",
                    }
                ],
            }
        },
    },
    {
        "name": "streaming_sse",
        "job": {
            "input": {
                "input": "Streaming response test with several audio chunks.",
                "voice": "default",
                "stream": True,
            }
        },
    },
    {
        "name": "inline_control_tokens",
        "job": {
            "input": {
                "input": "<|emotion:elation|>We did it! <|sfx:laughter|>Haha, amazing news.",
                "voice": "default",
            }
        },
    },
]


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    name = case["name"]
    job = case["job"]
    start = time.perf_counter()
    result = handler_module.handler(job)

    if hasattr(result, "__next__"):
        chunks = []
        first_chunk_time = None
        for event in result:
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            if "error" in event:
                return {"name": name, "status": "fail", "error": event["error"]}
            chunks.append(event)
        elapsed = time.perf_counter() - start
        ttfa = (first_chunk_time - start) if first_chunk_time else None
        return {
            "name": name,
            "status": "pass",
            "elapsed_seconds": round(elapsed, 4),
            "time_to_first_audio_seconds": round(ttfa, 4) if ttfa else None,
            "chunk_count": len(chunks),
        }

    elapsed = time.perf_counter() - start
    if isinstance(result, dict) and "error" in result:
        return {"name": name, "status": "fail", "error": result["error"]}

    audio_b64 = result.get("audio_base64", "") if isinstance(result, dict) else ""
    audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
    return {
        "name": name,
        "status": "pass",
        "elapsed_seconds": round(elapsed, 4),
        "audio_bytes": len(audio_bytes),
    }


def run_functional_suite() -> list[dict[str, Any]]:
    results = []
    for case in TEST_CASES:
        print(f"[test_local] running: {case['name']}")
        try:
            outcome = run_case(case)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
            outcome = {"name": case["name"], "status": "fail", "error": str(exc)}
        print(f"[test_local] result: {outcome}")
        results.append(outcome)
    return results


def run_concurrency_benchmark(levels: list[int] | None = None) -> dict[str, Any]:
    """Sweep concurrent requests, measuring wall-clock RTF (real-time factor)
    and per-request latency for a fixed synthesis job."""
    levels = levels if levels is not None else [1, 2, 4, 8]
    job = {"input": {"input": "Benchmark sentence for concurrency sweep.", "voice": "default"}}
    audio_duration_seconds_estimate = 2.5  # approximate spoken duration of the benchmark sentence

    sweep_results = {}
    for concurrency in levels:
        latencies = []
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_timed_unary_call, job) for _ in range(concurrency)]
            for future in as_completed(futures):
                latencies.append(future.result())
        wall_clock = time.perf_counter() - start

        throughput_rps = concurrency / wall_clock if wall_clock > 0 else 0.0
        rtf = wall_clock / (audio_duration_seconds_estimate * concurrency) if wall_clock > 0 else 0.0

        sweep_results[str(concurrency)] = {
            "concurrency": concurrency,
            "wall_clock_seconds": round(wall_clock, 4),
            "mean_latency_seconds": round(statistics.mean(latencies), 4) if latencies else None,
            "p95_latency_seconds": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 4)
            if latencies
            else None,
            "throughput_requests_per_second": round(throughput_rps, 4),
            "real_time_factor": round(rtf, 4),
        }
        print(f"[test_local] concurrency={concurrency} -> {sweep_results[str(concurrency)]}")
    return sweep_results


def _timed_unary_call(job: dict[str, Any]) -> float:
    start = time.perf_counter()
    handler_module.handler(job)
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency-sweep", action="store_true", help="also run the concurrency benchmark")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("benchmark_results.json")),
        help="path to write benchmark_results.json",
    )
    args = parser.parse_args()

    functional_results = run_functional_suite()
    failures = [r for r in functional_results if r["status"] != "pass"]

    report: dict[str, Any] = {
        "sample_rate": SAMPLE_RATE,
        "functional_tests": functional_results,
    }

    if args.concurrency_sweep:
        report["concurrency_benchmark"] = run_concurrency_benchmark()

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[test_local] wrote {args.out}")

    if failures:
        print(f"[test_local] {len(failures)} test(s) failed", file=sys.stderr)
        return 1
    print("[test_local] all functional tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
