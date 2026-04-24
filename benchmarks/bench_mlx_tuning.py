#!/usr/bin/env python3
"""Benchmark MLX tuning iterations for Higgs.

This harness is designed to capture the tradeoffs that matter on Apple Silicon:
1. TTFT across short/medium/long prompts
2. Decode throughput
3. Long-context retrieval accuracy
4. Structured-output correctness
5. Prefix-cache speedup on multi-turn conversations

It runs five optimization iterations and produces both raw metrics and a
composite score so the best profile is obvious.

Usage:
    python3 benchmarks/bench_mlx_tuning.py <model_path>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

HIGGS_BIN = os.environ.get("HIGGS_BIN", "./target/release/higgs")
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"
MAX_TOKENS = 96
CACHE_SPEEDUP_CAP = 32.0

SHORT_PROMPT = "What is 17 + 25? Reply with digits only."
MEDIUM_PROMPT = (
    "Explain how KV cache reuse affects time-to-first-token and decode throughput "
    "for autoregressive transformer inference on Apple Silicon. Keep the answer "
    "technical and concise."
)
LONG_PROMPT = (
    "Write a technical note about optimizing LLM inference on unified-memory Apple "
    "Silicon systems. Cover TTFT, decode throughput, prompt length sensitivity, "
    "prefix cache reuse, chunked prefill, speculative decode, and quantized KV "
    "caches. Include concrete engineering tradeoffs and failure modes.\n\n"
    + "\n".join(
        f"Section {idx}: Repeated background detail about scheduler fairness, "
        f"prompt staging, and kernel launch overhead on MLX devices."
        for idx in range(1, 91)
    )
)

LONG_CONTEXT_FILLER = "\n".join(
    f"Paragraph {idx}: Cape Town cluster notes about unified memory pressure, "
    f"prefill staging, and context reuse across serving workloads."
    for idx in range(1, 181)
)
LONG_CONTEXT_NEEDLE = "CAPE-TOWN-7419"

PREFIX_CACHE_DOC = "\n".join(
    f"Policy {idx}: Route latency data through the prefill pipeline, keep the "
    f"region failover target as cape town, and retain prefix blocks for reuse."
    for idx in range(1, 161)
)

QA_CASES = [
    {
        "prompt": "Reply with digits only. What is 37 * 19?",
        "expected": "703",
    },
    {
        "prompt": "Reply with one lowercase word only. Which word appears twice in 'alpha beta gamma beta delta'?",
        "expected": "beta",
    },
    {
        "prompt": "Reply with digits only. How many vowels are in the word instrumentation?",
        "expected": "6",
    },
    {
        "prompt": "Reply with lowercase letters only. Reverse the string stressed.",
        "expected": "desserts",
    },
    {
        "prompt": "Reply with comma-separated digits only. Sort 7,1,9,1 ascending.",
        "expected": "1,1,7,9",
    },
]

STRUCTURED_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mlx_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "model_family": {"type": "string"},
                "iteration": {"type": "integer"},
                "prefill_focus": {"type": "boolean"},
                "kv_bits": {"type": "integer"},
                "primary_goal": {"type": "string"},
            },
            "required": [
                "model_family",
                "iteration",
                "prefill_focus",
                "kv_bits",
                "primary_goal",
            ],
            "additionalProperties": False,
        },
    },
}


@dataclass
class Iteration:
    slug: str
    label: str
    env: dict[str, str]
    args: list[str]
    notes: str


ITERATIONS = [
    Iteration(
        slug="baseline",
        label="1. Baseline",
        env={"HIGGS_MLX_PROFILE": "baseline"},
        args=[],
        notes="Current conservative defaults",
    ),
    Iteration(
        slug="latency",
        label="2. Latency Profile",
        env={"HIGGS_MLX_PROFILE": "latency"},
        args=[],
        notes="Favor single-pass prefill and speculative decode",
    ),
    Iteration(
        slug="balanced",
        label="3. Balanced Profile",
        env={"HIGGS_MLX_PROFILE": "balanced"},
        args=[],
        notes="Model-aware chunking plus larger paged KV budget",
    ),
    Iteration(
        slug="throughput",
        label="4. Throughput Profile",
        env={"HIGGS_MLX_PROFILE": "throughput"},
        args=[],
        notes="Bigger decode-oriented chunks and paged KV budget",
    ),
    Iteration(
        slug="throughput_turboquant",
        label="5. Throughput + Safe TurboQuant",
        env={"HIGGS_MLX_PROFILE": "throughput"},
        args=[
            "--kv-cache",
            "turboquant",
            "--kv-bits",
            "3",
            "--kv-key-bits",
            "2",
            "--kv-value-bits",
            "3",
            "--kv-adaptive-dense-layers",
            "8",
        ],
        notes="Adds quality-preserving KV quantization after MLX runtime tuning",
    ),
]


server_proc: subprocess.Popen[bytes] | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


def api_request(
    endpoint: str,
    body: dict[str, Any],
    timeout: int = 300,
) -> tuple[dict[str, Any], float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload, time.perf_counter() - started


def stream_chat(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = MAX_TOKENS,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.0,
    timeout: int = 600,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if response_format is not None:
        body["response_format"] = response_format

    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    first_token_time = None
    output_chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choice = obj.get("choices", [{}])[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            if content and first_token_time is None:
                first_token_time = time.perf_counter()
            if content:
                output_chunks.append(content)
            usage = obj.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

    ended = time.perf_counter()
    if first_token_time is None:
        first_token_time = ended

    if completion_tokens == 0:
        output_chars = sum(len(chunk) for chunk in output_chunks)
        completion_tokens = max(1, int(output_chars / 3.5)) if output_chars else 0

    ttft_s = first_token_time - started
    decode_s = max(ended - first_token_time, 0.001)
    decode_tps = max(completion_tokens - 1, 0) / decode_s

    return {
        "ttft_ms": ttft_s * 1000,
        "decode_tps": decode_tps,
        "total_ms": (ended - started) * 1000,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output": "".join(output_chunks),
        "rss_mb": get_rss_mb(),
    }


def chat(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = MAX_TOKENS,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.0,
    timeout: int = 300,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        body["response_format"] = response_format
    payload, elapsed = api_request("chat/completions", body, timeout=timeout)
    choice = payload.get("choices", [{}])[0]
    usage = payload.get("usage", {})
    return {
        "output": choice.get("message", {}).get("content", ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_ms": elapsed * 1000,
        "rss_mb": get_rss_mb(),
    }


def start_server(model_path: str, iteration: Iteration) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        **iteration.env,
        "HIGGS_ENABLE_THINKING": "0",
        "HIGGS_NO_CONFIG": "1",
    }
    cmd = [
        HIGGS_BIN,
        "serve",
        "--model",
        model_path,
        "--port",
        str(PORT),
        *iteration.args,
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{BASE}/v1/models", timeout=2)
            return proc
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                raise RuntimeError("server exited before becoming ready")
    proc.kill()
    proc.wait(timeout=5)
    raise RuntimeError("server failed to start within 120 seconds")


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(2)


def get_model_id() -> str:
    req = urllib.request.Request(f"{BASE}/v1/models")
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read())
    return payload["data"][0]["id"]


def get_rss_mb() -> float:
    global server_proc
    if server_proc is None:
        return 0.0
    try:
        pgid = os.getpgid(server_proc.pid)
        out = subprocess.check_output(["ps", "-o", "rss=", "-g", str(pgid)], text=True).strip()
        total_kb = sum(int(line.strip()) for line in out.splitlines() if line.strip())
        return total_kb / 1024
    except (subprocess.CalledProcessError, ProcessLookupError, ValueError):
        return 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def clamp_cache_speedup(speedup: float, cap: float = CACHE_SPEEDUP_CAP) -> float:
    return min(speedup, cap)


def compute_accuracy_score(result: dict[str, Any]) -> float:
    qa_acc = result["qa"]["accuracy"]
    long_acc = result["long_context"]["accuracy"]
    structured_acc = result["structured_output"]["accuracy"]
    cache_acc = result["prefix_cache"]["accuracy"]
    return (qa_acc * 0.45) + (long_acc * 0.25) + (structured_acc * 0.15) + (cache_acc * 0.15)


def compute_speed_score(result: dict[str, Any], best_ttft: float, best_decode: float) -> float:
    ttft_score = best_ttft / result["prompt_sweep"]["weighted_ttft_ms"] if best_ttft else 0.0
    decode_score = (
        result["prompt_sweep"]["weighted_decode_tps"] / best_decode if best_decode else 0.0
    )
    return (ttft_score * 0.55) + (decode_score * 0.45)


def compute_iteration_score(result: dict[str, Any], best_ttft: float, best_decode: float, best_cache: float) -> dict[str, float]:
    accuracy = compute_accuracy_score(result)
    speed = compute_speed_score(result, best_ttft, best_decode)
    cache_speedup = (
        clamp_cache_speedup(result["prefix_cache"]["speedup"])
        if result["prefix_cache"]["passed"]
        else 0.0
    )
    cache_score = cache_speedup / best_cache if best_cache else 0.0
    return {
        "accuracy": accuracy,
        "speed": speed,
        "cache": cache_score,
        "composite": 100.0 * ((accuracy * 0.45) + (speed * 0.45) + (cache_score * 0.10)),
    }


def rank_results_by_score(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda r: r["score"]["composite"], reverse=True)


def prompt_sweep(model: str, repeats: int) -> dict[str, Any]:
    prompts = {
        "short": SHORT_PROMPT,
        "medium": MEDIUM_PROMPT,
        "long": LONG_PROMPT,
    }
    weights = {"short": 0.2, "medium": 0.3, "long": 0.5}
    metrics: dict[str, Any] = {}

    for label, prompt in prompts.items():
        runs = []
        for attempt in range(repeats):
            if attempt == 0:
                _ = stream_chat(
                    model,
                    [{"role": "user", "content": f"[warmup {label}] {prompt}"}],
                    max_tokens=8,
                )
            result = stream_chat(model, [{"role": "user", "content": prompt}], max_tokens=64)
            runs.append(result)
        metrics[label] = {
            "ttft_ms": median([run["ttft_ms"] for run in runs]),
            "decode_tps": median([run["decode_tps"] for run in runs]),
            "prompt_tokens": median([run["prompt_tokens"] for run in runs]),
            "completion_tokens": median([run["completion_tokens"] for run in runs]),
        }

    weighted_ttft = sum(metrics[label]["ttft_ms"] * weights[label] for label in prompts)
    weighted_decode = sum(metrics[label]["decode_tps"] * weights[label] for label in prompts)

    return {
        "by_prompt": metrics,
        "weighted_ttft_ms": weighted_ttft,
        "weighted_decode_tps": weighted_decode,
    }


def run_qa_suite(model: str) -> dict[str, Any]:
    results = []
    passed = 0
    for case in QA_CASES:
        response = chat(
            model,
            [{"role": "user", "content": case["prompt"]}],
            max_tokens=16,
        )
        output = normalize_text(response["output"])
        success = output == case["expected"]
        passed += int(success)
        results.append(
            {
                "prompt": case["prompt"],
                "expected": case["expected"],
                "output": output,
                "passed": success,
            }
        )
    return {
        "passed": passed,
        "total": len(QA_CASES),
        "accuracy": passed / len(QA_CASES),
        "results": results,
    }


def run_long_context_needle(model: str) -> dict[str, Any]:
    prompt = (
        "Read the following deployment notes carefully.\n\n"
        f"{LONG_CONTEXT_FILLER}\n\n"
        f"Important hidden code: {LONG_CONTEXT_NEEDLE}\n\n"
        f"{LONG_CONTEXT_FILLER}\n\n"
        "Question: reply with the deployment code only."
    )
    response = chat(model, [{"role": "user", "content": prompt}], max_tokens=8)
    output = normalize_text(response["output"]).replace(" ", "")
    expected = LONG_CONTEXT_NEEDLE.lower()
    return {
        "expected": expected,
        "output": output,
        "passed": expected in output,
        "accuracy": 1.0 if expected in output else 0.0,
        "prompt_tokens": response["prompt_tokens"],
    }


def run_structured_output(model: str, iteration_index: int) -> dict[str, Any]:
    prompt = (
        "Return structured JSON only. Facts: model_family=qwen, iteration="
        f"{iteration_index}, prefill_focus=true, kv_bits=3, primary_goal=latency."
    )
    response = chat(
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=64,
        response_format=STRUCTURED_OUTPUT_SCHEMA,
    )
    try:
        parsed = json.loads(response["output"])
    except json.JSONDecodeError:
        return {"passed": False, "accuracy": 0.0, "raw": response["output"]}

    passed = parsed == {
        "model_family": "qwen",
        "iteration": iteration_index,
        "prefill_focus": True,
        "kv_bits": 3,
        "primary_goal": "latency",
    }
    return {"passed": passed, "accuracy": 1.0 if passed else 0.0, "json": parsed}


def run_prefix_cache_suite(model: str) -> dict[str, Any]:
    prefix_messages = [
        {
            "role": "system",
            "content": (
                "You are reviewing an operations handbook. Study the material and "
                "reply READY only.\n\n"
                + PREFIX_CACHE_DOC
            ),
        },
        {"role": "user", "content": "Read the handbook and reply READY only."},
    ]
    followup_messages = [
        *prefix_messages,
        {"role": "assistant", "content": "READY"},
        {
            "role": "user",
            "content": "What is the failover region? Reply with two words only.",
        },
    ]

    cold = stream_chat(model, followup_messages, max_tokens=8)
    _ = stream_chat(model, prefix_messages, max_tokens=4)
    warm = stream_chat(model, followup_messages, max_tokens=8)

    warm_output = normalize_text(warm["output"])
    passed = "cape town" in warm_output
    speedup = cold["ttft_ms"] / warm["ttft_ms"] if warm["ttft_ms"] > 0 else 0.0
    return {
        "cold_ttft_ms": cold["ttft_ms"],
        "warm_ttft_ms": warm["ttft_ms"],
        "speedup": speedup,
        "answer": warm["output"],
        "passed": passed,
        "accuracy": 1.0 if passed else 0.0,
    }


def benchmark_iteration(
    model_path: str,
    iteration_index: int,
    iteration: Iteration,
    repeats: int,
) -> dict[str, Any]:
    global server_proc

    log(f"\n{'=' * 80}")
    log(f"{iteration.label}")
    log(f"Notes: {iteration.notes}")
    if iteration.args:
        log(f"Args: {' '.join(iteration.args)}")
    log(f"{'=' * 80}")

    server_proc = start_server(model_path, iteration)
    try:
        model = get_model_id()
        log(f"Model: {model}")
        log("Warmup...")
        _ = stream_chat(model, [{"role": "user", "content": "Say ready."}], max_tokens=4)
        log(f"RSS after warmup: {get_rss_mb():.0f} MB")

        sweep = prompt_sweep(model, repeats)
        qa = run_qa_suite(model)
        long_ctx = run_long_context_needle(model)
        structured = run_structured_output(model, iteration_index)
        prefix_cache = run_prefix_cache_suite(model)

        log(
            "Weighted prompt metrics: "
            f"TTFT={sweep['weighted_ttft_ms']:.0f} ms  "
            f"decode={sweep['weighted_decode_tps']:.1f} tok/s"
        )
        log(
            "Accuracy checks: "
            f"qa={qa['passed']}/{qa['total']}  "
            f"needle={'pass' if long_ctx['passed'] else 'fail'}  "
            f"json={'pass' if structured['passed'] else 'fail'}  "
            f"cache={'pass' if prefix_cache['passed'] else 'fail'}"
        )
        log(
            "Prefix cache: "
            f"cold={prefix_cache['cold_ttft_ms']:.0f} ms  "
            f"warm={prefix_cache['warm_ttft_ms']:.0f} ms  "
            f"speedup={prefix_cache['speedup']:.2f}x"
        )

        return {
            "iteration": iteration.slug,
            "label": iteration.label,
            "notes": iteration.notes,
            "args": iteration.args,
            "prompt_sweep": sweep,
            "qa": qa,
            "long_context": long_ctx,
            "structured_output": structured,
            "prefix_cache": prefix_cache,
            "rss_mb": get_rss_mb(),
        }
    finally:
        stop_server(server_proc)
        server_proc = None


def score_results(results: list[dict[str, Any]]) -> None:
    best_ttft = min(r["prompt_sweep"]["weighted_ttft_ms"] for r in results)
    best_decode = max(r["prompt_sweep"]["weighted_decode_tps"] for r in results)
    best_cache = max(
        (
            min(r["prefix_cache"]["speedup"], CACHE_SPEEDUP_CAP)
            for r in results
            if r["prefix_cache"]["passed"]
        ),
        default=1.0,
    )

    for result in results:
        result["score"] = compute_iteration_score(result, best_ttft, best_decode, best_cache)


def print_summary(results: list[dict[str, Any]]) -> None:
    ordered = rank_results_by_score(results)
    log(f"\n{'#' * 80}")
    log("FINAL SUMMARY")
    log(f"{'#' * 80}")
    log(
        f"{'Iteration':32s} {'Score':>8s} {'TTFT':>10s} {'Decode':>10s} "
        f"{'QA':>6s} {'Needle':>8s} {'JSON':>6s} {'Cache':>8s}"
    )
    log("-" * 96)
    for result in ordered:
        log(
            f"{result['label'][:32]:32s} "
            f"{result['score']['composite']:>7.1f} "
            f"{result['prompt_sweep']['weighted_ttft_ms']:>9.0f} "
            f"{result['prompt_sweep']['weighted_decode_tps']:>9.1f} "
            f"{result['qa']['passed']:>2d}/{result['qa']['total']:<3d} "
            f"{'pass' if result['long_context']['passed'] else 'fail':>8s} "
            f"{'pass' if result['structured_output']['passed'] else 'fail':>6s} "
            f"{result['prefix_cache']['speedup']:>7.2f}x"
        )

    winner = ordered[0]
    log("")
    log(
        f"Winner: {winner['label']} "
        f"(score={winner['score']['composite']:.1f}, "
        f"TTFT={winner['prompt_sweep']['weighted_ttft_ms']:.0f} ms, "
        f"decode={winner['prompt_sweep']['weighted_decode_tps']:.1f} tok/s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Higgs MLX tuning profiles")
    parser.add_argument("model_path", help="Local model path or resolved HF cache directory")
    parser.add_argument("--repeats", type=int, default=2, help="Prompt-sweep repeats per iteration")
    parser.add_argument(
        "--output-json",
        default=f"bench_mlx_tuning_{time.strftime('%Y%m%d_%H%M%S')}.json",
        help="Path to write raw benchmark results",
    )
    args = parser.parse_args()

    if not os.path.isfile(HIGGS_BIN):
        raise SystemExit(f"Higgs binary not found: {HIGGS_BIN}")

    log("=" * 80)
    log(f"MLX TUNING BENCHMARK — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Binary: {HIGGS_BIN}")
    log(f"Model: {args.model_path}")
    log(f"Repeats: {args.repeats}")
    log("=" * 80)

    results = []
    for idx, iteration in enumerate(ITERATIONS, start=1):
        results.append(benchmark_iteration(args.model_path, idx, iteration, args.repeats))

    score_results(results)
    print_summary(results)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"model_path": args.model_path, "results": results}, f, indent=2)
    log(f"\nRaw results written to {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_server(server_proc)
        sys.exit(130)
