# Benchmarking

This document collects the benchmark methodology and the benchmark-driven defaults referenced from the README.

## Environment

- Benchmarks in the README were run on M4 Max 128GB.
- Temperature was set to `0`.
- Warmup passes were excluded from the reported numbers.

## MLX Tuning Harness

Use the benchmark harness below to compare five serving iterations on the same local model:

```bash
python3 benchmarks/bench_mlx_tuning.py ~/.cache/lm-studio/models/mlx-community/Qwen3.6-35B-A3B-4bit
```

The harness evaluates:

- TTFT across short, medium, and long prompts
- decode throughput
- short QA accuracy
- long-context retrieval accuracy
- structured-output correctness
- prefix-cache speedup on multi-turn conversations

## Iterations

The harness compares five iterations:

1. baseline
2. latency profile
3. balanced profile
4. throughput profile
5. throughput profile plus safe TurboQuant KV settings

## Benchmark-Driven Defaults

Higgs uses benchmark results to make `auto` a model-aware default rather than a static preset.

Current examples that informed the default:

- `mlx-community/Qwen3-1.7B-4bit`: `balanced` won with `91.8` composite, `339 ms` weighted TTFT, `345.7 tok/s` decode, and `20.36x` prefix-cache speedup.
- `mlx-community/Qwen3.6-35B-A3B-4bit`: `throughput` won with `95.7` composite, `842 ms` weighted TTFT, `119.2 tok/s` decode, and `56.19x` prefix-cache speedup.

That is why `auto` resolves to `balanced` for small and medium models, and `throughput` for large and huge models.

## Caveats

- Benchmark numbers depend on hardware class, prompt mix, quantization, and model family.
- README comparison tables should be read as directional comparisons rather than universal guarantees.
- If you change serving defaults or performance-sensitive behavior, rerun the harness and update any user-facing claims that depend on those results.
