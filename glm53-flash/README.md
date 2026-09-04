# GLM-5.3-Flash on one RTX PRO 6000: progress through September 4, 2026

The best measured result is **36.1237 generated tokens/s per user for four concurrent users**, each with an 89,400-token prefix, one sampled seed, and 512 measured output tokens. The target remains **60 tokens/s per user**. Correctness and profiling work has narrowed the remaining questions, but neither performance nor useful-answer acceptance has been reached.

## Configuration and measurement

The experiment uses one **NVIDIA RTX PRO 6000 Blackwell, 96 GB**, a community **AJ-IQ2_XXS GLM-5.3-Flash GGUF**, and a patched, text-only llama.cpp runtime. The two target shards total **87,346,006,560 bytes (81.35 GiB)**. All 46/46 target layers were GPU-offloaded; no CPU expert spill was configured. The runtime reports 313,326,811,966 loaded parameters. This count has not been reconciled tensor by tensor with the publisher's 320B-total/18B-active model headline.

Four slots each have a 90,112-token allocation. Prefixes were prepared sequentially before a barrier released all four measured streams. Each user's rate uses the **same wall interval, from the earliest first output to the latest last output**. Its numerator excludes that user's first output chunk, which establishes the start boundary: **511 tokens for the best run's 512-token outputs**. The measurement includes skew between streams and excludes long-prefix preparation. EOS was enabled.

| Runtime | Target cache | DFlash2 draft weights | Native draft depth | Common-window tokens/s per user |
| --- | --- | --- | ---: | ---: |
| v8 | Q8 | BF16 | 3 | 35.3066 |
| v8 | Q8 | Q8 | 3 | **36.1237** |
| v9 | Q4 | Q8 | 7 | 25.3180 |
| v9 | Q4 | Q8 | 3 | 35.5771 |

Every row produced exactly 512 streamed and reported tokens per user, reused the full 89,400-token prefix, and evaluated one appended prompt token. The best run's common window was **14.145857224 seconds**, with **12.558695700 seconds** of four-way output overlap. Its warm-append first-token latency was 72–187 ms, excluding prefill. There was one measured run per configuration with different prompt nonces; a repeated-run median is unavailable. The Q4 rows use the separate v9 cache implementation, so this is not a controlled cache-precision ablation.

Q8 draft GPU weights used **1,187.23 MiB**, versus **2,234.07 MiB** for BF16, saving **1,046.84 MiB**. Available-memory samples establish observed fit, not peak-memory headroom.

## Correctness findings

The sparse attention work corrected a CUDA padding launch-dimension problem and retained exact key/mask checks. The v8 change kept 512-wide VKQ accumulation, rescaling, and final staging/reduction in FP32. It passed **13/13 CUDA fixture stages**, against **11/13** for the prior control, without relaxing the **3e-4 absolute / 1e-3 relative-L2** tolerances. Maximum Q8 long-parity absolute error fell from **1.90997124e-3** to **1.12354755e-5**. Sampled independent FP64 checks corroborated the numerical correction; this does not establish real-model semantic correctness.

The earlier long-output workload used an ignored `enable_thinking` setting; this template instead consumes `reasoning_effort`. Its outputs included excessive reasoning and repetition, so the speed table does **not** establish useful-answer throughput.

A separate matched 4K source-grounded task compared Low/High reasoning with speculation disabled and with the Q8 draft. Each arm contained 32 requested rows across four users. Rendered prompts and token hashes matched across speculation modes at the same effort.

| Reasoning effort | No speculation | Q8 draft, depth 3 |
| --- | ---: | ---: |
| Low | 27/32 correct rows | 27/32 correct rows |
| High | 31/32 correct rows | 31/32 correct rows |

Both High arms produced identical final answers. Their remaining error approved a record with expected quantity 20 and received quantity 21 despite a rule requiring zero difference. One High/no-spec reasoning trace also triggered a repetition heuristic, independently of its correct final rows. These bounded results do not assign the errors to target quantization or speculative decoding.

The evaluator also incorrectly applied a limit-stop cache formula to EOS. A new source-derived contract distinguishes exact no-spec occupancy from bounded speculative EOS occupancy. All four matched arms passed the corrected accounting. The exact 512-token limit-stop contract remains unchanged; historical failed receipts remain failed.

## Profiling and next experiment

Neither profiling attempt produced valid bottleneck evidence. The first capture began after decoding. During the second, outputs stopped after **13/21/14/17 tokens** while profiling started; stopping collection timed out, and the report contained **no CUDA events**. Timing is consistent with an instrumentation-associated stall, but does not prove its internal cause or establish which tracing backend was active.

The next workflow is prepared but **not executed**: explicitly request software CUDA tracing, start collection while idle after prefills, stop after generation, and audit a three-second interval inside all four output streams. A short canary must first show actual CUDA activity and completed accounting before another long-context trace. Profiled timing remains diagnostic. The experiment is stopped, and baseline-service recovery was verified at **01:12:30 UTC on September 4**.

## Provenance

- Publisher: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash); the tested GGUF/runtime is a separate community conversion.
- Model: [aj9o9/GLM-5.3-Flash-GGUF](https://huggingface.co/aj9o9/GLM-5.3-Flash-GGUF/tree/07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2), revision `07c62fcdeaf1c05d22bd123c3da8058a1b1e63e2`.
- Runtime base: [llama.cpp PR 27752](https://github.com/ggml-org/llama.cpp/pull/27752), commit `c9ddd6821c93871c53741344d35d1e440d60d9ea`, plus separately recorded experimental patches.
- v8 snapshot-manifest SHA-256: `6b9542d8a7a966f6ebc789b333e2eabfddd198165503d37b4d6b5e8b54efce06`.
- Cumulative v8 source-patch SHA-256: `9d84fdbffbeef057fd44091d2290377e5a13b26c2895dff9141e931b12f32e27`.
- Experimental DFlash2 Q8 draft SHA-256: `d21f4f7166e5ade490479bb3886909bf37220bc618b4d8047c26400075d5f747`.
- Best-run raw-receipt SHA-256: `6c7b54fa0d6ff9aaea3a31f14ff37e239023ced76e1a0f17a72e03db786741aa`.

See [measurement details and normalized commands](measurement-notes.md) and the [minimized evidence summary](evidence-summary.json). Source receipt digests identify the original records; the public summary is an allowlisted derivative, not the original raw logs.
