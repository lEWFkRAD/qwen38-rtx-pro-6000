# Native MTP/EAGLE production addendum — 2026-08-27

This is a post-publication addendum to the original Flash-Next W4 PLE field
report. It records a later, controlled throughput promotion on the same RTX PRO
6000 deployment. The original PDF and `scripts/serve.example.sh` remain the
conservative non-speculative baseline and are intentionally unchanged.

## What changed

The checkpoint contains a native one-layer MTP head. The promoted profile adds:

```text
--mem-fraction-static 0.955
--speculative-algorithm EAGLE
--speculative-num-steps 1
--speculative-eagle-topk 1
--speculative-num-draft-tokens 2
--speculative-use-rejection-sampling
```

The decode CUDA-graph buckets are `[1,2,3,4]`; prefill graphs remain disabled.
Everything below stayed fixed:

- exact model revision and W4 PLE artifact;
- TP=1 on SM120;
- 131,072-token request context;
- BF16 target and draft KV cache;
- FP32 Mamba state;
- 1,024-token chunked prefill;
- four running and sixteen queued requests;
- FlashInfer GDN prefill, Triton GDN decode, and XQA QSA decode;
- radix cache, mixed/dynamic chunking, and automatic truncation disabled.

The runtime requested 163,072 shared tokens. Both target and draft KV pools
resolved to 139,072 tokens, leaving an 8,000-token margin above the qualified
per-request context.

## Comparable throughput

The control is the accepted 131K-context BS1/2/3/4 graph profile immediately
before MTP promotion. Values are aggregate output tokens per second for forced
512-token greedy decodes.

| Concurrency | Accepted control | Promoted production | Change |
| --- | ---: | ---: | ---: |
| c1 | 100.181 | 133.545 | +33.3% |
| c2 aggregate | 179.779 | 233.049 | +29.6% |
| c4 aggregate | 318.770 | 389.299 | +22.1% |

The c4 receipt contains one cold/outlier repeat at 257.343 tok/s with elevated
TTFT; subsequent repeats reached 389–407 tok/s. The published median is retained
rather than deleting the outlier.

The rejection-sampling candidate also completed stochastic workloads. Its
three-cell c1/c2/c4 medians were 114.650/200.404/333.618 aggregate tok/s with an
average speculative acceptance length of approximately 1.73. Same-seed
stochastic output was not byte-reproducible because deterministic inference was
disabled; this is a reproducibility caveat, not evidence that rejection
sampling was bypassed.

## Validation and fit

- 18/18 exact behavior, arithmetic, tool-call, strict-JSON, and reasoning-off
  cases passed.
- A 119,424-token prompt produced 512 completion tokens, reproduced the planted
  marker string, and emitted zero reasoning tokens. Historical-oracle erratum:
  its final instruction disclosed the combined marker, so this proves
  long-context generation stability rather than blind early/middle/late
  retrieval. The later workload campaign uses a leak-free replacement.
- A 130,500-token prompt returned the exact planted retrieval marker.
- A 131,272-token input was correctly rejected with HTTP 400 against the
  131,072-token limit.
- Maximum observed GPU use was 94,987 MiB; minimum free VRAM was 2,257 MiB.
- Maximum observed temperature was 68 C and maximum power was 454.61 W during
  the long-context gate.
- The managed production service finished with zero restarts, no container OOM
  flag, and no service/kernel fatal or Xid matches.

The stressed VRAM margin is deliberately narrow. Do not add another draft
width, graph family, model, or GPU workload without a separate memory and
correctness qualification.

## Workload acceptance and depth-sweep follow-up

The stricter workload campaign subsequently passed 34/34 on the promoted A0
arm. Weighted accept length was content dependent: chat `1.904`, tool/strict
JSON `2.020`, exact math `2.167`, and long retrieval `1.774`. The replacement
119K oracle did not disclose its joined answer: a 119,735-token prompt plus 512
completion tokens returned the independently planted codes, while a distinct
130,831-token prompt returned its blind exact marker.

The legal A1 depth arm (`steps=2`, `top-k=1`, `draft_tokens=3`) was tested only
as a guarded c1 candidate. It resolved target and draft KV pools to 97,664
tokens, below the 131,072 context, and was rejected before readiness or any
inference. A2 was therefore not attempted. Exact A0 was restored and its final
cold boot resolved both pools to 146,240 tokens with 5.562 GB reported ready
headroom, zero restarts, and no post-readiness fatal match. The A0 workload and
depth-Pareto receipts are linked below.

## Profiling result

An accepted decode trace showed that the custom W4 PLE gather represented only
about 0.05% of CUDA kernel time and was fully hidden behind other work. The
sidecar was the fit enabler, not the steady-decode bottleneck. That finding is
why native MTP was tested before additional PLE-kernel tuning.

## Evidence receipts

| Receipt | SHA-256 |
| --- | --- |
| [`post-promotion-greedy-c124-3x.json`](../evidence/post-promotion-greedy-c124-3x.json) | `3ee661455cba9a8f48080d11a3c28b1c562b6f9db8f6c9ed2c7cfd996a324821` |
| [`post-promotion-quality-gate.json`](../evidence/post-promotion-quality-gate.json) | `3e0503cd83fecc2c758bdcc7159be61c5a86f19613b6578375d6ca159a2799e5` |
| [`post-promotion-longctx-gate.json`](../evidence/post-promotion-longctx-gate.json) | `d2094c0da402cd1edff574de6d876e0d754291d7d236924927b52b8f7c254c74` |
| [`mtp-workload-a0-1-1-2-bf16kv-20260827.json`](../evidence/mtp-workload-a0-1-1-2-bf16kv-20260827.json) | `f674da2a77dbbc3640b3a94820a3581d6a44f1c05dfe78e8599f8adec6dde9e8` |
| [`mtp-workload-rolling-v2-20260827.json`](../evidence/mtp-workload-rolling-v2-20260827.json) | `2e065a0d33cd1c0da5a8287fbbcc772d17b9d5693c5c6a2396257f67c637f6bc` |
| [`mtp-depth-pareto-20260827.json`](../evidence/mtp-depth-pareto-20260827.json) | `78d9c88025a0b0fb9f70329ff81bd434c7a28da3d4a481422f79f358f3e0f5c3` |

## Operational boundary

This promotion does not make speculative settings portable to every GPU,
runtime revision, or fine-tuned checkpoint. Rebuild the W4 PLE sidecar for any
changed source weights, preserve rejection sampling for stochastic clients,
and rerun the full behavior, long-context, memory, cold-start, and corruption
gates.

The service's fallback is the sealed non-speculative Flash-Next profile—not an
older model. The W4 PLE artifact and original source pins remain unchanged.

## Future bring-up workflow

[`SouthpawIN/turbofit`](https://github.com/SouthpawIN/turbofit) is a promising
framework to evaluate for the *next* model campaign because it separates
hardware inventory, engine audition, physical-fit evidence, immutable failure
receipts, and intelligence scoring. It was not used for this promotion and is
not installed in this deployment.

Any trial should begin in an isolated fork or lab profile. Its catalog,
fallback, multimodal, and service-controller defaults are separate policy
choices; adopt the campaign/evidence machinery only after reviewing those
defaults and proving that it cannot mutate or route production unexpectedly.
