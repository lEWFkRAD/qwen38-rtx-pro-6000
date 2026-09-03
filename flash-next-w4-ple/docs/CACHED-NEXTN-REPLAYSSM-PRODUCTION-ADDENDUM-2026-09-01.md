# Cached NEXTN + ReplaySSM production addendum

- **Qualification date:** 2026-09-01
- **Hardware:** 1x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB
- **Model:** `RadixArk/Qwen3.8-Flash-Next-NVFP4`
- **Runtime:** pinned SGLang image plus the signed-W4 PLE and SM120 QSA overlays
- **Status:** qualified and promoted, but still probationary

This addendum updates the August 27 native-MTP result with the first qualified
profile that combines device prefix caching, the checkpoint's native shallow
NEXTN/MTP head, ReplaySSM speculative-state management, and full decode CUDA
graphs. It is a field report for one machine and one pinned downstream stack,
not a universal recipe or an upstream benchmark.

## Result

The final candidate improved decode and end-to-end throughput over a fresh
same-day cached/no-speculation control at every tested concurrency. Each cell
used one warmup followed by six measured runs with 512 forced output tokens.
Throughput is aggregate across concurrent requests.

| Metric | c1 control | c1 final | c2 control | c2 final | c4 control | c4 final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decode tok/s | 97.882 | **144.563** | 185.666 | **255.986** | 332.794 | **434.883** |
| End-to-end tok/s | 96.462 | **141.267** | 183.224 | **249.695** | 328.357 | **424.206** |
| Maximum TTFT median | 89.775 ms | 87.605 ms | 82.908 ms | 97.957 ms | 121.493 ms | 131.459 ms |
| Decode difference |  | **+47.690%** |  | **+37.874%** |  | **+30.676%** |

The throughput gain carried a latency tradeoff: from the raw medians, maximum
TTFT improved by 2.171 ms at c1, but regressed by 15.049 ms at c2 and 9.967 ms
at c4.

The earlier no-cache EAGLE qualification measured 142.070 / 252.322 /
419.138 tok/s at c1/c2/c4. The final cached profile was still 1.755% / 1.452%
/ 3.757% faster, so it recovered the speculative speed while retaining prefix
reuse.

The same-day control is the correct promotion reference, but it is not a
strict one-variable causal experiment. The control exposed 133,376 tokens,
the candidate exposed 151,040, and desktop GPU-memory normalization occurred
between cold starts. The TPS harness required 60 continuous idle seconds for
the control and zero for the candidate; the candidate had instead passed a
separately recorded pre-TPS idle gate. The combined result does **not** isolate
the independent gain from NEXTN, ReplaySSM, CUDA graphs, or the QSA correction.

## Qualified serving profile

| Dimension | Value |
| --- | --- |
| Tensor parallelism | TP1 |
| Quantization | ModelOpt NVFP4 plus the signed-W4/group-16 host PLE sidecar |
| Context | 131,072 tokens |
| Running / queued requests | 4 / 16 |
| Requested shared-token cap | 163,072 |
| Candidate / promotion pools | 2x 151,040 / 2x 151,808 tokens |
| Target and draft KV | BF16 |
| Mamba state | FP32 |
| Prefill | 1,024-token chunks; FlashInfer linear prefill |
| Decode | Triton linear decode; normal overlap scheduling |
| Prefix cache | Device radix, LRU eviction, cache reporting enabled |
| Mamba radix | `extra_buffer`; 28 states maximum; one retained state per path |
| Speculation | Native NEXTN through SGLang `EAGLE`; 1 step, top-k 1, 2 draft tokens |
| Verification | Rejection sampling; speculative attention mode `prefill` |
| ReplaySSM | Speculative path enabled; cache length 16 |
| CUDA graphs | Full decode graphs for batch sizes 1, 2, 3, and 4 |
| Prefill graphs | Disabled |

The candidate launcher was SHA-256
`535b413ce0f4c9b62cfec9360723b39353f79e722722106e288d2602b917a4b3`.
Its paired readiness gate was
`97b668783a400ac8a2d547cc220de1a2c1085f272e034c3c1199d708b34660d5`.
The final QSA overlay was
`9e7b00734f67f3928c2f846db459178690224daef9ea02ea27b52a7940af52fb`.
Those hashes identify the tested private release graph; this minimized public
evidence set does not publish the production service wrapper or recovery
control plane as a turnkey launcher.

## Why one retained Mamba state per path mattered

Device radix caching on this hybrid model also retains recurrent Mamba state.
With 1,024-token prefill chunks, an 11K branch can otherwise leave roughly
eleven intermediate recurrent checkpoints. Multiple branches then consume the
state pool and evict the useful branch tails even though a single-prefix test
appears healthy.

`mamba-max-states-per-path=1` changed the retained topology from many
intermediate checkpoints to one useful state per radix path. The candidate
then passed all nine branch-matrix scenarios, including sequential and
concurrent cold/hot variants, and retained all five 11K branch tails.

## Cache qualification

- A realistic 54,028-token repeat reused 54,016 tokens (99.978%).
- A two-turn 13.6K conversation extension reused 99.560% of its prior prompt.
- Five independent branches each reused 11,008 of 11,028 tokens.
- All nine topology scenarios passed; the public receipt preserves cold/hot
  token counts and response hashes without prompt or response bodies.
- Candidate and promotion requalification produced the same realistic-repeat
  and five-branch reuse results.

## Semantic and long-context qualification

The final canary workload passed 34/34 attempts: two warmups plus 32 measured
cases. One expected over-limit HTTP 400 has no inference metrics, which is why
the receipt reports one missing-metrics result while the campaign itself
passes.

| Workload class | Measured requests | Weighted draft acceptance |
| --- | ---: | ---: |
| Chat | 9 | 89.777% |
| Math | 10 | 94.444% |
| Tools and strict JSON | 10 | 99.333% |
| Long retrieval | 3 | 71.383% |
| **Overall** | **32** | **88.330%** |

The long-context gates included:

- a leak-free 119,735-token prompt plus 512 completion tokens;
- blind exact retrieval from a 130,831-token prompt;
- correct HTTP 400 rejection for an over-limit input.

Promotion requalification repeated identity, cache, workload, long-context,
and fault gates. It again passed 34/34 attempts and recorded 89.539% weighted
acceptance overall. The promotion start exposed two 151,808-token pools.

**Promotion did not rerun throughput.** The c1/c2/c4 numbers in this document
belong to the exact passed candidate release that was subsequently installed
and requalified.

The full private gate also scanned the service, container, kernel, and GPU
logs and found no invalid-probability, CUDA, OOM, or NVIDIA Xid event in the
qualification window. Those operational logs are intentionally excluded from
this minimized public bundle.

## Public receipts

Every published JSON wraps a filtered source receipt, records the SHA-256 of
the sealed source bytes, and lists fields removed for publication. TPS
receipts omit the loopback URL, cumulative idle counters, and absolute clock
fields; readiness receipts omit ephemeral container IDs and local container
paths; cache and branch receipts omit host-branded schema labels. The workload,
cache, and branch receipts contain synthetic measurements and no prompt or
response bodies.

- [same-day cached control TPS](../evidence/cache-nextn-control-tps-20260901.json)
- [final candidate TPS](../evidence/cache-nextn-replayssm-candidate-tps-20260901.json)
- [candidate readiness](../evidence/cache-nextn-replayssm-candidate-ready-20260901.json)
- [candidate cache acceptance](../evidence/cache-nextn-replayssm-candidate-cache-20260901.json)
- [candidate branch matrix](../evidence/cache-nextn-replayssm-candidate-branch-matrix-20260901.json)
- [candidate workload](../evidence/cache-nextn-replayssm-candidate-workload-20260901.json)
- [promotion readiness](../evidence/cache-nextn-replayssm-promotion-ready-20260901.json)
- [promotion cache requalification](../evidence/cache-nextn-replayssm-promotion-cache-20260901.json)
- [promotion workload requalification](../evidence/cache-nextn-replayssm-promotion-workload-20260901.json)

The repository `SHA256SUMS` covers the complete public subtree, including the
receipts and their validator.

## Evidence boundaries and remaining work

- This is external single-node evidence, not TurboFit-native campaign output
  and not a request for automatic catalog or leaderboard promotion.
- The public bundle proves the stated sanitized measurements but does not
  expose internal service units, operator paths, traffic counters, journals,
  rollback commands, or real client content.
- The profile remains probationary pending a deliberate rollback-and-return
  drill, deterministic cold-start/reboot qualification, and longer multi-turn
  speculative endurance.
- A factorial NEXTN-only / ReplaySSM-eager / ReplaySSM-graph campaign remains
  future work. Do not assign the combined gain to ReplaySSM alone.
- Mixed long-prefill and interactive QoS still needs a dedicated fairness
  campaign; aggregate tok/s does not establish an interactive latency SLO.

## Operational lessons

1. Hybrid QSA/GDN/Mamba caching needs branch-topology tests; a single cache hit
   is not a retention qualification.
2. Native NEXTN creates separate target and draft KV pools. Readiness must
   require exactly two pools and require each pool to meet the context floor.
3. A healthy live mount does not prove that its bind source will exist on the
   next start. Artifact presence and hash checks belong before every restart.
4. Cold-start desktop VRAM materially changes exposed capacity, so the startup
   environment belongs in the release receipt.
5. A release is one sealed graph: checkpoint, PLE sidecar, image, overlays,
   launcher, readiness, caller token policy, receipts, and rollback.
