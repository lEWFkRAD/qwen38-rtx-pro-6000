# September serving experiments

Evidence: [serving notebook excerpts](receipts/september-serving.json) and
[GLM closeout excerpts](receipts/glm.json). These are recorded results, with
original notebook snapshot hashes; the complete raw September campaigns are
not included in this checkout.

## GLM-5.3-Flash: token speed failed the answer gate

The September 4 four-user test used approximately 89,400-token prefixes and
512 measured output tokens per user. No speculation produced 34.3605 generated
tok/s per user; Q8 DFlash depth 3 produced 42.9186. **Both failed answer quality.**
Every Q8 output exhausted its budget in reasoning and produced no final-answer
tokens. The no-speculation arm produced final text only in two slots, with
substantive release/hold errors.

Prompt/source/token hashes matched, but the fourth slot's sampled prefill seed
differed. This is not an exact four-slot continuation comparison. Work stopped
rather than treating token rate as useful task throughput. The experiment used
an external patched llama.cpp runtime, not a qualified Colibri engine.

Qwen recovery initially failed actual KV-capacity admission. The September 4
repair retained EAGLE 2/1/3 and raised memory fraction from .965 to .972:
136,704 tokens per BF16 pool supported the 131,072 context contract.
This was a recovery check, not a fresh performance benchmark.

## NVIDIA Flash-Next checkpoint compatibility

The September 5 candidate pinned NVIDIA revision
`fab0aecb760cec45227f6656abcaafa11abca87a`. Its 128 FP8 PLE source shards and
outer scale matched the existing signed-W4 sidecar, allowing reuse of those
immutable sidecar bytes with updated provenance. This comparison therefore
**did not test the fidelity of NVIDIA's stock FP8 PLE path**.

Compatibility work expanded 1,536 FP8 block-128 MTP expert weights with their
stored scales into BF16, selected the mixed quantization loader and explicit
FlashInfer CUTLASS path, and retained 131,072 context, four requests, BF16 KV,
FP32 Mamba, EAGLE 2/1/3, ReplaySSM and chunk 1,024. Startup measured 81.31 GiB
weight memory and 163,072 KV tokens. Four exact/math/JSON/tool smoke checks
passed. Startup success alone did not establish broad quality or long-context
equivalence.

## SGLang update and admission scheduling

The September 12 CUDA-13 nightly candidate `a2077862` passed 34/34 deterministic
cases, a 119,735-token retrieval plus 512 generated tokens, and prefix-cache
reuse. The synchronized c4 harness measured:

| Metric | Sealed baseline | Nightly candidate |
| --- | ---: | ---: |
| Median decode rate, aggregate tok/s | 464.019 | 479.464 |
| Median cell-maximum TTFT, ms | 93.312 | 160.701 |

The candidate was rejected for the observed first-token delay. Later diagnosis
corrected an earlier explanation: SM120/XQA support was still present, and both
images displayed approximately 90/160 ms modes. The barrier result alone does
not identify a runtime-specific cause.

A separate baseline-image A/B/A tested
`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1`:

| Earliest response median, ms | A1 | B1 | A2 |
| --- | ---: | ---: | ---: |
| c2 | 155.38 | 80.09 | 156.83 |
| c4 | 159.22 | 80.33 | 158.26 |

All 330 structural requests passed, and the eight matched cells at each
concurrency reproduced the benefit and reversal. Batch shapes stayed 1+1 and
1+3. The change delivered the first completed prefill earlier; it did not prove
faster GPU decoding, better tail latency, or higher end-to-end throughput.
Live traffic was allowed, so absence of detected interference is not isolation.
The one-setting change was retained after separate correctness, long-context
and repeat gates.

The September 21 v0.5.20 canary ended after a runner declaration bug and a
cancelled retry. No valid B1 result or promotion exists for that attempt.

## Personal 27B profile

The September 23 experiment used the Heretic native-MTP-preserved NVFP4 revision
`98b34bd9ea6a74fb6e42ac482340fc44113dfa30`, SGLang 0.5.20,
180,224 total tokens per request, three active requests, FP8 KV, FP32 state,
EAGLE 2/1/3 and ReplaySSM. Shared KV capacity was 546,816 tokens.

Warm-cache 512-token streaming through the model selector measured 121.2 tok/s
solo and 350.2 aggregate tok/s with three submitted sessions. Prefix reuse
contributes. Full-context and broad quality stress were not run, so those
figures do not establish long-context speed or quality.
