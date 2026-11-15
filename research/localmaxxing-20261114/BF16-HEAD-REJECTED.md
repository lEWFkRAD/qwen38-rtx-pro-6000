# BF16 LM head vs the Q6_K pack head: rejected on measurement (2026-11-15)

The Qwen3.8-Flash-Next leaderboard has a 296 tok/s unverified run from pixelkaiser using an NVFP4 model with the LM head kept in BF16 (two-card rig). This page checks whether the same mixed-precision trick helps our Strata IQ3_S pack on one RTX PRO 6000.

## What our pack actually ships

The IQ3_S GGUF keeps the head (`output.weight`, 2560 x 248320) at **Q6_K**, not a low-bit quant. Attention projections are Q4_K to Q6_K per layer. So the candidate gain is speed, not precision; Q6_K is already within about half a percent of BF16.

## Method

Strata's engine refuses a BF16 `output.weight` today (NativeHead accepts the mmvq quantized types only; a BF16 head would need an engine patch). Before patching, the question "would BF16 even be faster" is answerable offline: the engine ships `b6_mmvq_bench`, which times the exact head shape through the exact kernel a patched engine would use (`bf16_gemv_fp32_mmvf_multi`, the same kernel the router and shared experts use) against the current quantized mmvq path. Cases `head Q6_K` and `head BF16` at 2560 x 248320 were added to the bench and run on the production card, 300 reps, median ms per call.

## Results

| Head | 1 col | 2 cols | 4 cols | 8 cols |
| --- | ---: | ---: | ---: | ---: |
| Q5_K | 0.270 | 0.291 | 0.412 | 0.643 |
| Q6_K (current pack) | **0.321** | **0.357** | **0.489** | **0.752** |
| BF16 | 0.776 | 0.778 | 0.788 | 1.369 |

All bitwise-identical across A/B copies, 0 comparisons failed.

## Finding

**Rejected.** The head is memory-bandwidth bound and BF16 is 2.4x the bytes of Q6_K, so the BF16 kernel is 1.8 to 2.4x slower at every width. The trick works on vLLM NVFP4 because that stack's 4-bit head path is inefficient and BF16 hits better kernels there. Strata's quantized mmvq kernels are already the fast path. No engine patch, no pack variant, no maintenance window needed.

Side observation: the bench also shows Q5_K consistently faster than Q6_K at head shape (0.27 vs 0.32 ms single-column). A Q5_K head would save ~50 us per decoded token (about 0.5% of a 65 ms token) at a small precision cost; not worth a pack rebuild on its own.

## Reproduce

`ninja b6_mmvq_bench` in the Strata build dir, then `./b6_mmvq_bench 300 0 1`. The two added cases live in `src/kernels/b6_mmvq_bench.cpp` CASES.
