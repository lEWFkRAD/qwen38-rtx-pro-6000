# Option #2 Saved for Further Testing — Screened Greedy LM Head (`BASALT_HEAD_SCREEN=1`)

The zero-VRAM transposed 2-bit screened greedy LM head (`q80_screen_top1_rows`) is compiled directly into `pinned-20261127-turbo3/basalt-engine` (SHA-256 `f2e158f8bf587e544b1b1072515b1ff6a8dfa249fe6a33286f452f91831bb66f`), gated behind `BASALT_HEAD_SCREEN=1` so production stays 100% mathematically bit-exact by default.

## How It Works
- Reuses DSpark's already-packed transposed 2-bit copy of `m_.head()` (`w_->head_sc`, 0 extra VRAM) via `q80_screen_top1_rows(..., transposed=true)`.
- Reads ~129 MB instead of 489.6 MB (`200,058 x 2,304` Q8_0) per greedy verify window (`T=0`, no penalties, no bias, no raw logits requested).
- Lazy `ensure_logits()` materializes full `logits_` on demand if `row_logprobs()`, `resample_row()`, or `copy_logits()` is ever called.
- Current target rescore window in `basalt-turbo3/src/kernels/cuda/q80_screen.cu`:
  - `kTargetMaxCand = 256`
  - `kTargetDelta = 16.0f`

## Measured Results (Clean GPU, Stock 500W, ECC ON, P1 13,365 MHz, `kv=fp16`, `spec=8`, `8k`)
- `turbo3` default (100% bit-exact, `2048/2048`):
  - `code-v1`: avg `505.1` / max `573.7` (wall `583.3 tok/s`)
  - `reasoning-v1`: avg `447.6` / max `466.5` (wall `472.9 tok/s`)
  - verify mean/med: `6.256 / 6.027 ms`
- `turbo3` + `BASALT_HEAD_SCREEN=1` (`2047/2048` exact = `99.95%`; `1024/1024` exact on `reasoning-v1`):
  - `code-v1`: avg `504.6` / max `576.4` (wall `588.3 tok/s`)
  - `reasoning-v1`: avg `473.7` / max `493.8` (wall `502.2 tok/s`, `+26.1 tok/s` avg over unscreened)
  - verify mean/med: `5.923 / 5.728 ms` (`-0.333 ms/window`)
  - single diff across 2,048 tokens: `code-v1[0]` tok 211 (`220` `'Ġ'` vs `150197` `'ĠWe'`)

## Ideas for Next Testing Pass
1. Test `kTargetMaxCand = 512` (fits 1:1 in `blockDim.x = 512` in `head_proxy.cuh`) and/or `kPer = 6..8` per tile to check whether the last `1/2048` miss disappears and measure rescore overhead.
2. Parallelize `q80_screen_final_rows_kernel` across multiple CTAs per row when `n > 64` so widening to 256–512 candidates costs `< 10 us` instead of `~50 us`.
3. Inspect the exact logit gap between `220` and `150197` at `code-v1[0]` tok 211 (to see whether it was a tile `kPer=4` miss, a `kTargetMaxCand=256` miss, or a sub-0.01 near-tie).

## Quick Launch
```bash
BASALT_HEAD_SCREEN=1 /home/jeffrey-watts/model-tests/basalt-inkling-20261121/pinned-20261127-turbo3/basalt-engine ...
```
Or run `./run-turbo3-headscreen-bench.sh` (when the GPU is free) to compare exact vs `BASALT_HEAD_SCREEN=1` head-to-head.
