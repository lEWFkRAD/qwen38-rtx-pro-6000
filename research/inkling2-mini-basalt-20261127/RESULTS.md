# November 27 — Inkling 2 Mini on one RTX PRO 6000 (Basalt 12×262K, bit-exact turbo)

Production deployment and controlled A/B optimization of **Inkling 2 Mini** (`NVFP4` experts + `Q8_0` dense/head + `DSpark` drafter) served via **Basalt** on a single **NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB)** with **12 resident 262,144-token slots** and **FP8 KV cache**.

The promoted `pinned-20261127-turbo` binary (`sha256: 915b1e0f4e85528c45a694fc30f89a3b59027e00a8d48de010e6746224e97834`) is **100% cross-build bit-exact** with `pinned-20261127-graphfix` across 4-, 8-, and 12-slot `batch_identity.py` verification while improving isolated HTTP throughput by **+1.2% to +6.2%** (up to **+7.8%** pre-soak on sampled `N=8`) and cutting multi-slot worst-case TTFT by **25–29%**.

## Hardware & Production Configuration

- **GPU:** 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB VRAM, 188 SMs, SM120, 500 W cap)
- **Model pack:** `Inkling-2-Mini-Uncensored-NVFP4-q8.basalt` (16.2 GiB on GPU + 0.9 GiB embedding in host RAM)
- **Serving flags:**
  - `--parallel 12 --slot-full-context --max-context 262144 --kv fp8 --vram-reserve-mib 6000 --spec 8`
  - `--vision dynamic --answer-room 0 --thinking-loop-tokens 0`
  - `--conversation-cache-mib 8192 --conversation-cache-slots 8 --conversation-cache-min-tokens 2048 --conversation-cache-disk-mib 24576`
- **Co-resident GPU footprint:** Basalt (`:8011` behind `:8010` door proxy) + GPU sidecar (`:8120`, embeddings + reranker + Whisper STT, ~4.0 GiB); total VRAM used **89,446 MiB** (`7,873 MiB` free after retiring `dory.service`).

## Isolated A/B Benchmark (`graphfix` vs `turbo`)

Both binaries were measured back-to-back on an isolated `127.0.0.1:18011` instance (production `:8010` door paused, identical production flags) using `hillclimb/fixed_cn.py` (4 repetitions × 400 output tokens per stream; rep 0 warms CUDA graphs and is excluded from headline medians).

| Workload | Baseline (`graphfix`) | Promoted (`turbo`) | Aggregate Δ | Per-Stream Median | Max TTFT Median |
|---|---:|---:|---:|---:|---:|
| **`N=1` (`T=0`, greedy)** | 390.22 tok/s (`sd 9.46`) | **394.89 tok/s** (`sd 3.32`) | **+1.20%** | 395.17 → **399.78 tok/s** | 15 ms → 15 ms |
| **`N=4` (`T=0`, greedy)** | 454.07 tok/s (`sd 10.95`) | **481.67 tok/s** (`sd 2.28`) | **+6.08%** | 130.58 → **133.31 tok/s** | 368 ms → **261 ms (-29.1%)** |
| **`N=8` (`T=0`, greedy)** | 655.10 tok/s (`sd 0.40`) | **673.98 tok/s** (`sd 1.17`) | **+2.88%** | 95.56 → **97.71 tok/s** | 727 ms → **535 ms (-26.4%)** |
| **`N=12` (`T=0`, greedy)** | 704.60 tok/s (`sd 9.87`) | **736.44 tok/s** (`sd 0.37`) | **+4.52%** | 69.35 → **71.41 tok/s** | 1,115 ms → **838 ms (-24.8%)** |
| **`N=8` (`T=0.6`, thinking on)** | 631.88 tok/s (`sd 0.34`) | **671.27 tok/s** (`sd 4.06`) | **+6.23%** | 93.63 → **97.34 tok/s** | 743 ms → **530 ms (-28.7%)** |

Deterministic outputs (`--disable-thinking --temperature 0`) matched identically across all measured reps (`stable_outputs=True`) and between builds. A post-promotion live canary through the `:8010` door completed at **409.2 tok/s** (`12/12` drafts accepted).

## Parity & Cross-Build Bit-Exact Gate

- `inkling2_kernels_parity`: **0 failures** (window/full E4M3 attention, `post_attn`, `moe_finish` bitwise equal to solo)
- `inkling2_dspark_parity`: **0 failures** (Q8_0 fold, fold bias, recurrence split)
- `tools/batch_identity.py` (400 tokens, `--kv fp8 --spec 8`, Inkling fixtures) compared against baseline reference `gf_q8.json`:

| Concurrent Slots | Baseline (`gf_q8.json`) | Promoted (`turbo`) | Delta | Output Parity |
|---|---:|---:|---:|---|
| **4 slots** | 1,044.4 tok/s | 1,026.4 tok/s | -1.73% | **100% BIT-EXACT** (all 4 slots + reference) |
| **8 slots** | 1,028.0 tok/s | **1,028.8 tok/s** | **+0.08%** | **100% BIT-EXACT** (all 8 slots + reference) |
| **12 slots** | 972.9 tok/s | **1,049.5 tok/s** | **+7.88%** | **100% BIT-EXACT** (all 12 slots + reference) |

## What Changed

1. **Eliminated 23–33 ms/slot GPU admission freezes (`request_finish.cpp`, `session_engine.cpp`):**
   Every `BGEN` admission previously reset `BSlot = BSlot{}`, freeing and reallocating a 64 MiB `SuffixDrafter` hash table (causing 22–24 ms of page faults per slot while all active GPU slots sat idle) and deep-copying ~100 MiB `ConvCheckpoint` running states (+9 ms on slot 0). Preserving and `reset()`ing `BSlot::sfx` and `std::move`ing `ConvCheckpoint` dropped admission copy time from `23–33 ms` to `0.0–0.1 ms` per slot and cut multi-slot TTFT by `25–29%`.
2. **Batched + parallel DSpark drafting & smart draft budgeting (`dspark.cpp`, `dspark.hpp`, `kernels/dspark.cu`, `session.cpp`, `batch_round.cpp`):**
   - Packs up to 4 drafting sessions (`32` rows) through captured `wgemm(32)` stages and batched `dspark_head_base(<=8)` passes instead of separate 8-row passes.
   - Forks per-session DSpark attention (`attn_qk_prepare` + `gqa_window_attn`) and per-draft Markov/head-pick steps concurrently across each session's own CUDA stream (double-buffered across chunks).
   - Budgets draft count when spare rows `cap - n_act` are tight (`c=7..12`): at `c=12` (`spare=4`), selects 1 round-robin probe + 3 highest-acceptance slots to draft 1 token each instead of wasting 8 unverified DSpark passes; at `c=7..8`, caps per-slot draft depth to `average + 1` (minimum 2) while leaving `c<=5` uncapped.
3. **Conditional parallel post-forward sampling (`batch_driver.hpp`, `session.cpp`):**
   Forks `batch_take` and `batch_draw` onto per-session streams when any active request in the batch uses non-greedy sampling (`T > 0`), running CUB wide radix sorts (`sample_wide`) concurrently across slots while avoiding multi-stream sync overhead on greedy (`T = 0`) steps.
4. **NVFP4 MoE exact `n=3, 4` specializations & batch read-ahead skip (`iq_kernels.cu`, `session.cpp`):**
   Adds `EXACT_N = true` specializations for `n == 3` and `n == 4` in `native_gu_multi_kernel` and `native_down_sub8_kernel`, and skips stale 8-expert `expert_read_ahead` during multi-slot batch windows (where 50–70 experts are active).

## Rejected Experiments (Measured & Discarded)

- Two-pass accumulator split in `row_dot_sub8` (`iq_kernels.cu`): broke NVFP4 cache-line coalescing (-4% to -6% uniformly).
- `GRP_NC = 8` and `native_gu_sub16_kernel`: register spills (-3% to -14.4% in NVFP4 MoE microbenchmarks).
- `__constant__` LUT for `nvfp4_scale_dev`: constant-cache serialization (-7% to -19%).
- Single-stream `DsparkHook::draft_batch()` without per-session stream forks: serialized 1-CTA Markov/head-pick and attention across sessions.
- Clamping `per = 1` draft at `c=8`: cut mean verified window length `dl` from `1.80 → 1.60` (-2.1% HTTP throughput).

## Files

- `turbo.patch` — full source diff over upstream Basalt (includes device-indirect `graphfix` + all `turbo` optimizations)
- `receipts/http_baseline/{c1_t0,c4_t0,c8_t0,c12_t0,c8_t06}.json` — isolated `fixed_cn.py` baseline receipts (`pinned-20261127-graphfix`)
- `receipts/http_turbo/{c1_t0,c4_t0,c8_t0,c12_t0,c8_t06}.json` — isolated `fixed_cn.py` promoted receipts (`pinned-20261127-turbo`)
- `receipts/batch_identity_turbo.json` — cross-build bit-exact `batch_identity.py` receipt (4, 8, 12 slots)
