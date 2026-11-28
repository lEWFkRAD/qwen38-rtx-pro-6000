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

## Update — `pinned-20261127-turbo2`: Bit-Exact Solo Kernel Suite (#1 Overall on LocalMaxxing: 552.6 tok/s) + RSI FP-Solve

Building directly on `pinned-20261127-turbo`, **`pinned-20261127-turbo2`** (`sha256: 9064330c58bff0f3f701567961c38d95edf6860299b50fa6410e1711c8d73885`, diff in `turbo2.patch`) adds five **100% cross-build bit-exact** CUDA kernel and draft-policy optimizations targeting solo and multi-slot decode latency:

1. **Q8-plane projections (`native_mmvq.cu`):** 2-rows-per-block `small_q8p_body` reusing `Q81Block` activation loads with PDL weight prefetch (`NPRE = 2` for `NC = 1..3`, `NPRE = 1` for `NC = 4`, `NPRE = 0` for non-MMA `NC >= 5`), plus lowering bit-exact INT8 Tensor Core `native_q8p_mma_kernel` (`pre[3]` PDL) to `NC >= 5` for `n_out >= 4096` (`proj` 2304→8664 verified `bitdiff_vs_nc1 = 0`).
2. **MoE Router (`router.cu`):** 2-row-per-block `router_logits_opt_kernel<3, 2, NT_MAX>` (`NT_MAX = 2` on decode `T = 3..32`, `NT_MAX = 8` otherwise; -11% to -41.5%) + atomic-free `router_select_direct_kernel<128>` (0 atomics, 2 `__syncthreads()`, PDL bias preload; -28.8%).
3. **Loop Controllers (`glue.cu`):** tight-`RMAX` `O = 2` `ctrl_down_kernel` dispatch (-7% to -24%) + 128-bit `uint4` `ctrl_up_opt_kernel<RMAX>` for rank-128 controllers (-2.4% to -5.2%).
4. **NVFP4 MoE Single-Pass Width (`iq_kernels.cu`):** expands `kGuNc` / `kDnNc` from `6` to `8` for NVFP4 (`TG/TD == 40`), avoiding redundant 2-pass expert weight reads on `T = 7..8` verify windows.
5. **All-VRAM `DraftPolicy` Prior (`draft_policy.cpp`):** calibrates `kShape` (`1.0 → 2.15`) to measured Blackwell all-VRAM window scaling instead of the default CPU-offload curve (`1.0 → 3.60`).

![LocalMaxxing Verified Leaderboard & Per-Window Latency](chart1_leaderboard_and_kernels.png)

### Verified LocalMaxxing Submissions (`thinkingmachines/Inkling-2-Mini`, stock `Inkling-2-Mini-NVFP4-q8.basalt`)

All four runs cleared server-side canonical verification (`APPROVED`, `verifiedRun: true`, `verificationIssues: []`) on one **RTX PRO 6000 Blackwell Workstation Edition** (stock 500 W, ECC ON, CUDA P1 13,365 MHz memory clock):

| Prompt | KV Cache | `tokSOut` | Prefill | TTFT | DSpark Accept | Live Submission |
|---|---|---:|---:|---:|---:|---|
| **`code-v1`** (`256` tok) | `fp16` | **552.6 tok/s** (#1 Overall) | 3,040.2 tok/s | 79.93 ms | 196/246 (79.7%) | [cmwzqhlba000drv01oxl1qyx4](https://www.localmaxxing.com/en/speed-tests/cmwzqhlba000drv01oxl1qyx4) |
| **`reasoning-v1`** (`256` tok) | `fp16` | **513.5 tok/s** (#1 Reasoning) | 3,931.8 tok/s | 74.77 ms | 188/255 (73.7%) | [cmwzqhm7g000grv01t8z2axgq](https://www.localmaxxing.com/en/speed-tests/cmwzqhm7g000grv01t8z2axgq) |
| **`code-v1`** (`256` tok) | `fp8` | **494.6 tok/s** | 3,039.3 tok/s | 79.86 ms | 192/234 (82.1%) | [cmwzptydq0007rv01ilvkhn1z](https://www.localmaxxing.com/en/speed-tests/cmwzptydq0007rv01ilvkhn1z) |
| **`reasoning-v1`** (`256` tok) | `fp8` | **467.1 tok/s** | 3,791.9 tok/s | 77.53 ms | 186/243 (76.5%) | [cmwzptz4z000arv01ozwma02c](https://www.localmaxxing.com/en/speed-tests/cmwzptz4z000arv01ozwma02c) |

- **Cross-build bit-exact gate (`batch_identity.py` vs `gf_q8.json`):** `4 slots = 934.5 tok/s`, `8 slots = 927.0 tok/s`, `12 slots = 936.6 tok/s` — **PASS (100% IDENTICAL)** across every slot and reference; `0 failures` across `inkling2_kernels_parity`, `inkling2_dspark_parity`, `mmvq_multi_parity`, and exhaustive direct-link kernel checks.

### Zero-Backprop RSI Loop-Dynamics Tuneup (`rse_looper` Staged `FP-Solve`)

Alongside the kernel work, we ran our zero-backprop closed-form **Staged `FP-Solve`** loop-dynamics solver (`lam_keep = 6.0`, 438 forward passes, **111.0 s** wall time, 168 loop-controller/attention/gate matrices across all 12 looped layers, **0 MoE expert weights modified**) on `Inkling-2-Mini`:

![RSI FP-Solve 4-Pass Loop Rescue & DSpark Preservation](chart2_rsi_fpsolve.png)

| Mode | Before `FP-Solve` | After `FP-Solve` (111 s, 0 Backprop) | Delta |
|---|---|---|---|
| **3-Pass Math** | 11/12 (91.7%, 0.0% loops) | **11/12 (91.7%, 0.0% loops)** | Held 100% (zero regression) |
| **3-Pass Tool Calling** | 4/4 (100%, 0.0% loops) | **4/4 (100%, 0.0% loops)** | Held 100% |
| **4-Pass Math** | 2/12 (16.7%, **83.3% rep loops**) | **9/12 (75.0%, 0.0% rep loops)** | **+58.3% acc, -83.3% loops** |
| **4-Pass Tool Calling** | 0/4 (0.0%, 25.0% loops) | **4/4 (100%, 0.0% loops)** | **+100% tool accuracy** |
| **DSpark Visit Cosine Sim** | 1.0000 | **Visit 5: 1.0000 · Visit 23: 0.9963 · Visit 41: 0.9957** | Drafter-compatible |

## Files

- `turbo2.patch` — full bit-exact source diff over upstream Basalt (`graphfix` + `turbo` + `turbo2` solo kernels + `DraftPolicy`)
- `turbo.patch` — earlier `pinned-20261127-turbo` diff
- `chart1_leaderboard_and_kernels.png`, `chart2_rsi_fpsolve.png` — benchmark & RSI FP-Solve charts
- `receipts/localmaxxing/*.json` — verified LocalMaxxing run receipts (`552.6`, `513.5`, `494.6`, `467.1` tok/s)
- `receipts/batch_identity_turbo2.json` — cross-build bit-exact `batch_identity.py` receipt for `turbo2` (4, 8, 12 slots)
- `receipts/rsi/onyx_live_tuneup_report.json` — full 12-layer Staged `FP-Solve` before/after evaluation receipt
- `receipts/http_baseline/{c1_t0,c4_t0,c8_t0,c12_t0,c8_t06}.json` — isolated `fixed_cn.py` baseline receipts (`pinned-20261127-graphfix`)
- `receipts/http_turbo/{c1_t0,c4_t0,c8_t0,c12_t0,c8_t06}.json` — isolated `fixed_cn.py` promoted receipts (`pinned-20261127-turbo`)
- `receipts/batch_identity_turbo.json` — cross-build bit-exact `batch_identity.py` receipt (4, 8, 12 slots)
