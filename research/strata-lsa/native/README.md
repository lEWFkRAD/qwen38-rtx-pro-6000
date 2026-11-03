# Native Strata LSA integration — 2026-11-03

The opt-in CUDA engine integration is complete and tested. Production remains paused under Jeff's existing research HOLD; Sunshine is stopped. The candidate was not promoted: its measured eight-slot decode throughput remains lower than the qualified control.

Source commit: `876307a9ded78bbca941189e63230b33cfdc6e2a` on local branch `strata-lsa-integration-20261103` in `strata-src/`.
Decoder: `build-engine/strata`.
Maps: `receipts/flashnext-learned-native-v1.bin` (SHA-256 `a5a69b217f23d9a2f7995516a91d1baa81e25b7733159ac5c9c1d0aa28c5342c`).

## Results and decision

Read `INTEGRATION-RESULTS.md` for the measured comparison and exact receipts. The final candidate retains the IQ3_S model/control vector, ordinary resident INT8 KV, eight slots, 131072 context, all 24576 GPU experts, MTP and image input.

Passed: native live Ward closure; all twelve real-map bridge checks; 1315 integrated GPU cache checks; 125076-token retrieval; prefix reuse; speculative acceptance/rejection; eight-slot decoding/cancellation; authenticated API text, forced tool arguments, high reasoning and a fresh image through the retained wrapper and matched official vision encoder. Default mode returned identical output token IDs to the original control on six serial fixtures.

One model accuracy case remains failed in both control and LSA: the correct sum-of-squares answer is 208, and both returned 188. This is retained as a quality failure, not relabelled passing. No new quality failure appeared in the finite tested fixtures; this does not establish general quality equivalence.

In this bounded comparison, 125076-token request wall time was 16.91 s control versus 17.78 s LSA. The four-window eight-slot smoke measured 414.6 versus 296.5 rows/s. These are finite smoke measurements, not a broad performance guarantee. The conditional requirement that a candidate prove better has not been met, so no production settings or recovery pins were changed.

## Implementation

- Strict versioned native map export/loading, immutable device upload and a full loaded-artifact cache identity.
- Native CUDA 512-token Ward closure (pool 4, 128-to-64 clusters,16-member cap), matching the pinned upstream merge partitions in all strict tests.
- Full FP32 projected-key histories preserve accepted prefixes across rejected proposals. Replacements reclose their blocks before causal visibility; no projected tail ring can overwrite accepted history.
- LSA selection is wired into ordinary decode, batched prompt prefill and speculative/multiple-slot verification. Actual selected widths feed the existing native INT8 attention.
- Conversation prefix copies include derived LSA data; authoritative KV restores rebuild it. Saved-session identity includes all map bytes and selector parameters.
- Batched prefill, parallel ordered expansion, shared projection inputs and batched verifier attention avoid the initial per-token bottlenecks.

The feature is CUDA-only and explicit. The root CMake option defaults OFF; in a feature-enabled build the original route remains active unless `STRATA_LSA_MAPS` is set. An explicit incompatible setup refuses startup. No Python/Triton callback or allocation occurs inside native decode graph replay. Full projected history is a correctness-first memory choice: the eight-slot engine also has a main prompt session, so its LSA state adds about 4 GiB in this configuration. Expert and slot capacity were verified rather than silently reduced.

## Build and isolated reproduction

The original research, maps and training evidence remain unchanged at `/path/to/lsa-prep-20261103`. That experiment's 09:16 EST handoff and consumed pilot test set are preserved. New whole-model fixtures are independent synthetic data. `INTEGRATION-AUDIT.md` is the earlier static planning record; its then-pending implementation list is superseded by this result.

```sh
cmake -S strata-src -B build-engine -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_LSA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/operator/model-tests/strata-20261005/Strata/build/_deps/strata_llamacpp-src \
  -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_MMQ_TAB=OFF -DSTRATA_BUILD_DSV41=OFF
cmake --build build-engine --target strata lsa_runtime_test --parallel 4
```

Compile only with model servers stopped. Keep build parallelism at four, never above eight on this node. GPU scripts default to plans or require the exact current maintenance owner, inactive production, bounded deadlines and owned child cleanup. Do not rerun after the HOLD changes without reviewing current state.

`test_integrated_engine.py` runs native CLI control/candidate comparisons. `test_candidate_api.py` runs the retained wrapper in an authenticated private loopback process. Its isolated readiness-gate override is replaced by exact process/asset/capacity checks before any request; production gate state and readiness receipts are untouched. Its private credential file is removed on exit.

## Current operational state and restoration

Production service `qwen38-flash-next-strata.service` is inactive/dead and remains enabled. Owned HOLD: `example-lsa-maintenance-owner`, entered at 15:33 EST. The watchdog must honor this intentional pause. Sunshine's user service is inactive/dead and remains enabled for startup; do not restart it unless requested. All test model/vision processes are stopped.

The retained serving/recovery hashes are in `receipts/pause-baseline.json`; the older capture-plan hash is not the restoration contract. When Jeff requests production restoration, confirm the research GPU window is free and use the current installed owned helper:

```sh
sudo -n python3 /usr/local/libexec/onyx-inference-recovery/strata-maintenance.py resume \
  --owner example-lsa-maintenance-owner --start
/usr/local/sbin/qwen38-modectl status
```

This restores the qualified baseline, preserving retry history. Do not delete HOLD manually, start retired inference units or promote this slower candidate by editing production configuration. No automatic restore, scheduled research run or fleet-route change was created.
