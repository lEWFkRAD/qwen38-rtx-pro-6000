# LSA preparation kit — RTX PRO 6000

Prepared by codex for Jeff on 2026-11-03. Scope: prepare an isolated synthetic Latent Sparse Attention kernel experiment while awaiting an attended GPU window.

Local kit: `C:/Users/OPERATOR/output/lsa-prep-20261103`
Onyx kit: `/path/to/operator/research/lsa-prep-20261103`

## Current status

Source and dependencies are pinned and installed. The kit passed 32 CPU regression tests, its reference smoke, and a real CPU child-process timeout check. The selected upstream CPU suite has 144 passes and three retained failures; see `UPSTREAM-FAILURES.md` before the GPU window. See `PREP-STATUS.json` and `receipts/` for completed checks and any unresolved items. No GPU qualification result exists until a later explicitly executed run writes a successful receipt. CPU checks cannot prove CUDA compatibility or performance.

## What the first experiment establishes

1. The released Triton indexer compiles and works on RTX PRO 6000 compute capability 12.0.
2. Small deterministic cases agree with the upstream PyTorch reference, subject to the upstream treatment of nearly tied scores.
3. Bounded synthetic workloads at 131072, 262144 and 1048576 tokens measure index construction and query selection with warm timings separated from compilation and warmup.
4. Results record actual GPU, dependencies, source identity, shapes, resource limits and failures.

These are indexer results. They do not establish model response quality, whole-model time-to-first-token, full-context prefill throughput, or Flash-Next/Strata compatibility. A small query batch is not a full prefill. Random normalized vectors are not learned model attention states. Full-model comparison remains a separate decision after this experiment.

## Sources and environment

- LSA commit: `906be195c1052aaa74f95852ad1917ce1926ba64` (Apache-2.0).
- Serving fork reference only: `042d955ba31b962911f63b114c2fb245effa86f2`, tag `lsa-v0.1.0`.
- Python 3.12; uv 0.13.0; dependency resolution from the unchanged upstream `uv.lock`.
- The local experiment uses the upstream cu129 dependency lock. The serving image uses cu130: these are distinct environments, and kernel results here do not qualify that image.
- `sources/` preserves pinned serving-fork notes and B300 image checks for inspection. The B300 image check is not our runner.
- `COMPATIBILITY.md` records source-level support and limitations. `sources.lock.json` records provenance and the upstream lock hash.

The test uses a dedicated `.venv`, kit-owned caches and receipt directory. No model weights, staff prompts, SGLang service, privileged install, driver change, fleet route, or startup task is needed. No training/evaluation extras are installed for this first experiment.

## Preparation and CPU verification

`prepare_env.sh` is restricted to the Onyx kit path. It hides CUDA, limits compute threads to two, fetches the exact source commit without rewriting an existing checkout, installs the locked environment, records versions and runs selected upstream CPU tests. Its external invocation is bounded to 30 minutes at nice 15 / idle I/O priority. It never starts a GPU test.

After staging, CPU-only custom checks use the kit interpreter with `CUDA_VISIBLE_DEVICES=''`, `NVIDIA_VISIBLE_DEVICES=void`, and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2`:

```sh
cd /path/to/operator/research/lsa-prep-20261103
.venv/bin/python -m unittest -v test_benchmark_lsa test_run_window
.venv/bin/python benchmark_lsa.py --mode cpu --output receipts/harness-cpu.json
.venv/bin/python run_window.py --plan
sha256sum -c SHA256SUMS
```

Consult each command's `--help` for the exact supported arguments. The dry plan is the default and submits no workload.

## At the attended GPU window

The window owner first coordinates existing jobs and the current inference-recovery procedure. Historical TensorFold/SGLang restoration scripts are not a valid rollback. The launcher itself never stops Strata, changes a recovery HOLD, acquires production mutation locks, or restarts a service. Maintenance ownership and restoration remain with that separate attended operation.

The launcher requires the current Strata service to be inactive, a supported device, sufficient free VRAM, and repeated low GPU-utilization readings. This creates an uncontended measurement window; it is not authorization to interrupt staff inference. A failed preflight returns a reason and starts no benchmark. It does not wait in a loop or auto-retry later.

```sh
cd /path/to/operator/research/lsa-prep-20261103
.venv/bin/python run_window.py --check-window
# Only after the window has been explicitly opened and the read-only check passes:
.venv/bin/python run_window.py --execute --window-label "attended LSA diagnostic"
```

The default diagnostic first reproduces the known near-copy reference cases and stops before performance timing if they remain unresolved. The timing code is staged for a later separately reviewed repair; this kit does not bypass the failing baseline. Start with the default capped workload. Inspect every correctness result before interpreting timings. A compilation failure, non-finite value, invalid selection, memory failure or deadline is a failed qualification, not a reason to relabel a reference fallback as a passing GPU run. Stop at the first material failure and retain the receipt.

During execution the launcher rechecks the primary service every few seconds. A restart, unreadable state, timeout or interruption stops only its own child process group and records the outcome. Normal completion records the worker result. The PyTorch allocation limit excludes CUDA context/compiler/library overhead; the free-memory reserve is additional protection, not a hard total-VRAM cap. The window owner restores the existing production configuration through its current maintenance procedure and verifies normal operation. The kit provides no production stop/restore commands.

## Decision after the experiment

- If correctness or compilation fails: preserve the failure, identify a bounded compatibility fix, and rerun only the affected qualification after review.
- If correct but indexing cost is unhelpful at the relevant context: keep the research result; do not infer a production benefit from the paper's 10M-token figures.
- If correct and promising: design a supported-model A/B comparison with identical weights, context, precision and decoding settings. Measure whole-model latency, memory and task quality. Flash-Next would need its own maps, integration and validation; the released 27B maps cannot be assumed to fit it.

No scheduled work, automatic GPU polling, service replacement, or deployment is part of this kit.

Receipt outputs are exclusive-create where noted: use a fresh --output filename when rerunning the harness. The first bootstrap used pip's ordinary download cache; the final script directs subsequent pip and compiler caches into this kit. Package installation stayed inside the isolated environment.

## [2026-11-03 07:55] Completed attended LSA GPU trial | codex

Jeff authorized production pause, persistent direct LSA testing, configuration adjustments, and promotion only if the candidate worked better with the required behavior. Production stayed stopped under owned HOLD codex-lsa-20261103 for the full test window; no fallback suite preceded the actual CUDA trial.

- GPU Triton index construction, selection and expansion passed at 131072, 262144 and 1048576 tokens, with 1 and 64 query rows. Warm median construction times: 3.60, 7.25 and 28.37 ms. These are synthetic indexer timings, not full prefill, model latency or speedup measurements.
- Original GPU kernels compiled and passed exact IID labels. Original CPU near-copy reference stalled while GPU completed. A separate pinned research commit 35b1bf9a10703b57da3eb073debfad6a66507269 symmetrizes the CPU Gram matrix; original source and failed receipts are retained. GPU kernels were not patched.
- Near-tie labels differed after repair. Four fixtures now require completion, valid canonical partition/caps, repeatability and upstream-style resultant-objective agreement rather than exact labels. Maximum observed objective relative error was 1.99e-8, under the 1e-5 criterion. Exact IID parity remains required.
- Actual gathered sparse-attention forward and Q/K/V gradient tests passed for one-head and eight-over-one GQA; all ten CPU Ward regressions passed: 12 passed, zero failed/skipped.
- Integrated synthetic projection -> GPU clustering -> selection -> expansion -> sparse attention passed at 4096/one-head, 4103/one-head (seven-key incomplete block), and 4096/four-query-head GQA. Output/gradient relative errors were about 0.21-0.25%, below upstream 1%; log-sum-exp maximum error 1.43e-6. Uses synthetic maps and fixed selections, not learned-model quality validation.
- NOT PROMOTED: Flash-Next already has native Strata QSA. Its corresponding map input is384 features versus768 for published Qwen27B maps, with12 versus16 attention layers. Native selector integration and speculative accepted/rejected-token state handling are missing; released SGLang path also requires BF16KV and refuses speculation. There is no configuration-only compatible candidate and no model-level better-than-production evidence.
- After all tests completed, resumed the original Strata configuration through installed owned maintenance helper. Config hash unchanged f3d1b869d3d36733fa58686b73537092733bc52b9c9c06ba7d2789bfb17c16a4; service active/running PID2796678,NRestarts0. Managed readiness and exact inference passed; later foreground counters advanced. Recovery HOLD released by the helper. No staff route, model weight, KV precision, slot count or engine change promoted.

Local evidence: C:/Users/OPERATOR/output/lsa-prep-20261103/GPU-RESULTS.json and results/. Matching experiment root on Onyx: /path/to/operator/research/lsa-prep-20261103. Earlier preparation-only status above is historical.

## [2026-11-03 09:13] Flash-Next LSA maps completed; integration deferred until after work | codex
|- Jeff requested the maps. Trained all12 Flash-Next QSA layer pairs384->64 on actual retained IQ3_S/control-vector/INT8-KV activations, with four training documents and separate two-document validation and test splits;36864tokens total. All96 native attention captures replayed within2.90e-6 scale-relative error.
|- Learned maps passed the declared pilot criteria against PCA: test KL0.49265->0.30981 (37.1% reduction), conditional long-range mass70.50->72.17% at512 keys and85.85->86.90% at1024. This is4-6K pilot evidence, not production speed/quality superiority. Worst test layer/document mass regression1.221percentage points; validation2.825points.
|- All12 learned maps passed real CUDA clustering/selection/expansion; Flash-Next24Q/2KV/256D gathered attention passed at layers3,27,47 with maximum relative output error0.02252%. First harness LSE-layout indexing failure retained and corrected. No map or GPU kernel changed for that correction.
|- Map exports and per-layer temperature sidecars roundtrip exactly. Reusable dataset copies repair unused dense-target padding; original native training targets unchanged. Whole-document split overrides original train_queries metadata; do not claim its last training row is held out. Test set consumed; preserve for reporting, not tuning.
|- Map test window completed under HOLD codex-lsa-maps-20261103. Retained production resumed through installed owned maintenance helper: active/running PID2876339,NRestarts0,healthy,exact inference/process/readiness verified,HOLDnull. Config unchanged f3d1b869d3d36733fa58686b73537092733bc52b9c9c06ba7d2789bfb17c16a4. No promotion: native Strata LSA selector/MTP-state integration and end-to-end comparison remain.
|- Jeff then instructed: We will wait until after work. Native integration and additional testing deferred; no scheduled run/reminder. Conditional promotion authorization remains; no permission reconfirmation needed solely for that existing authorization.
|- Deliverables: C:/Users/OPERATOR/output/lsa-prep-20261103/flashnext-maps/README.md; pilot-v1/ contains learned/PCA maps,temperature sidecars,checkpoints and report. Matching Onyx root /path/to/operator/research/lsa-prep-20261103/flashnext-maps retains raw captures and datasets.
