# LSA preparation kit — RTX PRO 6000

Prepared by codex for Jeff on 2026-11-03. Scope: prepare an isolated synthetic Latent Sparse Attention kernel experiment while awaiting an attended GPU window.

Local kit: `C:/Users/OPERATOR/output/lsa-prep-20261103`
Onyx kit: `/path/to/lsa-prep-20261103`

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
cd /path/to/lsa-prep-20261103
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
cd /path/to/lsa-prep-20261103
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
