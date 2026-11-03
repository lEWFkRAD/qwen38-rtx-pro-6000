# Source compatibility assessment

Pinned LSA revision: `906be195c1052aaa74f95852ad1917ce1926ba64`.

## RTX PRO 6000 / compute 12.0

`src/lsa/core/kernels/dispatch.py:39-59,73-85` limits the CuTeDSL allowlist to tested sm90/sm103 devices. `tests/unit/core/test_dispatch.py:8-22` explicitly uses `(12,0)` and checks that every CuTeDSL step declines it. Kernel front doors select the Triton implementation. The experiment must request `kernels="triton"` explicitly; no dispatch patch is needed for the initial trial. Real kernel compilation and execution remain untested.

The installed PyTorch 2.13.0+cu129 wheel reports compiled architectures `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, read from binary metadata with CUDA hidden. This supports proceeding to an eventual hardware test; it is not that test.

## Full serving

`src/lsa/sglang/models/qwen3_5/backend.py` (locate `flash_reads`) selects FlashAttention reads on Hopper and Triton sparse reads otherwise. This is a possible integration path, not evidence that Flash-Next, Strata, quantized PLE weights, native EAGLE/MTP or their combined serving behavior work with LSA. Released distilled maps/configurations target Qwen3.8-27B and GLM-5.3-Flash.

`src/lsa/evaluation/gpu.py:12-15,30-34` only recognizes H200 and B300. The direct harness must record the actual device and must not fake the device name to bypass that restriction.

The pinned B300 serving image targets sm103 and architecture lists `10.0a;10.3a`. Its cu130 environment differs from this kit's upstream cu129 lock. No serving image is installed by this kit. Pinned supporting files are in `sources/`.

## Correctness limits

The selected upstream CPU suite produced 144 passes and three failures. `UPSTREAM-FAILURES.md` and retained receipts explain the material Ward near-copy failure and the attention bitwise-comparison failure. They remain failures of the published baseline. FP64 diagnostic agreement is not a silent replacement oracle or a validated repair.

Before performance interpretation, the GPU qualification should investigate small normal/near-tied inputs, causal boundaries, padding, invalid/future/duplicate selections, and incomplete clustering. Kernel parity or synthetic index timings do not measure semantic recall, retained model quality, or end-to-end speedup.
