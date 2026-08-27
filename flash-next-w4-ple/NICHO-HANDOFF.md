# Independent contributor handoff

This is a portable handoff for Nicho—or any independent reviewer working from
their own GitHub and Codex accounts. No access to the production machine,
private network, or operator credentials is required or expected.

## Mission

Review and improve the public Qwen3.8-Flash-Next W4 PLE implementation without
touching a live deployment. Work in your own fork and branch, then open a pull
request or a findings issue against
[`lEWFkRAD/qwen38-rtx-pro-6000`](https://github.com/lEWFkRAD/qwen38-rtx-pro-6000).

The highest-value review areas are:

1. signed-INT4 packing and FP8 E4M3 group-scale correctness;
2. strict source/artifact binding and fail-closed corruption handling;
3. pinned-host-memory and CUDA-UVA gather safety;
4. time-of-check/time-of-use and partial-file failure modes;
5. portability beyond the exact pinned SM120, TP=1 runtime;
6. conversion of the script-style probes into repeatable automated tests; and
7. an upstream-ready patch structure for SGLang.

## Exact public baseline

| Component | Qualified value |
| --- | --- |
| Source checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Source revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Main-model quantization | ModelOpt NVFP4 W4A4 routed experts |
| Original PLE | FP8 E4M3, 51,200,245,760 bytes |
| Sidecar PLE | signed INT4, group 16, FP8 E4M3 scales, 28,800,138,240 bytes |
| Runtime output | BF16 gathered PLE rows |
| KV cache / Mamba state | BF16 / FP32 |
| Qualified topology | one SM120 GPU, TP=1, 131,072-token request context |
| SGLang image | `lmsysorg/sglang@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae` |
| SGLang Qwen4-Exp PR head | `73a255206f916366c8d26d4022f82ddfb0ab558d` |
| Sidecar artifact | `Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE` |
| Sidecar Hub revision | `8bf4dd3779b15732b303c0931e64961a332a0c78` |

The sidecar is lossy relative to the source FP8 PLE. “Bit-exact” in this
repository means parity between the CPU builder and CUDA runtime for the same
W4 representation, not lossless equivalence to FP8.

### Current production comparison point

The public baseline above remains the artifact and source-review contract. The
live deployment later added native EAGLE 1-1-2 speculative decoding with
rejection sampling, BS3 to the existing decode graph buckets, and
`mem-fraction-static=0.955`. Its comparable post-promotion greedy medians are
133.545/233.049/389.299 aggregate tok/s at c1/c2/c4. Reviewers should use the
exact flags and receipts in
[`docs/MTP-EAGLE-PRODUCTION-ADDENDUM-2026-08-27.md`](docs/MTP-EAGLE-PRODUCTION-ADDENDUM-2026-08-27.md)
for new performance comparisons. This does not broaden the no-production-access
boundary below.

## Start here

```bash
git clone --branch Cloud1 https://github.com/lEWFkRAD/qwen38-rtx-pro-6000.git
cd qwen38-rtx-pro-6000/flash-next-w4-ple
sha256sum -c SHA256SUMS
git switch -c nicho/independent-review
```

Read these files first:

- [`scripts/build_ple_w4_sidecar.py`](scripts/build_ple_w4_sidecar.py)
- [`src/qwen4_ple_w4_sidecar.py`](src/qwen4_ple_w4_sidecar.py)
- [`patches/0001-qwen4-exp-w4-ple.patch`](patches/0001-qwen4-exp-w4-ple.patch)
- [`patches/0002-qwen4-exp-sidecar.patch`](patches/0002-qwen4-exp-sidecar.patch)
- [`patches/0003-qsa-sm120-xqa.patch`](patches/0003-qsa-sm120-xqa.patch)
- [`tests/`](tests/)
- [`scripts/serve.example.sh`](scripts/serve.example.sh)
- [`README.md`](README.md)

Static checks that do not require the model or a GPU:

```bash
python -m py_compile scripts/build_ple_w4_sidecar.py \
  scripts/build_sidecar_overlay.py src/qwen4_ple_w4_sidecar.py
bash -n scripts/apply_patches.sh
bash -n scripts/serve.example.sh
```

The 28.8 GB artifact is available at the immutable Hub revision linked above,
but downloading it is unnecessary for a source review. GPU/runtime claims
should not be generalized beyond the pinned configuration without new tests.

## Safety and coordination boundary

- Do not connect to, probe, restart, benchmark, or modify any live deployment.
- Do not request or commit credentials, host inventories, private logs, or
  internal paths.
- Do not alter production routing or start competing inference workloads.
- Keep every change in your fork/branch until reviewed.
- Separate verified findings from hypotheses and include a minimal reproducer
  for each correctness claim.
- If a proposed change affects artifact bytes, require a newly built sidecar,
  new manifest and hashes, CPU/CUDA parity, and the full quality gate.

## Copy-paste prompt for a fresh Codex task

```text
You are an independent reviewer of the public repository
https://github.com/lEWFkRAD/qwen38-rtx-pro-6000 on branch Cloud1.

Work only in your own fork and a new branch. Focus on flash-next-w4-ple. First
verify its SHA256SUMS, read NICHO-HANDOFF.md and README.md, then audit the
builder, sidecar loader, three SGLang patches, tests, and serving example.

Prioritize: signed-INT4 group-16/E4M3-scale correctness; corrupt/truncated or
mismatched artifact rejection; source and revision binding; TOCTOU risks;
pinned-memory/UVA gather safety; portability assumptions; and missing tests.
Do not access or mutate any live service, private host, routing configuration,
or credentials. Do not run large downloads or GPU workloads unless I
explicitly authorize them. Make small reviewable commits. Report each finding
with severity, exact file/line, evidence, and a proposed test or fix. Open a PR
only after local static checks pass, and clearly label anything not exercised
on the exact SM120/TP=1 configuration.
```

## Expected hand-back

A useful contribution is either:

- a focused pull request with tests and no unrelated formatting churn; or
- a GitHub issue containing ranked findings, exact references, and suggested
  acceptance criteria.

The original operator remains responsible for live integration, qualification,
and release of any new binary artifact.
