# Qwen3.8-Flash-Next signed-W4 PLE sidecar

This directory contains the downstream code, patches, validation receipts, and
field report used to run
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
on one 96 GiB-class RTX PRO 6000 Blackwell workstation with 64 GB of system
RAM.

The key change is a persistent, host-resident signed-INT4 sidecar for the
model's PLE (n-gram embedding) table. SGLang gathers only selected rows through
CUDA UVA and dequantizes them to BF16 on the GPU. The sidecar does not contain
the model and cannot be used by itself.

> [!WARNING]
> This is experimental downstream code, not a released SGLang feature. It is
> pinned to an open Qwen4-Exp SGLang PR and was qualified only with TP=1 on an
> SM120 RTX PRO 6000. Keep the model and artifacts read-only, verify every
> digest, and leave sidecar fallback disabled.

## What fits

The base checkpoint already uses ModelOpt NVFP4 for routed experts, but its PLE
table remains FP8 E4M3. The PLE has 320,001,536 rows, 160 values per row, and
128 synchronized checkpoint shards.

| Representation | Decimal GB | GiB | Change |
| --- | ---: | ---: | ---: |
| Original FP8 PLE | 51.200 | 47.684 | baseline |
| `qweight.u8` | 25.600 | 23.842 | packed signed INT4 |
| `block_scales.f8` | 3.200 | 2.980 | one FP8 scale per 16 values |
| Complete sidecar | 28.800 | 26.822 | **43.75% smaller** |

Each group of 16 FP8 values becomes eight packed bytes and one FP8 E4M3 block
scale. The quantizer uses:

```text
scale = max(max_positive / 7, max_negative / 8).clamp_min(2**-9)
scale = cast_to_float8_e4m3fn(scale)
q = round_to_nearest_even(value / scale).clamp(-8, 7)
```

Even dimensions occupy the low nibble, odd dimensions the high nibble, using
four-bit two's-complement values. The checkpoint's outer BF16 PLE scale remains
exactly `0.00019931793212890625`.

This quantization is lossy relative to the FP8 source. "Bit-exact" in the test
receipts means that the CPU builder and CUDA runtime produce identical packed
bytes and dequantized BF16 values for the same W4 representation.

## Qualified artifact

The 28.8 GB companion table is published at
[`Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE`](https://huggingface.co/Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE).
The complete, remotely verified Hub revision with final license attribution is
[`8bf4dd3779b15732b303c0931e64961a332a0c78`](https://huggingface.co/Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE/tree/8bf4dd3779b15732b303c0931e64961a332a0c78).
The exact artifact contract is:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `qweight.u8` | 25,600,122,880 | `7e18b8dac400bda73b24a2dd135ca009163972031c7b83095ef1816650df3297` |
| `block_scales.f8` | 3,200,015,360 | `667736278723db00dc70a7afc37aaf9131deab6e9345047f467035add7e09a1c` |
| `manifest.json` | 13,451 | `a11028a945bac40c7a2d5f41f21c829a08f5d531c39184dda2a7c8731d7e1691` |

Do not substitute a similarly named artifact: the runtime binds the manifest,
source revision, model index, config, Hugging Face tree receipt, and raw PLE
digest. You may also regenerate the exact sidecar with the sealed builder below.

## Reproducibility pins

- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4`
- Model revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`
- Model index SHA-256: `da5ca9c3b65e48e151329e64e141c2fa700bf2f99aec53cc014e4b52a6ff7a84`
- Model config SHA-256: `e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624`
- Hugging Face tree receipt SHA-256: `f84acd65b08e4de8f9f1698b85136655f24ef04f9d8b2e739f102ff47c9fa572`
- Raw FP8 PLE SHA-256: `b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3`
- SGLang Qwen4-Exp PR #36497 head: `73a255206f916366c8d26d4022f82ddfb0ab558d`
- Container image: `lmsysorg/sglang@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae`
- Contained SGLang commit: `d91c3682b0b429e4c70df63cd57f819588ce29b0`
- FlashInfer: `0.6.17`
- PyTorch: `2.13.0+cu130`
- Triton: `3.7.1`

The SGLang PR and the SM120 QSA fix are open upstream work as of this report;
pinning these exact revisions matters.

## Reproduce the runtime overlay

Clone SGLang and fetch the exact Qwen4-Exp PR head:

```bash
git clone https://github.com/sgl-project/sglang.git
git -C sglang fetch origin pull/36497/head:qwen4-exp-pr
git -C sglang checkout 73a255206f916366c8d26d4022f82ddfb0ab558d
./flash-next-w4-ple/scripts/apply_patches.sh ./sglang
```

`apply_patches.sh` refuses any unexpected source or release-input hash. It
builds and compiles all three final files in temporary staging before changing
the checkout, then installs them with verified rollback on any handled failure.
Rollback steps are independent and best-effort. If any rollback operation or
verification fails, the installer retains the pristine source copies and a
`RECOVERY.txt` receipt in a `.qwen4-ple-recovery-*` directory inside the SGLang
checkout; do not rerun or start SGLang until those instructions are completed.
The staged changes are, in order:

1. signed-W4/group-16 PLE storage and Triton UVA gather;
2. strict persistent-sidecar loading and source binding;
3. SM120 routing to FlashInfer XQA for Qwen sparse-attention decode.

It then verifies the exact final source hashes and installs
`qwen4_ple_w4_sidecar.py` into the checkout.

## Build the sidecar yourself

Download the exact checkpoint with `hf download --revision ... --local-dir ...`.
A stock local-dir download may not create the aggregate tree receipt required
by the sealed builder, so install the published receipt before building. The
checkpoint is roughly 135.6 GB and the sidecar is another 28.8 GB; allow
generous temporary and free disk space.

```bash
export MODEL_DIR=/absolute/path/to/Qwen3.8-Flash-Next-NVFP4
export OUTPUT_DIR=/absolute/path/to/empty/ple-w4-g16-7b719225

install -Dm0644 \
  flash-next-w4-ple/provenance/hf-tree-7b719225242aacd3dbd3f9407468c2ee9a9d2594.json \
  "$MODEL_DIR/.cache/huggingface/trees/7b719225242aacd3dbd3f9407468c2ee9a9d2594.json"
echo 'f84acd65b08e4de8f9f1698b85136655f24ef04f9d8b2e739f102ff47c9fa572  '"$MODEL_DIR"'/.cache/huggingface/trees/7b719225242aacd3dbd3f9407468c2ee9a9d2594.json' \
  | sha256sum -c -

mkdir -p "$OUTPUT_DIR"
test -z "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)"

docker run --rm --entrypoint python3 \
  --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/release/src:/release/scripts \
  -v "$PWD/flash-next-w4-ple:/release:ro" \
  -v "$MODEL_DIR:/models/flash-next:ro" \
  -v "$OUTPUT_DIR:/output" \
  lmsysorg/sglang@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae \
  /release/scripts/build_ple_w4_sidecar.py \
  --model-dir /models/flash-next \
  --output-dir /output \
  --source-revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --chunk-rows 65536 \
  --threads 2
```

The qualified build took 550.112 seconds. The builder verifies all ten source
PLE LFS objects, hashes every raw shard while converting it, checks that source
files did not change during the build, writes atomically, and seals the output
manifest.

Every changed or fine-tuned checkpoint requires a newly built sidecar. Never
reuse a sidecar merely because tensor shapes match.

## Run the qualified profile

Set `MODEL_DIR`, `SIDECAR_DIR`, and `SGLANG_CHECKOUT`, then run:

```bash
./flash-next-w4-ple/scripts/serve.example.sh
```

The example verifies the aggregate tree receipt and all ten source PLE LFS
objects, hashes the sidecar, checks GPU/host-memory/swap headroom, refuses a
busy GPU or port, and binds only to `127.0.0.1:8002`. It does not configure
SGLang's available API-key option. If remote access is needed, set an API key
and keep the endpoint behind an authenticated VPN or reverse proxy; do not
expose port 8002 directly to the Internet.

The example names the container `qwen38-flash-next-w4-ple-example`, keeps it
after exit for log/inspect forensics, and bounds Docker's local logs. Remove the
stopped container explicitly before a later rerun only after inspecting any
failure.

The qualified serving profile uses:

- TP=1;
- ModelOpt NVFP4 routed experts;
- signed-W4 host PLE with FP8 group scales and strict no-fallback validation;
- 131,072-token per-request context;
- a 163,072-token requested shared pool (160,832 actually allocated in the
  captured run);
- at most four running and sixteen queued requests;
- BF16 KV cache;
- 1,024-token chunked prefill;
- FlashInfer GDN prefill, Triton GDN decode, and FP32 Mamba state;
- FlashInfer XQA for QSA decode on SM120;
- full decode CUDA graphs for batch sizes 1, 2, and 4;
- prefill graphs, radix cache, mixed chunking, dynamic chunking, and automatic
  truncation disabled.

Do not combine `--ple-offload-embedding` with generic SGLang CPU/layer offload:
the latter can stage the pinned PLE table back to the GPU.

## Validation summary

The exact receipts are under [`evidence/`](evidence/), and the full deployment
story is in the
[`field report`](docs/Qwen3.8-Flash-Next-W4-PLE-RTX-PRO-6000-Field-Report.pdf).

| Gate | Qualified result |
| --- | --- |
| Sidecar public quality receipt | 11 / 11 passed |
| Broader W4 qualification gate | 13 / 13 passed; 36 / 36 sustained requests |
| Concurrent exact outputs | 4 / 4 passed |
| Long retrieval | 119.5K and 130.5K prompt cases passed |
| Boundary handling | Requests beyond 131,072 rejected |
| Warm decode, concurrency 1 | about 81 tok/s |
| Warm decode, concurrency 2 aggregate | about 144 tok/s |
| Warm decode, concurrency 4 aggregate | about 258-261 tok/s |
| GPU residency | about 81.29 GB loaded; about 9.8 GiB free at runtime |
| Host cgroup peak during qualification | 52.257 GiB RAM plus 1.404 GiB swap |
| Prebuilt sidecar read | about 16.1-16.4 seconds |
| Production cold start to readiness | about 203 seconds |

The roughly 320 tok/s figure observed during an earlier 8K-pool experiment is
not comparable to this 131K-context production profile. Mixed long/short
prefill fairness also remains an upstream scheduler limitation.

## Run the validation scripts

The files under `tests/` are executable validation scripts, not a pytest suite.
Do not run `pytest tests` and mistake "no tests collected" for success.

Syntax and the CPU-only synthetic corruption/round-trip gate:

```bash
bash -n flash-next-w4-ple/scripts/apply_patches.sh
bash -n flash-next-w4-ple/scripts/serve.example.sh
python3 -m py_compile \
  flash-next-w4-ple/src/qwen4_ple_w4_sidecar.py \
  flash-next-w4-ple/scripts/*.py \
  flash-next-w4-ple/tests/*.py

docker run --rm --entrypoint python3 \
  -v "$PWD/flash-next-w4-ple:/release:ro" \
  lmsysorg/sglang@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae \
  /release/tests/test_ple_w4_sidecar.py

python3 flash-next-w4-ple/tests/test_apply_patches_transaction.py \
  --sglang-repo /path/to/pristine/sglang
```

The CPU sidecar command must print `PLE_W4_SIDECAR_TESTS_OK`; the installer
test must print `PATCH_TRANSACTION_TESTS_OK`. The latter creates and removes a
temporary linked worktree at the pinned SGLang commit, checks a fresh
`core.autocrlf=true` clone, rejects a tampered release input, injects failures
after every install stage and rollback operation, verifies retained recovery
material, and verifies an exact successful install. On Windows, the harness
sets `core.longpaths=true` and `core.autocrlf=false` command-locally while
creating its temporary SGLang worktree. A default long-path-disabled or CRLF
SGLang checkout otherwise refuses safely at checkout or the pinned hash gate.
The source repository's configuration is not changed. The CUDA probes
`probe_cpu_cuda_parity.py`, `probe_model_class_parity.py`, and
`test_overlay.py` must be run one at a time on an isolated SM120 GPU in the
exact image, with the pristine model mounted at `/models/flash-next`, the
release mounted at
`/release`, `PYTHONPATH=/release/src:/release/scripts`, and the patched SGLang
files mounted at the same container paths used by `serve.example.sh`. Their
success markers are respectively `PLE_W4_CPU_CUDA_PARITY_OK`,
`PLE_W4_ACTUAL_CLASS_PARITY_OK`, and `OVERLAY_CLASS_OK True`.

## Repository map

- `patches/`: the three pinned downstream patches;
- `src/`: strict sidecar manifest/streaming loader;
- `scripts/`: sidecar builder, overlay reproducer, patch installer, and safe
  serving example;
- `tests/`: corruption tests and CPU/CUDA/class parity probes;
- `evidence/`: sanitized benchmark and long-context receipts;
- `provenance/`: the exact public HF tree receipt required by the sealed
  builder;
- `docs/`: the publication-ready field report;
- `LICENSES/` and `NOTICE.md`: governing notices and scope.

## Licenses and attribution

The original downstream files are licensed under Apache License 2.0 as stated
in [`LICENSE.md`](LICENSE.md). The generated model sidecar is a derivative of
Qwen model weights and is distributed subject to the Qwen Community License
1.0. The SGLang-derived patches retain Apache License 2.0. Read
[`NOTICE.md`](NOTICE.md) and both files under [`LICENSES/`](LICENSES/) before
redistribution or commercial use.
Certain commercial Model-as-a-Service and AI Work Assistant uses require a
separate Qwen license.

This is an independent community implementation and is not an official release
of Alibaba Qwen, SGLang, NVIDIA, RadixArk, or FlashInfer.

## Upstream references

- [Qwen3.8-Flash-Next architecture](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [RadixArk NVFP4 checkpoint](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
- [SGLang Qwen4-Exp PR #36497](https://github.com/sgl-project/sglang/pull/36497)
- [SGLang SM120 QSA fix PR #36556](https://github.com/sgl-project/sglang/pull/36556)
- [FlashInfer 0.6.17](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.17)
- [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
