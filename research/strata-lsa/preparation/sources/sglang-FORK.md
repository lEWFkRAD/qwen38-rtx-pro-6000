# SGLang fork for Latent Sparse Attention serving

This is a fork of [sgl-project/sglang](https://github.com/sgl-project/sglang).
It carries the serving changes that the Latent Sparse Attention (LSA) paper needed to serve long contexts, up to 10 million tokens.
The paper evaluates GLM-5.3-Flash and Qwen3.8-27B.

LSA itself is not in this fork.
The LSA indexer is an SGLang plugin, released in [Prefrontal-AI/latent-sparse-attention](https://github.com/Prefrontal-AI/latent-sparse-attention).
This fork is the server underneath it, and it also serves the paper's baselines (native sparse attention and dense attention).

## Base and tag

The fork sits on upstream commit `0e2aac500bbd7010503bff67bd221bfbd7f04389`.
Title: "[ROCm][Perf] aiter: page-level KV view for gfx950 fp8 page-64 asm prefill".
Commit date: 2026-09-25 19:23 -0700 (2026-09-26 02:23 UTC).

The fork has one tag, `lsa-v0.1.0`, one commit on top of that base.
It is the code the paper's results come from.
It is built once per GPU kind: the default build for H200, and the same build with `--layer docker/b300` for B300.
The images behind earlier paper results were older states of this same code.
On H200 the code paths are those of the paper's H200 image, with one exception.
The trtllm DSA prefill is now refused under DCP on any GPU, and H200 never selects it by default.

Every change relative to upstream is a diff against the base.
Compare against upstream like this.

```bash
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream
git log --oneline 0e2aac500b..lsa-v0.1.0   # the fork's commit
git diff --stat 0e2aac500b lsa-v0.1.0      # every fork change
git diff 0e2aac500b lsa-v0.1.0 -- python/sglang/srt/layers/attention/dsa_backend.py
```

Leaving out this file, the README note and the build script's generic defaults, the fork changes 119 files, with 7003 lines added and 502 removed.
The commit is a squash, so the diff is the whole record.
The authors of the cherry-picked upstream work are credited below.
Nothing here is meant to be merged upstream as it stands.

## What the fork adds

Measurements are on H200 GPUs, on one 8-GPU node unless a row says otherwise.
The Blackwell-only rows are on B300.
Each measurement is one example, not a full result table.

| Area | Change | Main files | Measurement |
|---|---|---|---|
| GLM-5.3-Flash context parallel | Prefill context parallel (contiguous spans) and decode context parallel (DCP) in one server, with the KV cache split by token across the node. MLA and KDA decode heads are split over the prefill-CP group, and EAGLE/MTP runs under DCP | `srt/layers/cp/`, `srt/layers/dcp/`, `srt/layers/communicator_mhc_hybrid_cp.py`, `srt/layers/communicator_dsa_cp.py`, `srt/models/glm5_next.py`, `srt/arg_groups/` | A 10.49M-token prompt has a first-token time of 545.7 s on one node, against 1336.6 s on TP8 x PP2 over two nodes. One node holds about 13.4M tokens of cache in this layout, against about 3.5M on TP8 |
| DSA under DCP | Prefill on the fast kernel. A prefix gather of only the rows that the selected lists name (`--dcp-prefill-sparse-gather`). A fused, split-KV sparse decode kernel that compacts the rows a rank owns. int64 offsets in the index-key gather | `srt/layers/attention/dsa/`, `kernels/ops/attention/dsa/`, `srt/layers/attention/dsa_backend.py`, `srt/models/deepseek_common/` | Native GLM at 10M, first-token time 548.1 s to 519.0 s with the prefix gather. Decode time per token at 10M, 19.11 ms to 17.86 ms with the split-KV kernel |
| DSA dense prefix | A rope-free MLA latent is prefilled in chunks past the chunk capacity (128k). | `srt/models/deepseek_common/attention_forward_methods/`, `srt/model_executor/forward_batch_deepseek_mha_mixin.py` | Dense prefill serves contexts of 256k and up |
| DSA index-key offsets | The index-key prefix gather computed its page offset in int32, which wraps past 16.27M slots in use. It is int64 now | `kernels/ops/attention/dsa/index_buf_accessor.py`, `srt/layers/attention/dsa/` | Native GLM under DCP8 serves past 16.2M index-key slots. The test gathers pages on both sides of the wrap bit-exactly |
| KDA (Kimi-Linear, GLM-5.3-Flash linear layers) | Prefill context parallel on contiguous spans. State pre-scan and merge, conv halo exchange, no per-layer host syncs, and the radix checkpoint taken at the end of the extend. The state merge uses a tf32x3 dot | `srt/layers/attention/linear/kda_cp.py`, `kernels/ops/attention/fla/chunk_delta_h_cp.py`, `srt/layers/attention/linear/kda_backend.py`, `srt/models/kimi_linear.py` | The merge went from about 4.8 ms to about 0.1 ms per KDA layer and chunk. Native GLM at 10M, first-token time 549.2 s to 532.4 s after the host syncs were removed |
| QSA (Qwen sparse attention) | QSA on the dense Qwen3.8-27B decoder layer. Prefill index scoring split over keys. Chunk prefill reads K and V from the pool in place. QSA under DCP. A 2 GiB cap on the prefill index logits | `srt/layers/attention/qsa/`, `srt/layers/attention/qwen_sparse_attn_backend.py`, `srt/models/qwen3_5.py`, `srt/mem_cache/qsa_kv_pool.py` | A 32k prefill chunk at a 2M position took 67.8 s before and 3.8 s after the key split (4 GPUs). A 10.7M-token prompt is served at TP8 + DCP8 |
| QSA decode fix | The QSA decode K/V scratch is zeroed. A stale infinity past the last row turned into NaN with CUDA graphs on, two or more decoding requests and a context of 2M or more | `srt/layers/attention/qsa/sparse_attn.py` | With 4 concurrent needle prompts at 2.0M and 5.6M context, correct answers went from 1 of 4 to 4 of 4 |
| GQA under DCP | FA3 prefill for dense Qwen3.8-27B (GQA) under DCP, with the prefix pass kept unsplit so prefill memory does not grow with context. A fix for the triton extend on ranks that hold several KV heads. Every other attention backend is refused | `srt/layers/attention/flashattention_dcp.py`, `srt/layers/attention/flashattention_backend.py`, `srt/layers/attention/triton_backend.py`, `srt/layers/attention/attention_registry.py` | A 131k prefill takes 3.4 s with FA3, against 7.2 s on TP8 with triton. Before the triton fix, a 1M needle prompt got the answer that the document held no number |
| Blackwell (B300, sm_103): DCP sparse decode | The split-KV sparse decode kernel tiles 32 heads at a time on Blackwell, where a 64-head tile of GLM-5.3-Flash's 512-wide latent faults with a misaligned address. Hopper keeps 64 | `kernels/ops/attention/dsa/dcp_sparse_decode.py` | The 64-head tile asked 262,688 B of shared memory at two stages, and a B300 has 232,448 B. The 32-head tile takes 102,400 B and matches the reference to 3e-4, at 19.8 us for 2 rows of 4,612 keys |
| Blackwell: trtllm under DCP | On Blackwell the sparse-attention backend with pooled index keys resolves to trtllm. Its prefill read the rank's own KV pool with global slot ids and skipped the DCP gather. The fork now refuses that path, with a message that names `--dsa-prefill-backend tilelang` | `srt/layers/attention/dsa/dsa_backend_kpool.py`, `srt/layers/attention/dsa_backend.py` | GLM-5.3-Flash CP8 died at 1.28M to 1.31M tokens with a per-rank pool of 1,315,000, before the refusal. H200 resolves to FA3, which is handled after the gather |
| Blackwell image | `docker/b300/`: a layer over the stock CUDA 13.0 image. It pins the JIT arch lists to `10.0a;10.3a`, installs `uv`, and runs an import and arch check at build time. `check_image.py` also runs one Triton, CuTeDSL, FlashInfer and sgl-kernel call when a GPU is present | `docker/b300/Dockerfile`, `docker/b300/check_image.py` | B300 timings with these changes match the earlier B300 records within 0.6%, for example GLM TP8 native at 1M tokens, first-token time 27.16 s before and 27.03 s after |
| Build | `docker/enroot/build.sh` builds any commit as a Docker image and imports it as an enroot squashfs. The dev-dependency install caps `python-dateutil` below 3 so pip resolves | `docker/enroot/build.sh`, `docker/Dockerfile` | |

The Blackwell-only paths are the three Blackwell rows above.
The FA3 backend is sm_90a only, so on B300 name the attention backends explicitly.
The image's FA3 refuses `--attention-backend fa3` there.

Two flags and one setting come with the fork.
`--dcp-prefill-sparse-gather` turns on the sparse prefix gather, and it needs `--dcp-size` above 1 and a DSA model.
`SGLANG_DCP_SPARSE_SPLIT_DECODE` (default on) selects the split-KV sparse decode kernel.
Under DCP, `--max-total-tokens` is per GPU, so a server-wide cap of 10.52M tokens on 8 GPUs is `--max-total-tokens 1315000`.

## Plugin extension points

The LSA plugin loads through upstream's own plugin mechanism.
It uses the `sglang.srt.plugins` entry point and upstream's hook registry, `sglang.srt.plugins.hook_registry`.
The plugin attaches to about 35 SGLang internals by dotted path.
The list of paths is in the plugin's source, in the LSA repository.

The fork keeps these internals in place, and a rebase that renames or moves one of them breaks the plugin at server start.
The plugin checks that every target it needs still resolves, and it refuses to start and names the ones that do not.

| Area | What the plugin hooks |
|---|---|
| DSA backend and indexer | The indexer class, the backend's metadata calls and its FA3 and FlashMLA sparse reads |
| GLM model forward and hyper-connection pipeline | The GLM forward, whose pipeline stage boundary ships a residual that means nothing under hyper-connections |
| MLA forward | Where the model asks whether a layer runs its indexer, and the method that returns the absorbed query and the key latent |
| FA3 and triton backends | Constructor, metadata calls and both forwards, for Qwen's full-attention layers |
| Pool and KV-cache sizing | The cell size, the memory profile and the pool config |
| CUDA-graph capture | The decode runner's capture loop and the capture-variant getter |
| DP-attention collectives | The all-gather and all-reduce that give each rank every head's query |

## Tests

The fork adds tests next to upstream's, under `test/registered/`.
They run with the same tools as upstream's tests, with the GPU count that each file registers.

| File | What it checks | GPUs |
|---|---|---|
| `test/registered/e2e/dcp/test_glm53_flash_dcp.py` | GLM-5.3-Flash under DCP on GSM8K | 8 |
| `test/registered/e2e/dcp/test_glm53_flash_cp_dcp.py` | The one-server prefill CP + DCP layout: GSM8K, needles at chunk and span boundaries of a 63k-token prompt, concurrent requests of different lengths | 8 |
| `test/registered/dcp/test_dcp_sparse_prefill_gather.py` | The DSA sparse prefix gather on 2 and 8 ranks | 8 |
| `test/registered/kernels/ops/attention/test_dcp_sparse_decode.py` | The split-KV sparse decode kernel, including the 512-wide latent case for Blackwell | 1 |
| `test/registered/dcp/test_dsa_trtllm_dcp_guard.py` | The refusal of the trtllm DSA prefill under DCP | 1 |
| `test/registered/kernels/ops/attention/test_dsa_dcp_kernels.py` | DSA DCP decode kernels | 1 |
| `test/registered/kernels/ops/attention/test_dsa_dcp_prefill_kernels.py` | DSA DCP prefill kernels | 1 |
| `test/registered/kernels/ops/attention/test_dsa_kpool_gather_prefix_int64.py` | The index-key gather on both sides of the int32 wrap | 1 |
| `test/registered/kernels/ops/qsa/test_qsa.py` | QSA kernels | 4 (registered for B200) |
| `test/registered/attention/test_kda_cp_state.py` | KDA context-parallel state pre-scan and merge | 1 |
| `test/registered/unit/test_kda_cp_backend_helpers.py` | KDA CP shard metadata and conv-halo helpers | 0 (CPU) |

The fork also extends a few of upstream's tests (`test/registered/unit/mem_cache/test_qsa_kv_pool.py`, `test/registered/unit/managers/test_mamba_checkpoint_depth.py`, `test/registered/unit/server_args/test_server_args.py`).

```bash
pip install -e "python[dev]"
python3 test/registered/kernels/ops/attention/test_dcp_sparse_decode.py
python3 -m pytest test/registered/unit/test_kda_cp_backend_helpers.py -v
python3 test/registered/e2e/dcp/test_glm53_flash_dcp.py     # 8 GPUs, downloads GLM-5.3-Flash
```

A parity check of the one-server context-parallel layout against TP 8 is in `test/manual/glm53_flash_cp_parity/`.
Its README has the commands.
It compares prompt logprobs of the whole last chunk and 64 greedy tokens, with the same TP 8 server re-chunked as the noise floor.
At 32k, 128k and 1M, context parallel is as far from TP 8 as TP 8 is from itself.
At 2M, past the model's 1,048,576-token context, both distances grow together.

`benchmark/kernels/attention/bench_dcp_sparse_decode.py` times the sparse decode kernel.

## Pulling the image

The images are published on the GitHub Container Registry.

```bash
docker pull ghcr.io/prefrontal-ai/sglang:lsa-v0.1.0                     # H200
docker pull ghcr.io/prefrontal-ai/sglang:lsa-v0.1.0-b300                # B300
enroot import 'docker://ghcr.io#prefrontal-ai/sglang:lsa-v0.1.0'        # enroot or pyxis
```

Each is the build below of `lsa-v0.1.0`, and its `org.opencontainers.image.revision` label names the commit.
Building is only needed for another commit.

## Building an image

The paper's images are built from `lsa-v0.1.0` and `docker/Dockerfile`, target `framework_final`, with CUDA 13.0.3 and torch 2.13 (cu130).
The H200 image is the plain build, and the B300 image is the same build with the `docker/b300` layer on top.
Plain Docker works.

```bash
git checkout lsa-v0.1.0
docker buildx build --target framework_final --platform linux/amd64 --load \
    -f docker/Dockerfile \
    --build-arg CUDA_VERSION=13.0.3 \
    --build-arg BUILD_TYPE=all \
    --build-arg BRANCH_TYPE=local \
    --build-arg INSTALL_FLASHINFER_JIT_CACHE=1 \
    -t sglang-lsa:$(git rev-parse --short=10 HEAD) .
```

The image name is arbitrary.
This one matches the `<name>:<sha10>` tag that `build.sh` gives its images.
A plain Docker build does not stamp the commit into the version string, as `build.sh` does.

`docker/enroot/build.sh` does the same for any commit and imports the result as an enroot squashfs, for clusters that run enroot or pyxis.
It needs Docker with buildx and enroot, and no GPU.
It builds from `git archive` of the commit, so uncommitted changes never leak in.
It stamps the commit sha into the version a server reports.

```bash
bash docker/enroot/build.sh --ref lsa-v0.1.0 --out-dir /path/to/images    # default out-dir is ./images
bash docker/enroot/build.sh --ref <sha> --layer <dir>                     # put <dir>/Dockerfile on top
```

The image lands at `<out-dir>/<name>-<sha10>.sqsh`, with the default name `sglang-lsa` (`sglang-lsa-<layer>` with a layer).
The scratch directory defaults to `$TMPDIR` or `/tmp`, set it with `SGLANG_BUILD_SCRATCH`.
It must hold the source tree and enroot's cache.
The build wants many cores and a lot of memory - we built with 32 cores and 200 GB of memory, with a six-hour limit.
On Slurm, submit the script with `sbatch` and your own resource flags.
The script works from the directory it is submitted from, so submit from inside the checkout, or use `sbatch -D <checkout>`.

### Resources and time

A build needs about 100 GB free in Docker's data root on the build node.
A build that runs out dies with "no space left on device".
The output directory needs room for the image, which is about 26 GB per squashfs.

A cold build of the base image takes about 35 to 40 minutes on 32 CPUs.
The import is slow on a network filesystem.
`enroot create` of a 25 GB squashfs took about 35 minutes on one.
Point `ENROOT_DATA_PATH`, `ENROOT_CACHE_PATH` and `ENROOT_RUNTIME_PATH` at writable disk that is local to the node, and allow at least 2 hours for the job.

### Expected warnings

Two messages during the build are not failures.
Pip reports a conflict between protobuf 7.37.0rc1 and nvidia-cutlass-dsl 4.6.2, which requires protobuf below 7.
`kernels download` finds no sgl-flash-attn3 cubin for torch 2.13, so FA3 compiles at first use on H200.

### Checking an image

```bash
enroot create --name sglang-lsa-check /path/to/images/sglang-lsa-<sha10>.sqsh
enroot start --rw sglang-lsa-check python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
enroot remove sglang-lsa-check
```

The expected output is `0.0.0.dev0+g<sha10> 2.13.0+cu130`.
The dev version string is expected.
It is the commit stamp that `build.sh` writes, and not an upstream release number.

The script's `--layer` option builds `<dir>/Dockerfile` on top of the base image, passed as `BASE_IMAGE`.

### B300

```bash
bash docker/enroot/build.sh --ref lsa-v0.1.0 --layer docker/b300 --out-dir /path/to/images
```

The layer is `docker/b300/Dockerfile`, and the image name defaults to `sglang-lsa-b300`.
This command builds the base image again, which takes the same 35 to 40 minutes from a cold start.
To build H200 and then B300, run both in one job on one node.
The second build reuses Docker's layer cache and took about 12 minutes.
The script removes the base image when it exits, but the layer cache stays.
The base is the stock image, built with CUDA 13.0.3 and torch 2.13.0 (cu130), Triton 3.7.1, sglang-kernel 0.4.7, FlashInfer 0.6.18, sgl-deep-gemm 0.2.0 and nvidia-cutlass-dsl 4.6.2.
The layer sets `TORCH_CUDA_ARCH_LIST="10.0a;10.3a"` and `FLASHINFER_CUDA_ARCH_LIST="10.0a 10.3a"`, so JIT compiles target B200 and B300 on a build host without a GPU.
It runs `check_image.py --no-gpu` at build time, which fails the build if an import or an arch is missing.
Run `python3 /opt/sglang/check_image.py` in the image on a GPU node for the full check.

On sm_103, set the attention backends explicitly.
FA3 is refused there, and Blackwell defaults GLM's KV cache to fp8 and the DSA backend to trtllm.
Under DCP, use `--dsa-prefill-backend tilelang` and `--kv-cache-dtype bfloat16`.
FlashInfer's autotune runs on sm_100 and above with a dummy batch, so `--disable-flashinfer-autotune` turns it off.

Then run the server in the image with the LSA plugin installed, from the LSA repository, as its README says.

## Rebasing onto upstream

```bash
git fetch upstream
git switch -c rebase/<new upstream sha10> lsa-v0.1.0
git rebase --onto <new upstream sha> 0e2aac500b
```

Resolve conflicts, then run the tests above that the fork added on a GPU node.
Check the plugin's dotted-path targets, as the section on extension points says.

## Credits

Two sets of upstream commits were cherry-picked into the fork while they were still open upstream.
Their authors keep the credit.

- `kpham-sgl`: decode context parallel for GLM-5.3-Flash. The target-only decode CP, and EAGLE/MTP support under DCP.
- `luoyuan.luo`: prefill context parallel for KDA (Kimi-Linear). The state pre-process for chunked prefill, the CP backend helpers (shard metadata and conv-halo exchange), the contiguous strategy, the end-to-end serving path, the TTFT benchmark, the lifted attention-TP restriction and the MHA one-shot route for contiguous prefill.

Everything else is a change of this fork, on top of the work of the SGLang community.

## License

SGLang is licensed under the Apache License 2.0, in `LICENSE`.
The fork's changes are under the same license.

## Per-file change list

Every file that differs between the base `0e2aac500b` and `lsa-v0.1.0`, grouped by area, with lines added and removed.
The list leaves out `FORK.md`, the README note and `docker/enroot/build.sh`.
Generate it with `git diff --numstat 0e2aac500b lsa-v0.1.0 -- . ':!FORK.md' ':!README.md' ':!docker/enroot/build.sh'`.

### Serving arguments, validation and environment (10 files)

```
+10 -2  python/sglang/srt/arg_groups/fields/parallel.py
+27 -1  python/sglang/srt/arg_groups/model_override_base.py
+2 -1  python/sglang/srt/arg_groups/model_overrides/kimi_k3.py
+18 -0  python/sglang/srt/arg_groups/model_overrides/qwen3_5.py
+8 -20  python/sglang/srt/arg_groups/overrides.py
+78 -0  python/sglang/srt/arg_groups/parallel_hook.py
+5 -1  python/sglang/srt/arg_groups/pipeline.py
+12 -0  python/sglang/srt/arg_groups/speculative_hook.py
+2 -1  python/sglang/srt/configs/model_config.py
+1 -0  python/sglang/srt/environ.py
```

### DSA (GLM-5.3-Flash sparse attention) kernels, backend and index-key cache (17 files)

```
+333 -0  python/sglang/kernels/ops/attention/dsa/dcp_sparse_decode.py
+26 -18  python/sglang/kernels/ops/attention/dsa/index_buf_accessor.py
+50 -9  python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
+10 -2  python/sglang/kernels/ops/kvcache/mla_buffer.py
+196 -0  python/sglang/srt/layers/attention/dsa/dcp_backend_ops.py
+173 -0  python/sglang/srt/layers/attention/dsa/dcp_sparse_gather.py
+23 -2  python/sglang/srt/layers/attention/dsa/dsa_backend_kpool.py
+4 -4  python/sglang/srt/layers/attention/dsa/dsa_indexer.py
+44 -4  python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py
+24 -14  python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py
+13 -4  python/sglang/srt/layers/attention/dsa/utils.py
+107 -25  python/sglang/srt/layers/attention/dsa_backend.py
+6 -0  python/sglang/srt/layers/attention/mqa_logits_utils.py
+2 -37  python/sglang/srt/mem_cache/dsa_cache_layer_split.py
+22 -9  python/sglang/srt/mem_cache/index_key_cache.py
+2 -2  python/sglang/srt/mem_cache/pool_host/dsa.py
+1 -1  python/sglang/test/kits/dsa_metadata_kit.py
```

### KDA, linear attention and Kimi-Linear (11 files)

```
+636 -0  python/sglang/kernels/ops/attention/fla/chunk_delta_h_cp.py
+8 -2  python/sglang/kernels/ops/attention/fla/index.py
+29 -0  python/sglang/kernels/ops/attention/fla/kda.py
+75 -18  python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
+131 -4  python/sglang/srt/layers/attention/linear/kda_backend.py
+238 -0  python/sglang/srt/layers/attention/linear/kda_cp.py
+72 -0  python/sglang/srt/layers/attention/linear/kda_head_shard.py
+7 -0  python/sglang/srt/layers/attention/linear/kernels/kda_triton.py
+7 -0  python/sglang/srt/layers/attention/mamba/mamba2_metadata.py
+4 -0  python/sglang/srt/layers/radix_linear_attention.py
+87 -24  python/sglang/srt/models/kimi_linear.py
```

### QSA and Qwen3.5 (7 files)

```
+19 -6  python/sglang/srt/layers/attention/qsa/mqa.py
+63 -1  python/sglang/srt/layers/attention/qsa/qsa_indexer.py
+174 -7  python/sglang/srt/layers/attention/qsa/sparse_attn.py
+123 -30  python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
+9 -1  python/sglang/srt/mem_cache/qsa_kv_pool.py
+63 -3  python/sglang/srt/models/qwen3_5.py
+1 -50  python/sglang/srt/models/qwen4_exp.py
```

### GQA under DCP (FA3 and triton) (3 files)

```
+77 -1  python/sglang/srt/layers/attention/flashattention_backend.py
+234 -0  python/sglang/srt/layers/attention/flashattention_dcp.py
+10 -4  python/sglang/srt/layers/attention/triton_backend.py
```

### Context-parallel and DCP layers (12 files)

```
+13 -0  python/sglang/srt/distributed/parallel_state.py
+31 -4  python/sglang/srt/layers/communicator_dsa_cp.py
+223 -0  python/sglang/srt/layers/communicator_mhc_hybrid_cp.py
+14 -2  python/sglang/srt/layers/cp/base.py
+286 -0  python/sglang/srt/layers/cp/contiguous.py
+2 -1  python/sglang/srt/layers/cp/interleave.py
+8 -6  python/sglang/srt/layers/cp/utils.py
+39 -0  python/sglang/srt/layers/dcp/comm.py
+14 -0  python/sglang/srt/layers/dcp/layout.py
+2 -0  python/sglang/srt/layers/dcp/metadata.py
+12 -7  python/sglang/srt/layers/dcp/planner.py
+28 -0  python/sglang/srt/layers/utils/cp_utils.py
```

### Other attention backends (DCP plumbing and shared fixes) (10 files)

```
+23 -28  python/sglang/srt/layers/attention/aiter_backend.py
+44 -0  python/sglang/srt/layers/attention/attention_registry.py
+10 -0  python/sglang/srt/layers/attention/base_attn_backend.py
+4 -6  python/sglang/srt/layers/attention/cutedsl_mla_backend.py
+5 -6  python/sglang/srt/layers/attention/flashinfer_mla_backend.py
+7 -8  python/sglang/srt/layers/attention/flashmla_backend.py
+1 -0  python/sglang/srt/layers/attention/hybrid_attn_backend.py
+2 -0  python/sglang/srt/layers/attention/tbo_backend.py
+5 -10  python/sglang/srt/layers/attention/tokenspeed_mla_backend.py
+25 -33  python/sglang/srt/layers/attention/trtllm_mla_backend.py
```

### Models and model executor (10 files)

```
+26 -0  python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py
+33 -17  python/sglang/srt/model_executor/pool_configurator.py
+7 -2  python/sglang/srt/model_executor/runner/eager_runner.py
+5 -0  python/sglang/srt/models/deepseek_common/attention_backend_handler.py
+24 -3  python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
+36 -9  python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
+10 -7  python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py
+133 -0  python/sglang/srt/models/deepseek_common/mla_decode_head_view.py
+40 -1  python/sglang/srt/models/deepseek_v2.py
+164 -1  python/sglang/srt/models/glm5_next.py
```

### Memory pools, scheduling, disaggregation, utilities (10 files)

```
+1 -1  python/sglang/srt/disaggregation/checksum.py
+6 -5  python/sglang/srt/disaggregation/decode.py
+3 -3  python/sglang/srt/disaggregation/prefill.py
+6 -1  python/sglang/srt/managers/schedule_batch.py
+1 -0  python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py
+2 -0  python/sglang/srt/mem_cache/hisparse_memory_pool.py
+21 -1  python/sglang/srt/mem_cache/kv_cache_configurator.py
+69 -18  python/sglang/srt/mem_cache/memory_pool.py
+1 -1  python/sglang/srt/mem_cache/mla_host_dedup.py
+15 -1  python/sglang/srt/utils/common.py
```

### Build (3 files)

```
+3 -1  docker/Dockerfile
+23 -0  docker/b300/Dockerfile
+181 -0  docker/b300/check_image.py
```

### Benchmarks (1 files)

```
+79 -0  benchmark/kernels/attention/bench_dcp_sparse_decode.py
```

### Tests (25 files)

```
+53 -0  test/manual/glm53_flash_cp_parity/README.md
+83 -0  test/manual/glm53_flash_cp_parity/parity_client.py
+35 -0  test/manual/glm53_flash_cp_parity/parity_compare.py
+1 -0  test/manual/layers/attention/dsa/test_index_buf_accessor.py
+1 -1  test/registered/attention/test_aiter_fp8_q_unified_attention.py
+318 -0  test/registered/attention/test_kda_cp_state.py
+258 -0  test/registered/dcp/test_dcp_sparse_prefill_gather.py
+37 -0  test/registered/dcp/test_dsa_trtllm_dcp_guard.py
+4 -0  test/registered/dcp/test_trtllm_mla_family_dcp_metadata.py
+131 -0  test/registered/e2e/dcp/test_glm53_flash_cp_dcp.py
+126 -0  test/registered/e2e/dcp/test_glm53_flash_dcp.py
+166 -0  test/registered/kernels/ops/attention/test_dcp_sparse_decode.py
+86 -0  test/registered/kernels/ops/attention/test_dsa_dcp_kernels.py
+158 -0  test/registered/kernels/ops/attention/test_dsa_dcp_prefill_kernels.py
+73 -0  test/registered/kernels/ops/attention/test_dsa_kpool_gather_prefix_int64.py
+182 -0  test/registered/kernels/ops/qsa/test_qsa.py
+1 -0  test/registered/unit/layers/attention/test_kv_translate_ownership.py
+1 -0  test/registered/unit/layers/attention/test_verify_mask.py
+48 -2  test/registered/unit/managers/test_mamba_checkpoint_depth.py
+2 -2  test/registered/unit/mem_cache/test_mla_host_dedup_primitives.py
+37 -1  test/registered/unit/mem_cache/test_qsa_kv_pool.py
+1 -0  test/registered/unit/model_executor/model_runner_components/test_attention_backend_setup.py
+11 -5  test/registered/unit/server_args/test_server_args.py
+1 -0  test/registered/unit/spec/test_dflash_overlap_hostsync.py
+236 -0  test/registered/unit/test_kda_cp_backend_helpers.py
```
