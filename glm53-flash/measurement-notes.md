# Measurement details and normalized command reconstruction

This is an external **llama.cpp** experiment on GLM-5.3-Flash. It does not measure Colibri's `c/glm53.c` engine. The accompanying evidence JSON is an allowlisted derivative of retained receipts, with original receipt hashes for audit linkage. It contains no raw prompts, generated answers, machine identities, endpoint addresses or service configuration.

## Hardware and coverage

The September 3 canary inventory records one **RTX PRO 6000 Blackwell 96 GB**, **Intel Core Ultra 9 285K**, and **62 GiB host RAM**. CPU/RAM are prior inventory, not a fresh hardware survey for this publication. The storage device, filesystem and storage benchmark were not recorded in the reviewed measurement sources. The profile metadata records driver **610.43.02** and Nsight Systems export version **2026.4.1.191**.

The measured application was a custom text-only llama.cpp build targeting `120a-real`. The pinned SGLang container image supplied the build/runtime environment; SGLang did not serve this GGUF. Model shard hashes, upstream commit, required CPU MoE fix, runtime snapshot manifests, source patch hash and Q8 draft hash are in `evidence-summary.json`.

The target remained AJ-IQ2_XXS throughout. Q8-cache rows used **v8**; Q4-cache rows used **v9**, which adds Q4 pooling support. They are distinct snapshots. The private Q8 DFlash2 conversion has SHA-256 `d21f4f7166e5ade490479bb3886909bf37220bc618b4d8047c26400075d5f747`; it is not distributed with this report.

## Effective best-run server command

The following is a **normalized reconstruction**, not the original host command and not a turnkey deployment script. Placeholder paths replace deployment-specific locations. It assumes an already prepared immutable custom runtime and the exact Q8 draft; obtaining upstream llama.cpp alone does not reproduce those bytes. It omits container/service ownership and recovery orchestration.

```bash
env \
  NVIDIA_TF32_OVERRIDE=0 \
  LD_LIBRARY_PATH="<CUSTOM_RUNTIME>/build-sm120/bin:<CUDA_LIBRARY_DIR>" \
  LLAMA_GLM5NEXT_SPARSE_DECODE=1 \
  LLAMA_GLM5NEXT_SPARSE_TELEMETRY=1 \
  LLAMA_GLM5NEXT_FUSED_KPOOL=1 \
  LLAMA_GLM5NEXT_FUSED_KPOOL_DIAG=1 \
  LLAMA_GLM5NEXT_NATIVE_RS=1 \
  "<CUSTOM_RUNTIME>/build-sm120/bin/llama-server" \
  --model "<MODEL_DIR>/GLM-5.3-Flash-AJ-IQ2_XXS-00001-of-00002.gguf" \
  --n-gpu-layers 99 --flash-attn on \
  --ctx-size 360448 --parallel 4 --no-kv-unified \
  --batch-size 512 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 --no-context-shift \
  --model-draft "<DRAFT_DIR>/GLM-5.3-Flash-DFlash2-Q8_0-pure.gguf" \
  --spec-type draft-dflash --spec-draft-ngl all \
  --spec-draft-n-max 3 --spec-draft-n-min 1 --spec-draft-p-min 0 \
  --cache-type-k-draft f16 --cache-type-v-draft f16 --ctx-checkpoints 0 \
  --jinja --metrics --log-verbosity 3 \
  --threads 16 --threads-batch 16 \
  --host "<LOOPBACK_BIND_ADDRESS>" --port "<LOCAL_PORT>"
```

CUDA graphs, graph reuse and ordinary fusion were enabled: the launcher did not set `GGML_CUDA_DISABLE_GRAPHS`, `LLAMA_GRAPH_REUSE_DISABLE`, or `GGML_CUDA_DISABLE_FUSION`. The separate `GGML_CUDA_GRAPH_OPT` optimization was not enabled. These are the frozen launcher's choices, not assumptions about an arbitrary user's inherited environment. No Nsight wrapper was active for the four performance-table rows.

All 46/46 target layers were reported offloaded. The container used one GPU, a 52 GiB memory/swap-total limit, and a read-only runtime/model mount. The 16-thread setting and memory limit are configuration, not hardware capacity measurements. Peak VRAM was not established.

For the other observations, replace only the documented inputs: BF16 draft with v8/Q8 cache/depth3; v9/Q4 cache/Q8 draft/depth7; v9/Q4 cache/Q8 draft/depth3. This table records distinct runs, not a fully controlled one-variable experiment.

## Effective benchmark command and workload

```bash
python3 "<FROZEN_BENCHMARK_DIR>/run-native-sustained-common-window.py" \
  --base-url "<LOCAL_API_BASE>" \
  --output "<RESULT_DIR>/sustained90k.json"
```

The same directory must contain the exact `run-four-user-depth.py` and `run-warm-cache-discriminator.py` helpers. The benchmark's source hash is in the evidence summary. This reconstruction shows the workload selection; the original invocation's operational timeout override is not in the performance receipt. The process was separately bounded by its canary window and service lifetime.

The frozen harness fixes **four users, 89,400 rendered prefix tokens each, one sampled seed each, and 512 measured generated tokens each**. It uses the native `/completion` API with pinned slots, full token-array prompts, `cache_prompt=true`, temperature0, top_p1, seed530053, and returned token IDs. A sequential one-token prefill must evaluate/cache the entire prefix. The sampled seed is then appended to that exact prefix. Four threads wait at one barrier before beginning measured requests. `--ignore-eos` was **not** supplied.

The 360,448-token server context is divided into four **90,112-token slots**. The harness separately uses a conservative **90,000-token accounting limit**; its 89,401-token input and 512-token output fit within both. The measured limit-stop cache occupancy must be exactly **89,912**, with `cache_n=89400`, `prompt_n=1`, and 512 streamed/reported generated tokens.

## Warm-up, timing and limitations

- There was **one measured full benchmark per configuration**, with different generated prompt nonces. No repeated-run median or confidence interval is available.
- Native short correctness checks preceded the measured queue. The Q8-draft and Q4/depth7 short checks ran on the corresponding full-context servers; the Q4/depth3 comparison reused the already established Q4 gate. There was no extra full 90K benchmark warm-up run.
- The immediate preparation for each measurement was four sequential full-prefix prefills and one seed each. This work warms the exact prefix caches and is excluded from the decode interval. Best-run preparation was roughly 145–149 seconds per slot; exact durations and prompt-processing rates are in the evidence JSON.
- Every user's reported common-window rate is `(streamed tokens - first output chunk tokens) / (latest last output - earliest first output)`. All four best-run first chunks contained one token, so the numerator is **511**, while total generated output remains512. The best window was14.145857224 seconds, yielding36.123650332 tokens/s each. Warm-append TTFT excludes prefix preparation.
- Model outputs from this historical workload included reasoning/repetition. Its `enable_thinking=false` template argument was ignored. The later finite task explicitly selects `reasoning_effort`; its correctness results do not retroactively validate the earlier output.
- The finite EOS evaluator uses a source-derived occupancy contract, distinct from the unchanged exact limit-stop contract above. Correct accounting does not imply correct answers.
- Both profiling attempts remain invalid. One began after output; the other stalled and recorded no CUDA events. No bottleneck or HES-specific failure cause is established.

The next software-trace/idle-start workflow is prepared but unrun. It requires actual CUDA events and a valid interior four-user interval in a short canary before a new long trace. The target remains four correct, useful outputs at at least60 generated tokens/s **per user**.
