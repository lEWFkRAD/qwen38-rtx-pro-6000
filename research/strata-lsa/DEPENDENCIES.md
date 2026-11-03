# Dependencies and asset identities

This is the recorded environment for an isolated November 3, 2026 experiment. It is not a turnkey installer or a claim that arbitrary current package versions are equivalent. Model, runtime and build assets must be obtained separately under their own licenses. Example locations such as `/srv/lsa/model`, `/srv/lsa/runtime` and `/srv/lsa/build` are operator-selected role paths.

## Native engine and build

The integration targets [Strata](https://github.com/Niko1221/Strata) app 0.2.5 / engine 0.1.50 at upstream commit `1aff6d14117b88171fcfcec8f2b55fdc47f2b165`. The retained capture parent is local commit `3f83801ea1862652806fc6ebddab7f6cf57f3af6`; completed integration is local commit `876307a9ded78bbca941189e63230b33cfdc6e2a`. Obtain the corresponding published patch/source snapshot as described in [PROVENANCE.md](PROVENANCE.md).

The measured native build used Linux x86-64, CUDA toolkit 13.0.88, GCC 13.3, C++17, CMake/Ninja and target SM120. Strata's pinned llama.cpp/ggml build sources and their original notices remain dependencies. The feature is CUDA-only and must be built explicitly with `STRATA_ENABLE_LSA=ON`; this option defaults off. A feature-enabled binary still uses its original selector unless `STRATA_LSA_MAPS` names a compatible native map artifact. An explicit unsupported setup refuses startup.

The qualified integration scope is one GPU, ordinary fully resident unrotated INT8 KV, 24 query heads, two KV heads, head width 256, and at most 131,072 tokens. HIP, SYCL, Metal, rotated INT8, Q4/hybrid KV, streamed main KV and multiple GPUs were not qualified for this feature. The eight-slot experiment also has a main prompt session; retaining full FP32 projected histories adds about 4 GiB of working memory in this configuration. The experiment verified full expert and slot capacity rather than reducing either to make the new state fit.

Native decoder inference does not call Python or Triton and performs no allocation inside decode graph replay. Native map export and the research/reference tests have separate Python dependencies.

## Research and reference environment

The initial [Prefrontal AI LSA](https://github.com/Prefrontal-AI/latent-sparse-attention) source pin is `906be195c1052aaa74f95852ad1917ce1926ba64`. The isolated CPU Ward reference repair is local descendant `35b1bf9a10703b57da3eb073debfad6a66507269`. The unchanged upstream `uv.lock` SHA-256 is `a577ed7cdd5e0625395b818399ab0bc8af994dbd0f20bc6f389bd47172dd6853`.

Recorded research dependencies are Python 3.12, PyTorch `2.13.0+cu129`, Triton `3.7.1`, `nvidia-cutlass-dsl` `4.6.2`, `apache-tvm-ffi` `0.1.13.post3`, and uv `0.13.0` for bootstrap. The initial SM120 reference experiment selected Triton explicitly. Its cu129 wheel environment is separate from the native CUDA 13.0 build and the official vision bundle. The B300-oriented SGLang image was a source reference and was not the serving engine installed for this experiment.

## Retained API and vision runtime

The API canary retained Strata's Python/API runtime from v0.1.42 with the existing downstream compatibility/recovery behavior. The public v0.1.42 source identity is `61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`; [the existing runtime dependency record](../../strata-c8/DEPENDENCIES.md) documents the compatibility patch and original Python file provenance. The native candidate decoder was tested through that retained wrapper, not through a newly upgraded Python server.

The matched official vision executable and CUDA 13.0 libraries came from the official engine 0.1.50 bundle, build `1aff6d14117b+cuda13.0`. The saved manifest gives the official `strata-vision` SHA-256 as `cebdf0f985fe5de5428993835f8d60efab4ecabbde8c723e0f64f47bc24ef0a0`. The official decoder's hash is `ae49d5da81b9b65d50af1108c4b3f8b289ee198d9992d66268b7e885cf1a4776`; that is separate from both source-built comparison decoders. Keep the matched runtime libraries and vision executable together. This publication does not redistribute CUDA libraries or replace the host's driver.

The private API test bound only to loopback and used a fresh temporary key. Its wrapper gate override was limited to isolated test admission checks; production gate paths and recovery pins were unchanged. This harness is not a production-readiness receipt or an installer for the deployment's maintenance controller.

## Same-model assets

The model package is [alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF/tree/d4af3be5701549435e0770d625f79dd2def1358f), pinned to `d4af3be5701549435e0770d625f79dd2def1358f`. Its retained card attributes unchanged quantized model shards to [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09), revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.

| Asset | Saved publisher-manifest SHA-256 |
|---|---|
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |
| `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` | `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38` |

These hashes identify the retained verified-download/publisher record; publication preparation did not rehash the large weight files. The first shard supplies the native model and the second supplies the PLE/n-gram table. The experiment retains `--ple-io ram`, matching generated Strata pack/tokenizer/MTP data and the source expert profile. Those generated dependencies must match their archived provenance; substituting other packs, maps or tokenization is a new experiment.

The separate control vector is applied at strength 1.0, layers 4–44, `project`, `per-layer`. This modifies model behavior, including refusal behavior. It is not part of the LSA numerical optimization, and the comparison is not a claim about unmodified vanilla Qwen. Model, projector, vector and model-derived data retain their applicable upstream terms; the saved package card identifies Qwen Community License 1.0. No Apache or MIT source-code grant here replaces those terms.

The learned maps were trained for this Flash-Next geometry. The released 27B maps and the earlier synthetic million-key indexer trial are not interchangeable model assets and do not extend this engine's measured 131,072-token capacity.
