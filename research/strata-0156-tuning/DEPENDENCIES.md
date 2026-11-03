# Dependencies and asset identities

This records the November 3, 2026 stock serving-setting experiment. It is not a turnkey installer, environment lock or production deployment recipe. Obtain model/runtime assets separately under their terms. Paths such as `/srv/strata-bench/engine-0156`, `/srv/strata-bench/prompt-source`, `/srv/strata-bench/api-runtime` and `/srv/strata-bench/model` are portable role examples, not original operator paths.

## Official native engines and matched libraries

The selected decoder is the official **engine 0.1.56**, from [app release v0.2.10](https://github.com/Niko1221/Strata/releases/tag/v0.2.10), build `06698c674673+cuda13.0`, source [`06698c674673008395ba6e971580538d6b1ea6e0`](https://github.com/Niko1221/Strata/tree/06698c674673008395ba6e971580538d6b1ea6e0). No source-built or LSA-patched decoder was used.

| Official dependency | Recorded identity |
|---|---|
| [Linux CUDA13 engine archive](https://github.com/Niko1221/Strata/releases/download/v0.2.10/strata-engine-0.1.56-linux-x86_64-cuda-cuda13.tar.gz) | 667,699,126 bytes; SHA256 `df6a6dfd806289caa42becd2acc14081815bedac9ec8f5efa3821a7d10582ee5` |
| Actual 0.1.56 decoder | SHA256 `21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788` |
| Matched 0.1.56 vision executable | SHA256 `bf6c2a1ae8a33100cc9afb6d44c83f893d6ae5d2ab5d1ff17095d9059adc729b` |
| Original signed engine manifest | SHA256 `02497065b6118a4b7639f834571b5ac5fe9591a56c57f687617127eca5ed2989` |
| Official retained 0.1.50 control decoder | SHA256 `ae49d5da81b9b65d50af1108c4b3f8b289ee198d9992d66268b7e885cf1a4776`; build/source `1aff6d14117b+cuda13.0` / `1aff6d14117b88171fcfcec8f2b55fdc47f2b165` |

Keep each engine with its own verified `strata-vision`, `libstrata_kq.so`, `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13` and manifests. The host NVIDIA driver is separate. The measured machine was Linux x86-64, RTX PRO 6000 Blackwell SM120, driver 615.71.09 and the retained 550 W policy. No engine compilation was part of this campaign.

The public [feed](https://github.com/Niko1221/Strata/releases/download/v0.2.10/strata-feed.json), [detached signature](https://github.com/Niko1221/Strata/releases/download/v0.2.10/strata-feed.json.sig) and [release checksums](https://github.com/Niko1221/Strata/releases/download/v0.2.10/SHA256SUMS.txt) were checked against retained trusted public update keys. Feed and component signatures, archive/file hashes and independent OpenSSL verification are recorded in [official-bundle-verification.json](receipts/official-bundle-verification.json) and [independent-openssl-verification.json](receipts/independent-openssl-verification.json). Public metadata copies are included during publication; use original upstream bytes for any raw-file hash changed by documented text normalization. Vendor executables, CUDA libraries and the archives themselves are not republished here.

The [Control 0.1.56 archive](https://github.com/Niko1221/Strata/releases/download/v0.2.10/strata-control-0.1.56-linux-x86_64.tar.gz), 99,658,667 bytes, SHA256 `9fde2c69d8509b22984a8402505cb49300f70df82f4d6033ab73e32592e799e5`, was downloaded/prepared and signature/hash checked. **It was not used as the API wrapper or Python environment for the measured requests.** Preparing it is not an API upgrade result.

## Prompt preparation and retained API are separate roles

The native benchmark imports only pure Python `Tokenizer` and `ChatTemplate` from a source checkout, then sends token IDs to the independent official executable. The historical `SOURCE` path happened to name an earlier LSA working tree. For reproduction, point that role at a clean [Strata 0.1.50 source pin](https://github.com/Niko1221/Strata/tree/1aff6d14117b88171fcfcec8f2b55fdc47f2b165); neither an LSA map nor an LSA decoder is needed. The imported files were independently compared byte-for-byte with that upstream Git object:

| Pure prompt helper | SHA256 |
|---|---|
| [`tools/strata_tokenizer.py`](https://github.com/Niko1221/Strata/blob/1aff6d14117b88171fcfcec8f2b55fdc47f2b165/tools/strata_tokenizer.py) | `6db1ea62d3edbaa64dce139088146ccf370933998d5f997cd5f41a39300075a6` |
| [`serve/frontend.py`](https://github.com/Niko1221/Strata/blob/1aff6d14117b88171fcfcec8f2b55fdc47f2b165/serve/frontend.py), `ChatTemplate` | `e0fded1ab8c856afa8cda7464b40d2753737de18dead42fc75381def13b1c6a1` |

A retained source snapshot is also available as [strata-lsa-source-876307a.tar.gz](https://github.com/lEWFkRAD/qwen38-rtx-pro-6000/releases/download/strata-lsa-2026-11-03/strata-lsa-source-876307a.tar.gz), SHA256 `229b7da01955b3630f83d15aa947c1cae48dad13f05d9ab2b023bf1bc945853e`; it contains these unchanged helper files alongside unrelated earlier work. The clean upstream pin is sufficient for this role. The historical archive does not authorize or require enabling LSA.

The real API requests ran through the **retained Strata 0.1.42 Python wrapper**, with the deployment's existing compatibility/recovery behavior. Its public source identity is [`61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406`](https://github.com/Niko1221/Strata/tree/61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406). The existing [runtime dependency record](../../strata-c8/DEPENDENCIES.md) and [compatibility patch](../../strata-c8/patches/runtime-0142.patch) explain the earlier public provenance; the exact runtime hashes used now are in the completed API manifests. Do not replace the measured runtime with Control 0.1.56 and call it the same experiment. Strata remains MIT-licensed; see [LICENSES/Strata-MIT.txt](LICENSES/Strata-MIT.txt).

The harnesses used Python 3.12 and require `regex` and Jinja2 for the pure helpers, plus Pillow for the synthetic image check and the retained wrapper's own dependencies. PyTorch/Triton training or LSA map preparation is not required by this stock inference campaign. A complete new Python environment lock is not supplied. System inspection/guard adapters call tools such as `nvidia-smi`, `systemctl`, `ss` and a node-specific maintenance helper. Set your own configuration/role paths and authoritative ownership guards; no private serving configuration or credential is included.

## Same-model assets and behavior

The retained package is [alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF/tree/d4af3be5701549435e0770d625f79dd2def1358f), revision `d4af3be5701549435e0770d625f79dd2def1358f`. Its retained card attributes unchanged quantized shards to [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09), revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.

| Asset | Retained verified-download/publisher SHA256 |
|---|---|
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |
| `Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` | `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38` |

These hashes come from the existing verified-download/publisher record in the [prior asset dependency documentation](../strata-lsa/DEPENDENCIES.md); publication did not rehash the large weights. The first shard supplies the native model and the second supplies the PLE/n-gram table. The experiment retains `--ple-io ram`, the matching generated Strata pack/tokenizer/MTP runtime data and source expert profile. Those generated dependencies must match the recorded model/tokenizer provenance. Substituting another pack, quantization, MTP vocabulary or chat template is a new experiment.

The separate control vector is applied at **strength 1.0, layers 4–44, `project`, `per-layer`**. This changes served behavior, including refusal behavior. The measured custom variant is not vanilla Qwen. The API sampling block retains `experimental_speed_projection=true`, while no request-specific sampling values were added for the throughput spot.

Both arms retained ordinary full-resident INT8 KV, eight 131,072-token slots and all 24,576 experts. Requested `--kv-resident 131072` is reported by the engine as `kv_resident=0` when fully resident. MTP used `--spec 6 --spec-min-p 0.5` with prompt lookup, and the selected profile retained 32,768-token prefill chunks and decode share 0.75. No context extension, weight training, LSA inference or deployment occurred.

Model, MTP, projector, control vector, tokenizer/model assets and model-derived responses retain applicable upstream/model/output terms. The retained package card identifies **Qwen Community License 1.0**. The downstream Apache and upstream MIT source-code notices do not replace those terms or grant new blanket rights over model weights or generated model responses. Attribution boundaries are in [CREDITS.md](CREDITS.md); measured outcomes and limits are in [RESULTS.md](RESULTS.md).
