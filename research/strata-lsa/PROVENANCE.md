# Evidence and reproduction scope

This publication preserves the November 3, 2026 Latent Sparse Attention research and native Strata integration on one RTX PRO 6000 Blackwell Workstation Edition, 96 GB, compute capability 12.0. It includes successful, failed and superseded research evidence. The final native integration passed its recorded functional gates, but did not meet the performance condition for deployment. No production promotion was performed.

## Source identities

| Component | Recorded revision | Meaning |
|---|---|---|
| Upstream Latent Sparse Attention | `906be195c1052aaa74f95852ad1917ce1926ba64` | Original pinned [Prefrontal AI release](https://github.com/Prefrontal-AI/latent-sparse-attention/tree/906be195c1052aaa74f95852ad1917ce1926ba64) |
| Isolated LSA reference repair | `35b1bf9a10703b57da3eb073debfad6a66507269` | Local descendant restoring symmetric pair costs in the CPU Ward reference; the original baseline failures remain recorded |
| Strata upstream base | `1aff6d14117b88171fcfcec8f2b55fdc47f2b165` | [Strata app 0.2.5 / engine 0.1.50 source](https://github.com/Niko1221/Strata/tree/1aff6d14117b88171fcfcec8f2b55fdc47f2b165) |
| Isolated Strata capture parent | `3f83801ea1862652806fc6ebddab7f6cf57f3af6` | Local descendant adding bounded INT8 attention calibration capture only |
| Completed native integration | `876307a9ded78bbca941189e63230b33cfdc6e2a` | Local integration revision used for the final source checkpoint |
| LSA serving-fork reference | `042d955ba31b962911f63b114c2fb245effa86f2`, tag `lsa-v0.1.0` | [Prefrontal AI SGLang fork](https://github.com/Prefrontal-AI/sglang/tree/042d955ba31b962911f63b114c2fb245effa86f2); source reference, not the serving engine used here |

The local descendants above identify historical research states; they are not claimed to be commits in the upstream projects. Published patches and cleaned source snapshots preserve the corresponding work. Portable path substitutions and public receipt projections are distinct publication files, with their own checksums; an original source hash does not imply that a sanitized copy has identical bytes.

The final candidate decoder SHA-256 was `ce96085e004cd202f129f6a8ab9c788f5464b52b9c156f386b9fb05904459720`. The same-day control was the retained capture-only Strata 0.1.50 build, SHA-256 `70067facc003c2971f4f085b21766176c632c279edaeb8e1b8d938d3e5843045`. These are identities of measured builds, not claims that they are official upstream release binaries.

## Corpus, captures and learned maps

The pilot uses eight whole documents: four 4,096-token training documents, two 4,096-token validation documents, and two 6,144-token final-test documents, totaling 36,864 tokens. Source manifests record the token/text hashes and each public-source input. Technical documents and part of the code corpus derive from pinned LSA documentation/code; the training code document derives from pinned MIT-licensed Strata source. The other operations and narrative documents are deterministic fictional material. The corpus contains no staff conversations or client workpapers.

There are 96 native calibration captures: eight documents across the 12 QSA layers, numbered 3, 7, through 47. They preserve actual queries, stored INT8 key/value codes and scales, native selected IDs, and native attention outputs needed for replay. Structural and numeric validation is recorded separately from map evaluation and whole-model behavior.

The learned maps contain 24 FP32 matrices of shape 384 by 64, with separate per-layer temperatures. Each layer trained for 600 steps. Frozen PCA comparisons, learned model states, temperatures, original captures, original prepared inputs, corrected compatible prepared inputs and the failed/passing receipts are retained in the research data archives. The compatible copies repair unused dense-target padding; the native training targets remain unchanged. Saved model states do not contain optimizer or random-number-generator state.

The entire document defines its train/validation/test split. A trailing row from a training document is not held out. The pilot test set has been consumed; tuning against it requires a new independent test set. Improved pilot KL versus PCA does not establish improved model answers, retrieval or production speed.

The saved learned `.pt` file SHA-256 is `4d8a606e0ddd33858cb662e7f82f723dd54ca96776f3c383af28b30935a85d02`. The native exported map SHA-256 is `a5a69b217f23d9a2f7995516a91d1baa81e25b7733159ac5c9c1d0aa28c5342c`. The export preserves all 589,824 matrix values byte-exactly and keeps temperatures separate from normalized maps. Its complete loaded payload and selector parameters contribute to saved-session identity. A session fingerprint is not authentication; use the artifact SHA-256 for integrity.

## Measurements and limitations

The native comparison retains the same IQ3_S/control-vector model variant, INT8 KV, eight slots, 131,072 context, MTP and all 24,576 resident GPU expert slots. It tests independent synthetic whole-model fixtures, prefix reuse, speculative acceptance/rejection, multiple slots and cancellation. The API canary additionally tests plain text, exact forced-tool arguments, high reasoning and a fresh image through the retained Python wrapper and matched official vision encoder.

The final 125,076-token retrieval request completed in 17.78 seconds with LSA versus 16.91 seconds for control. The short eight-slot batch-window measurement was 296.5 versus 414.6 rows/s. Those batch summaries cover four windows each, not sustained client throughput. One sequential full-fixture pass per arm is not a statistical performance benchmark. Configured capacity is distinct from a measured prompt length; no larger model context is established by the initial synthetic indexer tests at up to one million keys.

The arithmetic fixture expected `208`; control and LSA both returned `188`. That quality failure remains failed. No new failure appeared in the finite tested fixtures, but this is not general quality equivalence. Functional integration and component parity do not satisfy the condition that a production candidate prove better.

## Public copies and archive boundaries

The publication retains meaningful research code, source changes, notes, corpus provenance, data and results. Large source/data payloads are separate release assets, with byte counts and checksums identified by RELEASE-ASSETS.json. Each archive is independently downloadable; verify its checksum before extraction. File-integrity checks do not authorize a GPU experiment or service interruption.

Readable public copies replace personal filesystem prefixes, private mesh addresses and the live maintenance owner with role placeholders. No credentials or private serving configurations are included. Source hashes connect these copies to the saved originals, while publication checksums cover the files actually distributed. The unchanged native integration checkpoint retains its original experiment paths and ephemeral test process IDs; it was separately scanned for credentials and unrelated private material. Unrelated continuity history, private recovery snapshots, installed environments and dependency caches are omitted. Base model weights and proprietary CUDA components remain external dependencies.

The final source archive omits unrelated upstream community benchmark results containing unexplained credential-like fields and one provider URL test fixture. SOURCE-ARCHIVE-EXCLUSIONS.json lists the omitted files. Native source and build files are unchanged. The earlier raw Strata capture source archive is superseded by the cleaned final source snapshot and retained capture patch; the original and repaired LSA source archives are retained unchanged. Tiny publicly tracked upstream fixture/control-vector GGUF files are source fixtures, not the large production model shards.

Historical preparation notes describe what was known then. Final integration results supersede earlier statements that native wiring or GPU validation was pending. Original failures, incomplete runs and intermediate slower candidates remain historical evidence, not final deployment recommendations. Publishing this work does not restart a service, change recovery policy or promote the candidate.
