# Flash-Next LSA pilot maps — 2026-11-03

Status: trained, evaluated and saved. Jeff deferred native Strata integration and further testing until after work. No automatic run or reminder was created.

Production is restored: qwen38-flash-next-strata.service active/running, PID2876339, NRestarts0. The installed maintenance helper reports healthy: exact inference, process identity and readiness verified; HOLD null. Config SHA256 remains f3d1b869d3d36733fa58686b73537092733bc52b9c9c06ba7d2789bfb17c16a4. Nothing promoted.

## Deliverables

- pilot-v1/flashnext-learned-384x64.pt: trained query/key projections for all12 QSA layers (3,7,...47),24 FP32 tensors,589824 parameters,2.25MiB tensor payload.
- pilot-v1/flashnext-pca-384x64.pt: frozen PCA comparison.
- Matching .json files: input contract, per-layer temperature and weight SHA256, verified with upstream map_tau. Temperature is separate from the normalized maps.
- pilot-v1/report.json: immutable training, validation/test metrics, source-input hashes and export checksums.
- receipts/compatibility-audit.json: all96 capture identity/hash/sampling/step checks, dataset compatibility and metadata audit.
- receipts/gpu-map-smoke.json: real learned maps through actual CUDA clustering, selection, expansion and gathered attention.
- capture-plan.json, corpus-manifest.json, scripts and receipts/capture.patch: reproduction/provenance.

Remote experiment: /path/to/lsa-prep-20261103/flashnext-maps. Raw captures, original prepared data and corrected compatible prepared copies stay there. Original training inputs were retained unchanged.

## Training and held-out results

Four4096-token training documents; two4096-token validation documents; two6144-token final-test documents. Total36864 tokens, comprising pinned public technical/code text and deterministic fictional documents. No staff prompts. The entire document defines the split. All maps were frozen before validation/test; the test set is now consumed and must not become a tuning set.

Maps consume384 features: the192 nonrotary coordinates of each of two INT8-dequantized KV heads. Queries sum the12 matching Q heads within each KV group. Project to64 features and normalize each projected row. Production's existing native QSA maps are a different representation.

Training uses the current IQ3_S weights/control vector and FP32 dequantized INT8 cache inputs. Each layer trained600 AdamW steps against the actual native-attention probability mass, summed across24 independent heads and conditioned on visible long-range keys. PCA and temperature initialization use training documents only.

| Split / metric | PCA | Learned |
|---|---:|---:|
| Validation mean KL |0.37416|0.22257|
| Validation long-range mass,512 keys |74.49%|75.40%|
| Validation long-range mass,1024 keys |90.31%|90.80%|
| Test mean KL |0.49265|0.30981|
| Test long-range mass,512 keys |70.50%|72.17%|
| Test long-range mass,1024 keys |85.85%|86.90%|

Each split averages24 layer/document pairs. Both passed the declared pilot gate: mean KL improves, mean retained mass at both budgets does not decrease, and no layer/document mass regression exceeds3 percentage points. Worst validation regression is2.825 points at layer43/512 keys; worst test regression is1.221 points at layer43/512. Thus improvement is not uniform.

The37.1% test KL reduction is versus PCA maps, not a speed or quality improvement over production Strata. Data is4–6K tokens;128K behavior and end-to-end task quality are not qualified.

## Verification and corrections

All96 layer/document captures independently replay native attention within2.90e-6 maximum scale-relative output error (declared tolerance1e-4). Header identity, exact32-token sampling, token hashes, native step widths, grouping, causal IDs and probability normalization verified.

All12 learned maps passed CUDA projection/clustering/selection/expansion checks on eight actual query positions per layer from a6144-token held-out document. Each returned exactly1024 unique long-range keys plus512 window keys and4 sinks. Three layers (3,27,47) also passed gathered attention with24 query heads,2 KV heads and256 head dimensions. Maximum output relative L2 error0.0002252 (0.02252%); maximum LSE error2.80e-5. The oracle uses the same selected set and FP16-rounded inputs. FP16 input-rounding effect is reported separately. This is a standalone correctness smoke, not a native Strata serving test.

The first CUDA harness attempt indexed the returned LSE tensor as[B,T,H] instead of[B,H,T]. The harness was corrected; first-failure evidence is retained in receipts/gpu-map-smoke-attempt1. The GPU kernels and maps were not modified.

Original prepared files contain unused dense-target IDs padded with-1; native-attention training was unaffected. package_maps.py made prepared-compatible copies with zero-mass padding at valid ID0 and verified dense/native targets; all native targets stayed bitwise equal. Original train_queries metadata suggests a row holdout that this custom whole-document fit did not use. Do not use training-document eval_rows as held-out evaluation; compatible metadata explicitly records this limitation.

## Remaining integration

Strata does not load these LSA maps yet. The isolated Strata change only captured activations; it did not replace the native selector. Remaining work is a native map loader and LSA selector/cache adapter, incremental and speculative accepted/rejected-token state handling, followed by same-model quality/latency/VRAM comparisons with the required control vector, INT8 KV, MTP and131072 context.

Jeff's conditional promotion authorization remains valid if the complete candidate proves better and preserves the required behavior. Native integration/testing is deferred until after work at Jeff's latest instruction. No automatic production pause, experiment, service replacement or fallback test is scheduled.
