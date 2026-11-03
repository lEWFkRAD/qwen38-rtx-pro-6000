# Resume Flash-Next LSA integration

Saved 2026-11-03T14:16:36.670539+00:00 on Onyx-Linux, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, GPU-REDACTED.

Checkpoint: /path/to/lsa-prep-20261103/checkpoints/flashnext-maps-20261103T141636Z
Working experiment: /path/to/lsa-prep-20261103

Completed: all12 learned384x64 map pairs, PCA comparison,600-step model checkpoints, temperature sidecars,96 native activation captures, original and compatible prepared data, split corpus, source archives, scripts, failed and passing receipts. The learned maps passed the declared4-6K pilot gate and all12 real-map CUDA selection checks. Three full Flash-Next attention geometry checks passed. This does not establish production speed/quality superiority.

Start with flashnext-maps/maps/pilot-v1/report.json and flashnext-maps/README.md. The working remote layout is maps/pilot-v1 and corpus/manifest.json, unlike the flattened Windows delivery paths mentioned in that README.

Production was healthy with no HOLD at checkpoint time; see state.json. This is a dated observation. Creating or restoring this checkpoint must not pause, replace or restart production. Jeff deferred integration and further testing until after work; no automatic run is scheduled.

Next: native Strata map loader and LSA selector/cache adapter, preserving INT8 KV, the selected IQ3_S/control-vector model,8slots,131072context and MTP. Validate incremental state and accepted/rejected speculative tokens, then actual model quality and latency. Promotion is already conditionally authorized only if the complete candidate proves better and preserves required behavior.

All weights and data are saved. prepared/ is immutable v1 training evidence; prepared-compatible/ repairs unused dense-target padding with identical native targets. Whole documents define train/validation/test. Do not call the last training-document row held out. The test set has been consumed; new tuning needs a new independent test set. Checkpoints contain model states, not optimizer/RNG states.

Source archives preserve the original LSA commit, isolated Ward-repaired LSA commit and isolated Strata capture commit. The Strata change only captures activations; it does not enable LSA. Extract archives under this checkpoint root to recover their expected source paths. Scripts still contain provenance paths from the original working experiment; inspect and adjust only when intentionally relocating. Current .venv, CUDA/runtime libraries and external model assets remain at their existing paths and are not duplicated here. Check capture-plan.json and state.json before reproduction. Source and data restoration alone never authorizes starting a test.

Integrity: SHA256SUMS lists every payload file. Run sha256sum -c SHA256SUMS from this checkpoint directory; this is a file-integrity check only.
