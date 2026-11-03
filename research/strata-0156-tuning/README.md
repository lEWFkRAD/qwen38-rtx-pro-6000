# Stock Strata 0.1.56 benchmark and serving tuning

The stock engine benchmark and settings tuning are complete. The recommended profile is `profiles/fused_prefill.json`; a reviewable serving overlay is `tuned-serving-overlay.json`. Production remains paused and was not upgraded. Read [RESULTS.md](RESULTS.md) for all decisions, tests and limitations.

Three alternating matched pairs measured eight-stream whole-request output at 369.7 versus 412.6 tokens/s (+11.6%), and a 125K-token retrieval request at 16.87 versus 14.92 seconds. A separate real-API spot using eight roughly-4K prompts measured 178.8 versus 221.6 tokens/s (+23.9%); it is a different workload and a single spot, not a repeated statistical benchmark. Tools, reasoning and a fresh image passed. Context remains 131,072 tokens per slot.

Both engines retain arithmetic failures in the finite fixtures; no new pass/fail regression appeared. This is performance tuning, not a model-accuracy upgrade.

Jeff requested benchmarking the newest Strata release and then tuning it, and explicitly deferred LSA for this campaign. Fine tuning here means serving-performance settings with the existing model weights.

The official Strata 0.2.10 release ships Control/engine 0.1.56. Its Linux CUDA engine is built from `06698c674673008395ba6e971580538d6b1ea6e0`, distinct from the app release tag target. The signed release feed, component manifest and installed files are verified in `receipts/official-bundle-verification.json`.

## Fixed comparison contract

- Actual saved production control: official engine 0.1.50 at `/opt/onyx-strata-0150-candidate/strata`, SHA-256 `ae49d5da81b9b65d50af1108c4b3f8b289ee198d9992d66268b7e885cf1a4776`.
- Official candidate: `official-engine-0.1.56/strata`, SHA-256 `21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788`, with its matched CUDA libraries and vision executable.
- Same IQ3_S shards, generated pack, tokenizer, MTP, control vector at strength 1.0/layers 4–44/project/per-layer, ordinary resident INT8 KV, eight slots and 131,072-token configured context. Require all 24,576 experts to remain GPU-resident.
- No LSA map, precision reduction, smaller slot allocation, model-weight training or competing model server.

## Campaign

First compare only the engine/libraries with saved settings unchanged. Then test a bounded set of explicit profiles based on the newest wrapper's recommended parallel/GDN settings. The retained older wrapper does not automatically supply those newer settings, so native profile records must name them explicitly. Four of the nine parallel flags were already set in the control; the vendor's 1.45× gain over 0.2.8 is not a measured gain over this baseline.

Measure real eight-request streams, common-window aggregate output, individual responsiveness, whole-request time, and a single active request within the eight-slot allocation. Tiny cancellation and native-window tests are structural diagnostics, not application throughput. Keep tuning fixtures distinct from final held-out quality fixtures; preserve failures and actual output lengths. Validate the strongest profile with interleaved repeated comparisons, long/near-capacity retrieval, prefix reuse, speculative acceptance/rejection, cancellation, tool arguments, reasoning and a fresh image.

The first host observation is `receipts/initial-state.json`: production and Sunshine remain stopped, no compute process owns the GPU, and the original owned maintenance HOLD remains in place. Config/recovery hashes are unchanged. Native and private-loopback tests own and clean up only their own child processes. They require the exact existing HOLD throughout. The private API credential is temporary and removed after use.

Production is not restored or upgraded by the benchmark harness. Changes to the saved serving configuration require a separate qualified deployment step; current research alone must not clear the owned HOLD or recovery history.

## Earlier work

LSA is archived separately at https://github.com/lEWFkRAD/qwen38-rtx-pro-6000/releases/tag/strata-lsa-2026-11-03. Its 2.52 GB of research downloads and all ten GitHub release asset checksums were verified. It is excluded from this stock-engine comparison.

Sunshine must stay stopped. Jeff reiterated that it is not to be brought back; any later inference restoration must exclude Sunshine unless he explicitly requests it.
