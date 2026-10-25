# One GPU, eight workflows: the October 22 Strata recipe

For the later saved settings and observed readiness issue, see the
[October 25 configuration snapshot](CURRENT-CONFIG-2026-10-25.md).
The configuration and measurements on this page retain their October 22 scope.

This dated field recipe runs eight concurrent Qwen3.8-Flash-Next sessions on one
RTX PRO 6000 Blackwell 96 GB. The measured short-prompt common decode window
delivered **493.7 aggregate output tokens/s**, or **61.7 output tokens/s per
active session**, with all eight generating. That is a **48.0% improvement**
over the matched earlier engine profile on the same model and workload.

<img src="assets/qwen-c8-firm-throughput-v2.png" alt="Eight sessions share one batched decoder and GPU, with expert and KV residency and separate steady and whole-request throughput results" width="800">

For a firm sharing an inference server, aggregate output measures capacity and
per-session speed describes responsiveness. Prompt-admission latency matters as
well. Report those quantities together: eight actual simultaneous streams are
a different measurement from a single-stream test or eight times its rate.
These are synthetic inference results, not measured employee productivity.

## What was measured

| Metric | Engine 0.1.42 profile | Engine 0.1.46 profile |
| --- | ---: | ---: |
| C8 common-window aggregate output tokens/s | 333.4865 | 493.6775 |
| C8 common-window output tokens/s per session | 41.6860 | 61.7095 |
| Matched common-window gain | — | +48.034% |
| Current-profile whole-request aggregate output tokens/s, admission included | Not used for a controlled comparison | 421.368 |

The steady result is the median of two qualified common decode windows per
engine. Each round sent eight distinct 51–53-token synthetic counting prompts
and requested 512 output tokens per session. Token arrival accounting was
reconciled against server-reported usage. The common interval begins 250 ms after
all eight streams have produced their first content and ends 250 ms before the
first stream finishes producing content. It counts the content tokens that arrived
within that interval; it excludes prompt admission and end-of-sequence tokens.

The current-profile whole-request result is the median of two clean rounds,
409.603 and 433.133 output tokens/s, including admission and completion time.
The earlier profile's first whole round had external work at its final snapshot;
**the +48% is not an end-to-end gain claim**. Later rounds also had limited
prompt-cache reuse. The comparison is a steady eight-way decode comparison,
not a uniformly cold-prompt test.

All 16 paired 512-token output hashes matched across the earlier/current arms.
That is output parity on this workload; it does not establish general model
quality. The engine bundle and selected batch settings changed together, so
the measured gain belongs to the combined profile, not any single flag.

[Benchmark method and reproduction](BENCHMARK.md) and
[selected machine-readable evidence](evidence/results-2026-10-22.json) retain
the precise measurement boundaries and source identities.

## The saved configuration

| Component | Qualified setting |
| --- | --- |
| Hardware | One RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM |
| Power policy | 450 W limit; not a measured power draw or energy-efficiency result |
| Model | Qwen3.8-Flash-Next **GSQ-RCO IQ3_S** assets and a control vector; see dependencies |
| Decoder and vision engine | Strata 0.1.46, matched official CUDA 13.0 bundle |
| Python/API layer | Retained Strata 0.1.42 runtime with downstream compatibility patches |
| Concurrency | Eight resident conversation slots |
| Context capacity | 131,072 tokens per slot |
| KV policy | INT8; full main-context and all eight slot KV allocations in VRAM |
| Expert cache | All 24,576 expert slots GPU-resident |
| PLE I/O | RAM; GPU expert/KV residency is not a claim that every model component lives in VRAM |
| Prefill and speculation arguments | Prefill cap 16,384; `--spec 4`; `--spec-min-p 0.5`; native MTP asset |

The context number is configured capacity. The 494 tokens/s result used short
prompts, not eight simultaneous 131K prompts. A separate 80K correctness/reuse
check and a seven-generators-plus-cold-prompt test are recorded in the evidence;
they do not turn the short-prompt result into a long-context speed guarantee.

The conceptual schematic emphasizes GPU expert weights and KV storage. The
host-side PLE/RAM path and runtime processes are not a complete memory diagram.

The decoder child receives these settings through the serving JSON `env`:

```json
{
  "STRATA_BATCH_DECODE_SHARE": "0.75",
  "STRATA_PF_FUSED": "0",
  "STRATA_BATCH_SELF_COMMIT": "1",
  "STRATA_QSA_SLOTS": "1",
  "STRATA_SLOT_COPY_DIRECT": "1",
  "STRATA_SLOT_COPY_BATCHSYNC": "1"
}
```

The profile uses shared batched decoding, resident expert weights, resident
per-session state, and the newer engine's batch/copy paths. A native drafter is
configured, but every measured C8 stream reported zero drafted/accepted tokens:
**this C8 result is batched target decode, not batch MTP throughput**.
On this SM120 GPU, enabling QSA slots does not imply that every attention step
uses a fused cross-slot kernel. `STRATA_PF_FUSED=0` preserves the earlier
IQ3_S prompt-math path. Neither the optional packed-Q6 trial nor the PDL trial
is part of this default; neither showed a useful repeatable gain in this test.

The parent runtime also uses the environment in
[service-environment.example.conf](config/service-environment.example.conf).
Do not confuse the parent environment with the decoder-child JSON settings.
The example retains the drafter ring and `--kv-resident 131072`; an engine
metric of `kv_resident=0` denotes full residency in this profile.

## Reproduce the profile on your own machine

1. Read [dependencies and provenance](DEPENDENCIES.md). Obtain the pinned
   upstream engine bundle, matching model/pack/tokenizer/MTP assets, and the
   retained Python runtime. This is an IQ3_S modified checkpoint recipe, not
   a claim about an arbitrary download with a similar model name.
2. Prepare an isolated runtime using the documented base and downstream patch.
   Keep decoder, vision encoder, and CUDA libraries from the same verified
   bundle. This repository distributes examples and patches, not those binary
   model/runtime assets.
3. Adapt [strata-server.example.json](config/strata-server.example.json) to your
   asset locations and set a new API key. The published paths are generic and
   the example binds to loopback. Apply the parent-runtime environment from
   the companion example before starting that runtime.
4. Confirm the engine reports eight slots, context 131,072, INT8 KV with full
   residency, and 24,576 expert slots. Inspect actual library resolution and
   confirm only the intended model server owns the GPU. Available VRAM varies
   with desktop and request activity; a prior free-memory snapshot is not a
   fixed reservation.
5. Run the [bounded HTTP benchmark](scripts/benchmark_c8.py) on an idle test
   server with the documented method. For an A/B, change only the documented
   profile, retain identical prompts and assets, compare complete output
   hashes, and report admission-inclusive and common-window figures separately.
6. Qualify your actual workload, including tools, images, disconnects and long
   histories. Save the serving configuration, matching dependency hashes and
   recovery policy together before making it a deployment default.

The source deployment saved this profile as its default, with startup enabled,
process-bound readiness checks and a protected matching backup. Those are
deployment controls, not an assertion that a fresh installation has already
passed the same checks. This publication did not restart the live model or run
another GPU benchmark.

## Validate this publication offline

From the repository root:

```bash
python3 strata-c8/validate.py
python3 strata-c8/scripts/benchmark_c8.py --help
python3 -m unittest discover -s strata-c8/tests -v
python3 experiments/validate.py
```

The first command checks the new bundle's manifest, settings, evidence
arithmetic and local links. The last command checks the unchanged October 15
archive. Neither performs model inference. [SHA256SUMS](SHA256SUMS) covers
the files in this recipe, excluding itself.

This is an independent deployment report. Upstream engine, model and library
licenses still govern their respective components; see
[dependencies and attribution](DEPENDENCIES.md).
