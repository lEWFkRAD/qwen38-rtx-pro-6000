# Eight-session throughput: measured limits and one future candidate

Current production completed two clean rounds of eight simultaneous 1024-token
responses at **262144-token capacity per slot**. Individual whole-request rates
had medians **49.21 and 53.00 tokens/s**; aggregate rates were **393.07 and
423.33 tokens/s**. All eight slots were observed decoding for approximately
18.10 and 17.85 seconds. These were modest 1215–2171-token synthetic inputs,
not eight near-capacity histories. Production settings and identity stayed
unchanged. See [the current results](RESULTS.md) for the receipts and scope.

The earlier approximately 28 tokens/s figure included cold prompt admission and
used only 256 output tokens. It was not a steady decoder ceiling:

| Historical 131K-capacity workload | Per-session observation |
| --- | --- |
| Direct native engine, 104–106-token inputs, 512 outputs, three paired rounds | 57.16 tokens/s median in measured common steady windows; range 56.63–57.52 |
| Retained API wrapper, approximately 4K cold inputs, 256 outputs, one spot | 27.87 tokens/s median from request start to completion; range 27.71–28.16 |
| Same API spot, after first visible content to completion | Approximately 39.35 tokens/s median; range 30.34–56.09 |

The last row uses final usage counts divided by elapsed delivery time after the
first content event. SSE buffering makes it an approximation; chunks were never
counted as tokens. Time to first content was 0.65–4.67 seconds, median 2.67 seconds.
Earlier streams also waited through later prompt admissions. The direct native
steady windows instead counted actual token arrivals while all eight streams
were live. See [the earlier comparison](../strata-0156-tuning/RESULTS.md).
Different prompt lengths, output budgets, capacity allocations and timing scopes
prevent treating the current result as a measured gain over that API spot.

The concrete source-level distinction is **solo speculation versus ordinary
batch decoding**. The managed process was read-only verified to have neither
`--batch-mtp` nor enabled `STRATA_BATCH_MTP`/`STRATA_BATCH_DF2`. Its
`STRATA_SPEC_COST` was absent. Ordinary continuing `BGEN` slots emit one token per
verifier window. They do not receive solo MTP drafts or prompt-lookup drafts
merely because the solo path reports those features. Solo confidence-floor
changes therefore do not address the continuing C8 path.

This is explicit in official engine source
`06698c674673008395ba6e971580538d6b1ea6e0`:

- `docs/BATCHING.md:161–169,628–630` describes solo-to-batch migration and the
  default absence of batch MTP drafts.
- `src/program/generate.cpp:13881–13962` assigns one row per ordinary slot;
  enabling batch MTP assigns two rows per slot, limits each eight-row window to
  four slots, and rotates when eight are active.
- `generate.cpp:13407–13416` applies `STRATA_SPEC_COST=1` to solo speculative
  chain/window selection. It does not change the fixed one-proposal batch MTP
  loop at `14118–14132`. The automatic cost-aware default in
  `src/core/nv_defaults.cpp:54–80` is specific to GB10/sm121, not this sm120 GPU.

**A future C2/C4 candidate is `--batch-mtp` (equivalently
`STRATA_BATCH_MTP=1`) with every other setting held fixed.** It is startup-only
and verifies one MTP proposal per slot. It is not a promising simple C8 fix:
`kVerifyNarrowT` is fixed at eight (`include/strata/kernels/window_limits.hpp:21`),
and the batch-MTP path uses that limit even in a wider-window build. Four slots
with two rows each can emit at most eight tokens per window at perfect draft
acceptance; ordinary C8 already emits eight. Rejected proposals reduce that
count and drafting adds work. Only a different kernel/layout cost could overcome
this disadvantage, which has not been demonstrated. C8 regression is a specific
risk, not a qualified improvement. No batch-MTP trial was run in this screen.

Its exact additional memory requirement is **unmeasured**. Slot drafters share
the solo dense/expert weights and draft vocabulary head
(`src/core/mtp.cpp:285–298,784–794`; `generate.cpp:6855`); charging each clone the
whole solo-drafter allocation overstates the cost. Private state, logits,
residuals and CUDA graphs still need measurement. The default drafter attention
window is 32768, and its full host-history allocation can fall back to fully
resident GPU storage if pinned memory is unavailable (`mtp.cpp:380–413`). Do
not assume the current free memory guarantees a fit. Any future trial must keep
eight 262K contexts, ordinary INT8, all 24576 GPU experts and vision, and reject
fallback in capacity or residency.

No service, queue, model, source, precision, confidence default or configuration
was changed by this analysis. No startup-flag trial was performed and no setting
was promoted; production remains unchanged.
