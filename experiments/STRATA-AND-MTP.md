# Strata and batched MTP — October 5, 2026

Evidence: [configuration comparison](receipts/strata-comparison.json),
[wide-MTP matched pairs](receipts/batched-mtp.json), and
[dated notebook excerpts](receipts/strata-notes.json).

## Eight active histories

The retained Strata 0.1.39 base source was
`6f32ec070f23ced9f50e704d854d775da52591ab`, with a required resident-batch
fix. The IQ3_S checkpoint revision was
`d4af3be5701549435e0770d625f79dd2def1358f`, with the Huihui vector,
INT8 KV, 131,072 context, eight slots, 16,384 prefill rows, PLE in RAM,
all 24,576 experts on GPU, and `no-prefill-borrow`.

| Eight submitted short requests, 512 output tokens each | TensorFold | Strata |
| --- | ---: | ---: |
| Configured active slots | 4 | 8 |
| Median aggregate end-to-end tok/s, 3 rounds | 384.92 | 268.12 |
| Median time until all eight begin, s | 5.556 | 2.493 |

TensorFold supports eight slots in software; this comparison configured four.
It does not establish an inherent four-slot limit. Checkpoint and vector
differences also preclude a quality-equivalence claim.

Strata's sustained eight-request median was 296.03 aggregate tok/s, with a
separate retained-service run at approximately 297.91. Eight approximately
128K histories and cached 512-token followups passed with eight simultaneously
decoding slots; full-context aggregate throughput was approximately 114 tok/s.
Short-context rates should not be extrapolated to full histories.

Native batch parity and a 39K interleave/preemption/resume check passed.
Fused-prefill, decode-share and MTP-threshold sweeps did not establish a
convincing overall improvement. Broad model-quality evaluation remained open.

## Batched MTP: correctness repaired, speed not improved

The default-off candidate first failed all eight greedy comparisons. A concrete
PLE history bug supplied too few predecessor entries for the second row of a
speculative span. Per-row causal history fixed it; subsequent greedy and
fixed-seed sampled comparisons matched complete token streams.

The initial corrected timing runs had a PLE memory-lock warning in both arms.
Later runs preserved unlimited memory-lock limits. The 16-row candidate also
first failed an eight-column projection guard; the final candidate supported
16-row planning/buffers, chunked native dense projections at eight, and used
distinct graph keys. Those failed revisions remain historical.

Final wide candidate SHA-256:
`26144cccc4635e78e464c858971b3f7ee164cb58d9062d82e172a14180f0845d`.

| Native-protocol matched pair | MTP off, aggregate tok/s | Wide MTP on, aggregate tok/s | Full token parity |
| --- | ---: | ---: | --- |
| Greedy, default threshold | 275.11 | 249.48 | 8/8 streams |
| Fixed-seed sampled | 272.28 | 243.64 | 8/8 streams |
| Greedy, minimum draft probability 0 in both arms | 274.53 | 261.26 | 8/8 streams |

The best experimental pair remained **4.83% slower**. One round per condition,
native-protocol timings and matched corrected candidate arms do not establish
repeatable HTTP production performance.

Wider verification reduced greedy windows from 620 to 325, but graph time
rose from about 0.62 to 1.23 seconds; drafting and commit costs persisted.
The wider graph alone did not create a net speedup.

Verification included 5,567,162 host checks with sanitizers, six offline suites,
16 harness checks and 17,367 small CUDA checks. These do not replace missing
long-context, cache/interleave, cancellation, mixed-vector, boundary, broad
quality and repeated-performance gates. **The candidate was not promoted.**
The receipt explicitly keeps `full_model_qualified=false`.
