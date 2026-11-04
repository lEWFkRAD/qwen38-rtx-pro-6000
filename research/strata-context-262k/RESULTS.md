# 262144-token context results — November 3, 2026

The tested eight-slot/8K-prefill configuration passed measured full-residency allocation, native exact long retrieval and normal API features, including a 259,999-token API request. **Full eight-way deep decoding was not qualified.** All failed attempts remain failed. The user subsequently explicitly authorized promoting this memory-fit configuration; the promotion was deployed and verified through a separate record, which does not change the research outcome.

**Later production incident — recovery verified, November 4 UTC:** a credential-only edit failed the strict configuration pin and blocked restart, without changing serving arguments or environment; this was not a 262K memory failure. The user explicitly authorized an additive compatibility credential and one manual start before the automatic window, preserving existing credentials and resealing dependent pins. Start was at 02:26:17.188249 UTC; managed readiness and an exact canary passed at 02:27:10.620721 UTC. A separate new recovery audit passed 19/19 checks, three approved credentials passed and an invalid credential returned 401. Four remote status GETs and one exact synthetic completion passed at 02:27:57.7167785 UTC. Automatic limits and previous attempt history were unchanged; `production_final_ready=true`. See the [incident record](outage-20261104/INCIDENT.md) and [safe recovery summary](outage-20261104/recovery-summary-public.json). The original promotion's 19/19 audit and 7/7 canaries remain historical and unchanged; they were not rerun or relabeled as the new recovery evidence.

## Fixed components and scope

Tests used one RTX PRO 6000 Blackwell 96 GB, driver 615.71.09 and the existing 550 W policy. The decoder was the already qualified official Strata 0.1.56, source `06698c674673008395ba6e971580538d6b1ea6e0`, SHA256 `21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788`, with matched official vision/CUDA libraries. API requests used the retained 0.1.42 wrapper. Model/pack/tokenizer/MTP/projector, custom IQ3_S weights, vector 1.0 at layers 4–44/project/per-layer, speculation and fused serving settings were retained. No engine build, model training, LSA or positional-scaling argument change was involved.

Each private profile requested `--max-context 262144 --kv-resident 262144`, ordinary INT8 KV and all 24576 experts. Engine `kv_resident=0` denotes full residency. Decode share stayed 0.75. The three native shapes were four slots/prefill 32768, eight/prefill 16384 and eight/prefill 8192. The full API configuration additionally includes the matched vision process; native allocation success is not automatically an API-residency pass.

The frozen fictional inputs contain one 259999-token document with its opaque target at token 256019 and eight independent 251764–251835-token documents with five seals each. Values appear once in their own input and never in another slot document; an independent visible-record parser supplies exact JSON answers. Five-key positions are approximately 9K/62K/135K/193K/245K. There is no answer in the question. The accepted-answer continuation intentionally checks prefix reuse, rather than independent retrieval.

Configured 262144 and tested 259999/260055 input lengths are distinct: the tests leave answer allowance and do not exhaust the final exact context boundary. Evidence is finite synthetic retrieval/capacity behavior, not general model quality or a speed comparison against the 131K policy.

## Completed native evidence

| Attempt / shape | Full expert residency | Exact cold/reuse/per-slot/recovery checks | Peak live decodes | Minimum sampled free GPU |
|---|---:|---|---:|---:|
| native-r1: 4 slots, prefill 32768 |24576|Passed|1|12746 MiB|
| native-r1: 8 slots, prefill 16384 |24576|Passed|1|2333 MiB|
| native-r2: 8 slots, prefill 8192 |24576|Passed|1|4598 MiB|

Every assigned slot read its distinct ~252K input fully fresh with no reused tokens and returned exact five-key JSON. Each arm also passed the short initial canary, fresh 259999-token tail retrieval, deep accepted-prefix continuation and short recovery. The 8K-prefill cold request took 36.442 seconds; its continuation reused 259992 tokens. All native infrastructure and cleanup gates passed, and host available memory remained above 20 GiB in native-r1.

The short JSON answers finished before subsequent admissions, giving peak live decoding 1. Thus the native runs prove individual deep fills/retrieval across the assigned pools, not four/eight simultaneously generating requests or independently reusable eight deep histories after all fills. Source reports: [native-r1/report.json](native-r1/report.json), [native-r2 8K report](native-r2/slots8-prefill8192-report.json), [independent r1 audit](INDEPENDENT-R1-AUDIT.json).

## Completed private API evidence

| Attempt | Result | Peak active / decoding | Minimum sampled free GPU |
|---|---|---:|---:|
|api8-r1, prefill 16384|Rejected before features: 24007 experts instead of 24576|Not qualified|3645 MiB|
|api8-r2, prefill 8192|Six normal checks passed; deep load rejected unsupported batch `logit_bias` with HTTP 400|0 / 0 load|2876 MiB|
|api8-r3, prefill 8192|Normal EOS; one stream ended before eight-way overlap; inconclusive|8 / 4|2953 MiB|
|api8-r4, prefill 8192|Eight one-token priming requests; bounded reserve expired before eight-way overlap|8 / 2|2956 MiB|

All four **overall API attempt statuses are failed**. This preserves their intended full qualification gate while separately recording what passed. Every owned private process group was removed and every temporary credential file was deleted. No 8K-prefill API attempt reported a memory failure; failing to observe the required overlap within a bounded run is not proof that it is impossible.

`api8-r1` correctly rejected the 569-expert residency shortfall before running feature cases. That result is not relabeled as full residency or a model-quality error.

`api8-r2` retained all 24576 experts with matched vision and passed authentication rejection, exact text, forced tool-call JSON, high-reasoning arithmetic, the green-square image and exact near-capacity retrieval. The API reported **259999 prompt tokens and 27 completion tokens**, matching the frozen input. Its later load requests attempted EOS suppression through `logit_bias`; the retained batching API rejected them with HTTP 400 before load inference. This was a request/collector incompatibility, not evidence of memory exhaustion. Returned tool calls were inspected, never executed.

`api8-r3` removed bias and used normal EOS. Eight active full-prompt requests were observed, but only four decoding together. One response finished normally at 5373 usage completion tokens (5372 visible re-tokenized tokens) before the required eight-way condition. It therefore did not qualify eight-deep decoding.

`api8-r4` made eight sequential full-prompt priming requests with `max_tokens=1`, each reporting its full 251894–251965-token input and one output token. That is not proof of eight retained/preloaded slot histories: the native continuing-slot admission path requires an output allowance greater than one. The subsequent long requests used normal EOS, with no suppression. The intended gate required eight decoding slots at once, matching full prompt lengths, correct leading JSON for all eight documents and observed native output progress, followed by owned cancellation/drain and short recovery. Peak decoding was two; the bounded observation ended before its drain reserve. The eight-way gate and normal drain/recovery qualification were **not reached**. Client/process cleanup succeeded, which is a separate observation.

The load helpers changed system/tail instructions to elicit long continuations, so their capacity screens are distinct from normal exact-response retrieval. SSE chunks were never counted as tokens. Preserve the exact source for each attempt: r2/r3 archived helpers, current r4 helper and recorded source hashes. [test-outcomes-public.json](test-outcomes-public.json) and [INDEPENDENT-FINAL-AUDIT.json](INDEPENDENT-FINAL-AUDIT.json) summarize completed evidence without private identities.

## Source provenance and decision

The native-r1 plan records guard source `77b810ea2d830dfe7cb76871d24a824b0b13a78082eadbc68786b8d9a3e1863a`; its exact recovered file is retained alongside final `3011d627633691e0567507d7fe497d929d30a6a0050eb6f4f391cecc9b85bf84`. Only spaces in two refusal strings differ; no threshold or control-flow change is claimed. See [GUARD-PROVENANCE-NOTE.json](GUARD-PROVENANCE-NOTE.json). Sanitized public code has separate hashes; it does not rewrite the executed source identity.

The exported [memory-fit-profile.json](memory-fit-profile.json) changes only max-context to 262144, resident KV to 262144 and prefill to 8192, keeping parallel 8 and the fused environment unchanged. Its qualification is explicitly limited to measured fit, retrieval and features; full eight-deep decoding remains unqualified.

After the bounded trials, the prior 131072 production policy was successfully restored through normal managed readiness. The user then explicitly instructed that the 262K configuration become production. That later authorization supersedes the earlier no-promotion decision, not the failed test records. At promotion, the 262144/8-slot/prefill 8192 profile became live: normal managed readiness, 19/19 independent checks and 7/7 production canaries passed. Active Hermes client context metadata was also updated to 262144 with only the six-byte value change, ACLs retained and no restart. Sunshine remains stopped with enablement unchanged. See [PROMOTION.md](PROMOTION.md) and [deployment-summary-public.json](deployment-summary-public.json). Full eight-deep decoding remains unqualified; no failed research attempt is relabeled by this deployment.
