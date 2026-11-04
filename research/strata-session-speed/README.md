# Per-session generation-speed screen — November 4, 2026

**No faster setting was qualified or deployed.** The existing speculative-confidence floor 0.5 was descriptively fastest in the matched solo screen. Production keeps the approved Strata 0.1.56 configuration: eight 262144-token contexts, ordinary resident INT8 KV, all 24576 GPU experts, matched vision and prefill 8192. Sunshine remains stopped.

[RESULTS.md](RESULTS.md) separates valid timing observations, failed quality checks and incomplete concurrency measurements. [confidence-analysis.json](confidence-analysis.json) contains the independent arithmetic/timing audit. All 12 timed replies produced 1024 output tokens; outputs matched within each style across settings. Every setting, including the baseline, failed the same short arithmetic fixture, so the original confidence report remains **ineligible**.

The first concurrency collector stopped on a normal solo-to-batch prompt-count transformation. Its exact source/report are retained. The corrected collector then refused before sending requests because production was busy. A 180-second watch found no quiet gap; that attempt completed no C2 or full C1/C4/C8 comparison. A later, separate ordinary C8 run completed after production became idle, as described below. This is not a model or memory failure and supplies no concurrency speedup claim.


A later two-round **eight-session baseline passed** on the unchanged production configuration. Each session received a fictional 1215–2171-token prompt and generated 1024 tokens. Both rounds sampled all eight native decoding slots progressing together for about 18 seconds.

| Ordinary C8 round | Aggregate whole-request tokens/s | Per-session median tokens/s | Per-session range |
|---|---:|---:|---:|
|1|393.07|49.21|49.14–49.43|
|2|423.33|53.00|52.92–53.27|

These are measured baseline rates, **not a tuning gain** or an improvement over an earlier differently sized workload. Round 1 reported no cached prompt tokens; five of eight prompts reused a prefix in round 2. This qualifies overlapping eight-session generation on short prompts, not eight near-262K histories. See the [C8 analysis](c8-analysis.json), [independent audit](c8-audit.json) and [bottleneck note](C8-BOTTLENECK.md).

The post-run [health check](c8-final-health-public.json) at 11:18:45 UTC verified unchanged configuration pins, normal readiness, full capacity and vision, and Sunshine still stopped. No engine, service, sampler, control vector, queue or concurrency setting was changed.

The repository contains full fictional inputs, bounded ordinary-API collectors, CPU protocol/accounting checks and sanitized receipts. Private configurations, credentials, client content and live identities are excluded. See [DEPENDENCIES.md](DEPENDENCIES.md) and [CREDITS.md](CREDITS.md). Public copies contain role placeholders and are not a turnkey production installer.
