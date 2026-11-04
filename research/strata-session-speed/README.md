# Per-session generation-speed screen — November 4, 2026

**No faster setting was qualified or deployed.** The existing speculative-confidence floor 0.5 was descriptively fastest in the matched solo screen. Production keeps the approved Strata 0.1.56 configuration: eight 262144-token contexts, ordinary resident INT8 KV, all 24576 GPU experts, matched vision and prefill 8192. Sunshine remains stopped.

[RESULTS.md](RESULTS.md) separates valid timing observations, failed quality checks and incomplete concurrency measurements. [confidence-analysis.json](confidence-analysis.json) contains the independent arithmetic/timing audit. All 12 timed replies produced 1024 output tokens; outputs matched within each style across settings. Every setting, including the baseline, failed the same short arithmetic fixture, so the original confidence report remains **ineligible**.

The first concurrency collector stopped on a normal solo-to-batch prompt-count transformation. Its exact source/report are retained. The corrected collector then refused before sending requests because production was busy. A 180-second watch found no quiet gap; no C2 or full C1/C4/C8 comparison was completed. This is not a model or memory failure and supplies no concurrency speedup claim.

The repository contains full fictional inputs, bounded ordinary-API collectors, CPU protocol/accounting checks and sanitized receipts. Private configurations, credentials, client content and live identities are excluded. See [DEPENDENCIES.md](DEPENDENCIES.md) and [CREDITS.md](CREDITS.md). Public copies contain role placeholders and are not a turnkey production installer.
