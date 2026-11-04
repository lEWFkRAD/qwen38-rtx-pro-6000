# Per-session speed results — November 4, 2026

No persistent setting, service, runtime source, model weight or client policy changed. The existing eight 262K production profile remains active and healthy; Sunshine stayed stopped. These tests use the installed official 0.1.56 engine with the retained 0.1.42 Python wrapper. They do not qualify sustained generation from eight near-limit contexts.

## Matched solo confidence screen

The fixed workloads were fictional prose and code prompts of 1213 and 1348 tokens. Each setting was tested twice per style in balanced order, after warming the same prefixes. All 12 timed replies generated 1024 tokens, had valid native clocks and observed solo execution, and reported unchanged process/configuration/capacity. Each style's output hash was identical across settings. Sampling and the existing control vector were retained. No EOS suppression was used.

| Request-local confidence floor | Prose native tokens/s, median | Code native tokens/s, median | Equal-style geometric change vs 0.5 |
|---|---:|---:|---:|
|0.3|262.75|379.15|−4.44%|
|0.5, existing default |278.05|392.35|reference|
|0.7|275.75|391.25|−0.55%|

Both alternatives were descriptively slower in all four matched native-clock comparisons. The small 0.7 difference is not a statistical significance claim. Native generation rates exclude prompt/queue/transport time. End-to-end timings, token usage, cache counts, drafting acceptance and individual repetitions are preserved in the [original report](receipts/solo-confidence-r1/report.json) and [independent analysis](confidence-analysis.json). These warmed short-context timings cannot be substituted for speed on long real conversations or concurrent streams.

The three exact-marker checks passed. The independent Python fixture evaluates `17*19 - 23*7 + 11` to 173; every setting returned 177 while correctly sorting the list. These are three shared failures, including the baseline, with no observed new pass/fail regression. The predeclared strict quality gate therefore failed: the original report remains **ineligible**, the audit recommends no winner, and nothing was promoted. Valid timing observations do not turn this failed qualification into a pass.

## Baseline and concurrency limitations

The separate single pilot completed 1024 tokens at 205.65 tokens/s end to end and 219.4 native tokens/s. Its 101-token input differs from the longer screen; this is a pilot, not a repeatable speedup. The first collector's 1213-token solo case completed 1024 output tokens at 259.2 tokens/s end to end.

The same [r1 baseline](receipts/live-baseline-r1/report.json) stopped during C4 because the collector initially rejected a normal startup migration: a request can enter a native batch slot with its original prompt plus already-generated tokens. Only the collector's own HTTP requests were closed. Its failed status and [executed source](live_benchmark-r1.py) remain unchanged. The corrected source uses a unique initial candidate and stable slot binding supported by the installed wrapper, while final SSE prompt usage must still equal the original frozen prompt and completed/output ledgers must match. Forty CPU protocol, lifecycle and migration checks passed.

The [corrected attempt](receipts/live-baseline-r2/report.json) refused before sending any synthetic requests because four actual production requests were active. A subsequent 180-second observation found three real requests continuously active. Existing work was neither paused nor cancelled. That earlier sequence obtained no C2 or complete C1/C4/C8 matrix. A later separate C8 baseline is recorded below; no queue-policy gain is claimed. SSE chunks are never counted as tokens, and buffered delivery estimates are explicitly approximate.

A separate read-only observation of one production hour found 364 substantial replies with about 94.6 tokens/s median end to end; 94.6% of prompt tokens were cached. That heterogeneous workload had much longer prompts and variable overlap. It is not a matched comparison with this synthetic screen, and the larger solo numbers above are not an achieved production speedup.


## Later ordinary eight-session baseline

After the deferred attempts, production became idle and a separate two-round C8 run completed using seed `2026110411`. The engine and all serving settings were unchanged. Frozen fictional prompts were 1215, 1350, 1489, 1624, 1763, 1898, 2037 and 2171 tokens. All 16 replies generated exactly 1024 tokens and ended normally at their requested length cap. Each round's final usage totaled 8192 output tokens; completed-request and output-counter deltas were exactly 8 and 8192. SSE response IDs were unique, no contamination was observed, and the benchmark's HTTP threads drained.

| Metric | Round 1 | Round 2 |
|---|---:|---:|
|Aggregate whole-request output tokens/s|393.0683|423.3295|
|Per-session median whole-request tokens/s|49.2064|53.0020|
|Per-session range, tokens/s|49.1363–49.4314|52.9169–53.2748|
|Median first visible-content delay|1.414 s|0.541 s|
|First visible-content delay range|0.429–2.524 s|0.168–1.301 s|
|Observed eight-decoding-slot sample span|18.1018 s|17.8509 s|
|Common visible-output overlap|18.1906 s|17.9202 s|
|Jain fairness from whole-request rates|0.9999963|0.9999958|

The observation was more than a peak counter: 72 consecutive samples in round 1 and 71 in round 2 showed all slots 0–7 decoding, with every slot's generated count increasing by 1011 and 1009 tokens respectively across those sampled intervals. This establishes overlapping ordinary short-prompt decoding by eight sessions. It does not mean eight GPU kernels execute at the same instant, qualify eight near-limit deep histories, or supply an exact token-counted steady-state rate. SSE delivery is buffered; whole-request usage divided by elapsed time remains the primary rate.

Round 1 reported `cached_tokens=0` for all eight prompts. Round 2 reported `[0,1343,1482,0,0,1891,2030,2164]`: five reused prefixes and three reported no prefix reuse. It was not a fully warm round, and the between-round difference is not assigned to one cause. Prompt counts and token hashes matched the frozen fixtures, source hashes matched the executed collector, and the independent audit recomputed the accounting, rates, fairness and continuous slot progress.

Sources: [original C8 report](receipts/live-C8-r1/report.json), [full synthetic fixtures](receipts/live-C8-r1/fixtures.json), [numeric analysis](c8-analysis.json), [independent audit](c8-audit.json), and [bottleneck analysis](C8-BOTTLENECK.md). These two baseline repetitions are not a before/after engine comparison. In particular, do not claim a gain over an earlier 28-tokens/s observation, the warmed solo screen or heterogeneous long production sessions: their inputs, output budgets, cache state and overlap differ. The confidence screen above still has no qualified winner and its arithmetic failures remain unchanged.

The post-C8 [safe health record](c8-final-health-public.json), dated 11:18:45.562917 UTC, verified normal authenticated readiness and the static contract with unchanged configuration/personal pins, context 262144, eight resident slots, all 24576 experts and vision. The service was idle at that check; Sunshine was inactive. No production setting, service or queue policy was changed by this measurement.

## Next-step boundaries

Deployed Hermes has no supported provider-specific active-generation queue setting. Its global session limit and gateway API cap have different scopes and reject excess work; they do not supply a two-active/eight-cached Onyx queue. No scheduler, client limit or reduced concurrency was installed. Prioritizing fewer simultaneous generations would be a separate policy/implementation choice; the earlier eight-session preference stays in force pending that choice.

Official Strata release 0.2.10 uses the same 0.1.56 engine and describes DFlash2 as a single-request drafter. Auto selects MTP for two or more concurrent requests. Its reported single-stream gains are upstream measurements, not results on this node, and do not establish an eight-active replacement. No DFlash2 asset was downloaded or tested here. See the [official release notes](https://github.com/Niko1221/Strata/releases/tag/v0.2.10).

The earlier final independent file check found all five approved production pins unchanged, no maintenance HOLD and the service active at 262144 × 8; the later post-C8 health record above supplies the updated check. Production remained available throughout; there was no production restart. No LSA, precision change, new weight training, driver/power change or Sunshine activation occurred.
