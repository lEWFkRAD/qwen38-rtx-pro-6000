# First stock-engine screen

One sequential screening pass per version with the same settings. This is not a statistical speed qualification. Both used all eight slots, 131072 configured context, ordinary resident INT8 and all 24576 resident experts. LSA was absent.

| Metric | Official 0.1.50 | Official 0.1.56 |
|---|---:|---:|
| Eight-stream common-window output, tok/s | 454.6 | 461.4 |
| Admission-inclusive eight-request output, tok/s | 185.4 | 188.2 |
| One active request within eight slots, wall ms | 1574.9 | 1830.6 |

The common windows contain 227 and 225 observed tokens per slot, respectively, over about four seconds, with all eight streams alive. Each request generated its 256-token cap. Eight distinct roughly-4K prompts include fictional prose/runbook and code workloads.

Both versions failed the same two new arithmetic/code checks: 644 was answered as 588, and 53 as 10. Independent CPU evaluation confirms the reference values; those failures remain failed. The other four tuning checks passed.

Generated text differs across versions, including after token seven on the single-active-request workload. The observed draft acceptance difference therefore compares different trajectories; it does not isolate an MTP-policy regression. These are application-like fixed-prompt measurements, not identical-token kernel timings.

Decision: the engine-only eight-stream difference is small. Continue bounded settings screening; do not promote based on this pass. Raw outputs, timings, telemetry and retained harness source are in receipts/version-screen-r1/.
