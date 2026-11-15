# Serving 12 concurrent streams on one RTX PRO 6000: what the batch=1 leaderboard cannot see

Every speed-test entry on the localmaxxing.com leaderboard for Qwen3.8-Flash-Next, including ours, is a batch=1 single-stream run. This page records what happens when the same engine serves many streams at once. It is a capacity measurement, not a throughput claim.

## Setup

Same production endpoint as the rank-1 run: Strata with DFlash2 speculative decoding (6 draft tokens, min-p 0.5), IQ3_S pack, conversation cache 4096 MiB, `--pool-workers 12`, KV int8, max context 262144. Measured live on port 8002 on 2026-11-15 with no engine restart. Each stream requested 256 output tokens at temperature 0; one warm-up call preceded the trials.

## Method

For each concurrency level N, N identical chat completions were fired simultaneously and all N awaited. Aggregate tok/s is total completion tokens divided by wall time. Per-stream tok/s is each response's own token count over its own wall time. Numbers come from engine `usage` counters, not client-side estimates.

## Results

| Streams | Run 1 aggregate | Run 2 aggregate | Per-stream (run 1, min/med/max) |
| ---: | ---: | ---: | --- |
| 1 | 519 tok/s (0.49 s) | 517 tok/s | 520 |
| 4 | 285 tok/s (3.59 s) | 327 tok/s | 71 / 72 / 72 |
| 8 | 426 tok/s (4.81 s) | 444 tok/s | 53 / 54 / 54 |
| 12 | 406 tok/s (7.56 s) | 407 tok/s | 34 / 52 / 60 |

Each trial fired N identical 256-token completions simultaneously and awaited all N. Aggregate is total completion tokens over wall time; per-stream is each response's own tokens over its own wall time.

## Findings

- **Aggregate throughput is flat, not scaling.** Total output holds around 400 to 500 tok/s from one stream to twelve. The card is already compute-saturated at batch=1 because DFlash2 speculative decoding fills the idle compute. Extra streams share the same compute budget rather than adding to it. Anyone expecting a 12x aggregate win from concurrency will not find one here.
- **Concurrency is close to free.** The other side of flat throughput: serving 12 streams costs almost nothing in total tokens per second. Twelve interactive sessions each still see 34 to 60 tok/s, which is well above reading speed, while the card keeps roughly the same total throughput a batch=1 leaderboard run shows. A batch=1 engine would have to queue these sessions serially.
- **This is the number the leaderboard structure hides.** The canonical run answers "how fast can one request go." The table above answers "how many people can use this rig at once." Both rank-1 runs were recorded while this same engine was serving real multi-session traffic.

## Raw evidence

The trials above ran directly against the production endpoint with engine-reported `usage` counters. Script and captured responses: see `batch-concurrency-20261115.json` alongside this file.

## On the leaderboard (2026-11-15)

The c12 measurement is now a public run: [cmwhar7th026cmi01gv4pf586](https://www.localmaxxing.com/en/speed-tests/cmwhar7th026cmi01gv4pf586), recorded with batchSize 12 / concurrency 12 in its engine flags, tokSOut 367.7. It is unverified by design: the site's Verified badge requires batch=1, concurrency=1, and a canonical prompt, so no concurrency run can ever be Verified. It is the only entry on this model's board carrying concurrency metadata.

The full sustained-decode matrix (1/4/8/12 at 0 and 16K context, plus 8K-128K prefill scouts) was also measured with lmx's llm-decode-bench v0.7.7; the site's matrix leaderboard accepts only vLLM and SGLang engines, so that grid lives here as `matrix-20261115T034051Z.bench.json` instead. Sustained aggregate decode rose from 378.6 tok/s at concurrency 1 to 524.0 at concurrency 8 (497.8 at 16K context); the concurrency-12 cells were skipped by the bench (engine admitted 0 of 12 during its warmup window). Cold prefill measured 7,221 to 8,233 tok/s from 8K to 128K context.

## Limits

- Two trials per level, no interleaved prefill load. The 4-stream dip to 285-327 tok/s is within run-to-run variance for this engine and should not be read as a scheduling cliff.
- All streams used short prompts, so parked-KV reuse is not exercised here. Long-context concurrency interacts with the 4096 MiB conversation cache and is out of scope for this page.
