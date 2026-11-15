# November 14 — localmaxxing.com leaderboard: rank 1 and 2 of 806

Two Verified speed-test submissions put this rig at the top of the published
leaderboard for Qwen3.8-Flash-Next, plus a KV-cache sweep recorded with
engine-counter evidence.

| Run | Decode | Prefill | TTFT | Spec | Status |
|---|---:|---:|---:|---|---|
| [cmwh3v8ey](https://www.localmaxxing.com/en/speed-tests/cmwh3v8ey025fmi01dyj94u4y) | 414.6 tok/s | 1524.0 | 197.5 ms | DFlash2, 6 draft tokens | Verified, rank 1 of 806 |
| [cmwh3ubp5](https://www.localmaxxing.com/en/speed-tests/cmwh3ubp5025cmi01ftfoeso4) | 413.4 tok/s | 1578.9 | 192.5 ms | none | Verified, rank 2 of 806 |

Method: canonical `reasoning-v1` prompt, 256 max output tokens, one warm-up
plus three timed iterations, median reported, measured against the production
Strata 0.1.56 endpoint on port 8002. No engine restart; production served
real traffic throughout.

Context for the numbers: the next published Strata IQ3_S entry is 161.1
tok/s on a multi-GPU rig. Same-card other engines on the site publish vLLM
NVFP4 median 160.5, SGLang NVFP4 156.6, vLLM FP8 about 125.

## KV-cache sweep

Two-step remote probe (pre-warm prefix, then time a streaming request with
the same prefix). Reuse classified only from the timed response's own
counters. At every level from 10K to 80K context, 99.95% of prompt tokens
were served from cache; only the 19 chat-template tokens were re-prefilled.
TTFT stayed under 474 ms at 80K. Raw payload: `kvcache-sweep.json`.

The sweep rows themselves are not Verified-badge eligible on the site: the
probe completion is capped at about 4 output tokens by the engine regardless
of the requested cap, below the 256-token floor. The sweep rows were removed
from the leaderboard to keep it clean; the evidence files stay here.

## DFlash2 note

Engine counters recorded 195 of 271 draft tokens accepted (72%) on the
canonical prompt, yet decode measured the same as plain (414.6 vs 413.4).
At 304 prompt tokens there is nothing for speculation to compound on. This
matches the parked-KV picture: the drafter and cache pay off in long agentic
sessions, not short leaderboard prompts.

## Published report

Full write-up with reproduction commands:
[localmaxxing.com report](https://www.localmaxxing.com/en/models/Qwen/Qwen3.8-Flash-Next/reports/cmwh4ldbr0264mi016zsfwx3s)
and `REPORT.md` in this directory. The same evidence was submitted upstream
to Turbofit as discovery evidence: SouthpawIN/turbofit PR #83.

## Files

- `REPORT.md` — the published report source
- `speed-plain-reasoning-v1.json` — verified run payload (no spec decode)
- `speed-dflash2-reasoning-v1.json` — verified run payload (DFlash2, includes engineTimingsRaw)
- `kvcache-sweep.json` — sweep payload with per-level cacheReuse evidence
