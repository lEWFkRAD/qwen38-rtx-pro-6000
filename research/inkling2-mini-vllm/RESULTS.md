# November 21 — Inkling 2 Mini on one RTX PRO 6000 (vLLM)

vLLM serving Inkling 2 Mini (NVFP4 + DSpark, 262K max context) on the
single-96GB Blackwell. Reproduced on the production workstation; receipts are
the same evidence submitted upstream in Turbofit PR #118 (vLLM profile for
`1x96`). Reproduced as external evidence; not re-measured on this repo's box
by this PR.

## Configuration (current, S7-prob)

- vLLM image `vllm/vllm-openai:inkling-2-mini`, V1 engine, `--served-model-name inkling2-mini`
- 262,144 max context; `--gpu-memory-utilization 0.60`; `--max-num-seqs 16`; `--max-num-batched-tokens 8192`
- FP8 KV cache (load-bearing for 262K on one 96 GB card; BF16 KV left less headroom)
- Prefix caching enabled; inkling reasoning and tool-call parsers; `--enable-auto-tool-choice`
- DSpark spec 7, probabilistic sampling (sweep winner was S7-prob-CG128 at accept 3.14; S7-prob at accept 2.76 is the recommended default)
- Cold prefill is vLLM's strength: ~21.8K tok/s at 22K prompt, ~9.9K tok/s at 101K prompt

## Evidence

| Metric | Value |
|---|---:|
| C1 mean | 246.0 tok/s (prose 156.5 / code 372.4 / list 224.7 / agent 230.6) |
| C1 solo (C1 receipt) | 168 tok/s, TTFT 32 ms |
| C4 aggregate | 274.5 tok/s (per-stream 152.8) |
| C8 aggregate | ~429 tok/s |
| Cold prefill | ~21.8K tok/s @ 22K prompt (11.1 s wall, 1.0 s prefill); ~9.9K tok/s @ 101K prompt |
| Tool calls | 5/5 |
| Spec 0 baseline | 167.7 tok/s C1 mean, 265.4 aggregate C4 |

Sweep: S7-prob-CG128 (accept 3.14, 245.5 mean, 254.5 agg C4) is the best
sweep result; S7-prob is the recommended default (accept 2.76). Spec 0 is
167 tok/s C1 and 265.4 aggregate C4 baseline.

## Interpretation for this rig

Same-card Strata/Basalt production remains the reference for Qwen3.8
Flash-Next (414.6 tok/s leaderboard, 262K C8 context). Inkling 2 Mini via vLLM
is the fast-prefill alternative: decisive cold-agentic-prompt advantage
(~21.8K tok/s prefill vs Strata/Basalt's ~1.5K tok/s), at the cost of a lower
C1 decode (~246 tok/s mean vs 414 tok/s for Qwen3.8 on this card). FP8 KV is
required for 262K context on one 96 GB card.

## Files

- `runtime-profile.yaml` — the copyable vLLM profile (schema: turbofit.runtime-profile/v1)
- Receipts (above): Turbofit PR #118, `research/vllm-inkling2-mini-rtx-pro-6000-20261121.txt` and `research/vllm-inkling2-mini-sweep-20261121.jsonl`
