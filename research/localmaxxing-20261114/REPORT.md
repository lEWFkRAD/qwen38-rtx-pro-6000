# Qwen3.8-Flash-Next IQ3_S on one RTX PRO 6000: fastest published decode, and near-total KV reuse at 80K context

## Setup

- Model: Qwen3.8-Flash-Next, GGUF from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, quantization IQ3_S
- Engine: Strata 0.1.56 (Niko1221/Strata), eight-slot serving, parked-KV conversation cache
- Speculative decoding: DFlash2 drafter (qwen3.8-flash-next-dflash2.q8.pack), 6 draft tokens, min-p 0.5
- Hardware: one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB), Intel Core Ultra 9 285K, 64 GB RAM, Ubuntu 24.04.5
- Endpoint: local OpenAI-compatible server on port 8002, measured from the same machine

## Methodology

Speed tests used the canonical reasoning-v1 prompt with 256 max output tokens, one warm-up and three timed iterations, median reported. Both speed-test runs are Verified on the site (rank 1 and 2 of 806 runs for this model).

KV-cache sweeps used the lmx two-step remote probe: pre-warm a long prefix, then time a streaming request with the same prefix. Reuse is classified only from the timed response's own counters (`usage.prompt_tokens_details.cached_tokens`), never from warm-up snapshots.

## Results

| Metric | Plain decode | With DFlash2 |
| --- | ---: | ---: |
| Decode | 413.4 tok/s | 414.6 tok/s |
| Prefill | 1578.9 tok/s | 1524.0 tok/s |
| TTFT | 192.5 ms | 197.5 ms |

KV-cache sweep (timed-request counters):

| Context | Cached / prompt tokens | Uncached | TTFT |
| ---: | ---: | ---: | ---: |
| 10,000 | 10,041 / 10,060 | 19 | 110 ms |
| 20,000 | 20,041 / 20,060 | 19 | 316 ms |
| 40,000 | 40,041 / 40,060 | 19 | 209 ms |
| 80,000 | 80,041 / 80,060 | 19 | 474 ms |

Speculative decoding evidence from the engine's own counters: 195 of 271 draft tokens accepted (72% acceptance rate) on the canonical prompt.

## Observations

- **Finding:** 414.6 tok/s is the fastest published run for this model across all 806 submissions. The next Strata IQ3_S entry is 161.1 tok/s on a multi-GPU rig, so a single RTX PRO 6000 runs this quant about 2.6x faster than the next published setup.
- **Finding:** On the same card, other engines publish far lower medians: vLLM NVFP4 160.5 tok/s, SGLang NVFP4 156.6, vLLM FP8 about 125. Strata with IQ3_S is roughly 2.5x the same-card alternatives.
- **Finding:** The parked-KV cache reuses 99.95% of the prompt at every context length tested, up to 80K tokens. Only the 19 chat-template tokens are re-prefilled per turn, which is why TTFT stays under half a second at 80K context. Other published runs on this model show multi-turn TTFTs of 5 to 53 seconds, which is cold prefill every turn.
- **Caveat:** On the short 304-token canonical prompt, DFlash2 measures the same as plain decode (414.6 vs 413.4). Speculation gains concentrate in longer agentic contexts, and the leaderboard prompt is too short to show them.
- **Caveat:** The KV sweep rows themselves are not Verified-badge eligible. The probe completion is capped at about 4 output tokens by the engine regardless of the requested cap, below the 256-token verification floor. The cache-reuse numbers come from the engine's own usage counters and stand on their own.

## Reproduction

```bash
# Speed test (verified run)
LMX_MODEL_API_KEY=*** lmx speed-test run strata \
  --mode remote --base-url http://127.0.0.1:8002 \
  --served-model qwen3.8-flash-next \
  --hf-id ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --quantization IQ3_S --hardware hardware.json \
  --canonical-prompt reasoning-v1 \
  --spec-method dflash2 --spec-draft-model qwen3.8-flash-next-dflash2.q8.pack \
  --spec-num-tokens 6

# KV-cache sweep
LMX_MODEL_API_KEY=*** lmx kvcache run strata \
  --mode remote --base-url http://127.0.0.1:8002 \
  --served-model qwen3.8-flash-next --hf-id Qwen/Qwen3.8-Flash-Next \
  --quantization IQ3_S --hardware hardware.json \
  --levels 10000,20000,40000,80000 --output-tokens 256
```
