# C8 measurement and reproduction — 2026-10-22

The public summary is in [results-2026-10-22.json](evidence/results-2026-10-22.json).
The executable reproduction aid is [benchmark_c8.py](scripts/benchmark_c8.py).

Use these labels:

- ~494 aggregate OUTPUT tokens/s while all eight sessions generate.
- ~61.7 OUTPUT tokens/s per session at C8.
- +48.0% matched steady C8 decode versus engine 0.1.42.
- ~421 aggregate request tokens/s on the candidate, admission included.

## Configuration

One RTX PRO 6000 Blackwell, 96 GB, 450 W limit; Qwen3.8-Flash-Next IQ3_S;
eight resident slots; 131,072-token context per slot; full INT8 KV GPU
residency for main context and batch slots; 24,576 GPU-resident expert slots.
The candidate engine/vision bundle is 0.1.46, CUDA 13.0; the Python/API runtime
remains 0.1.42 with its existing compatibility patch. Both steady arms use
the same model assets, capacity, KV policy, scheduling share 0.75 and request
auxiliary CPU setting 0. The JSON records the saved candidate environment.

## Steady C8 method

Launch eight greedy SSE requests together, thinking disabled, limited to 512
output tokens each. Their distinct counting sequences start at 1, 101, 201,
301, 401, 501, 601 and 701. Check the first 40 numbers and full-output hashes.
All 16 candidate texts match their baseline counterparts across two rounds.

Each common window starts 250 ms after the latest first-content event and
ends 250 ms before the earliest last-content event. Count each stream's
output directly inside that interval. Accepted rounds have exactly 512
content events matching 512 usage tokens and a length finish. EOS may count
in usage without visible content; that case is recorded separately and is
not accepted as a full counting canary. Characters and words are not tokens.

Sample activity about every 100 ms. Qualify windows only with eight running
requests, no extra queue/prefill/admission/outside-slot work, no completed
request, and adequate coverage. All four published windows have no observed
competition and maximum sample gaps below 105 ms. Sampling cannot rule out
sub-sample transients. The local harness supports separately proven gated
queues; none occurred in these windows. The public harness uses a stricter
zero-queue rule and needs no process access.

There are TWO qualified windows per arm; eight correlated batch members do
not create 16 independent speed samples. Median reported session rates are
41.686 and 61.7095 tokens/s (+48.034%); aggregate medians are 333.4865 and
493.6775. Retained original rounding can cause tiny arithmetic differences.

## Whole-request scope

Divide each round's 4,096 output tokens by request launch through last
response completion. This includes admission, streaming and HTTP overhead;
it excludes startup, model loading and pre-launch idle waiting. Candidate
rounds are 409.603 and 433.133 tokens/s; their median is 421.368. Both are
clean. Only one baseline whole run is clean: the other has newly arriving
work at its final snapshot, after the accepted steady interval. The 48%
claim therefore does NOT describe controlled whole-request speed or latency.

## Reproduction

Use Python 3 and an existing Strata server with GET /metrics and streamed
POST /v1/chat/completions including final usage. Set STRATA_API_KEY in your
environment if authentication is enabled. Run identical options under each
recipe; deploy recipes separately through your normal process:

```bash
python3 strata-c8/scripts/benchmark_c8.py --output baseline.json --rounds 2 --max-tokens 512 --aux-cpus 0 --model qwen3.8-flash-next-custom-iq3-s
python3 strata-c8/scripts/benchmark_c8.py --output candidate.json --rounds 2 --max-tokens 512 --aux-cpus 0 --model qwen3.8-flash-next-custom-iq3-s --compare baseline.json
```

Defaults: loopback port 8002, model alias qwen3.8-flash-next-custom-iq3-s (matching the example config). Override with
--base-url and --model. Use fresh output filenames. Each round waits at most
60 seconds for idle; default total budget is 180 seconds. Existing work is
never interrupted. No idle start means no inference for that round. The
script saves hashes, numeric timing/activity and canary results, not keys,
endpoints or raw generated text. Incomplete/contaminated arms cannot produce
a qualified comparison gain. Admission order, cache reuse and load may vary.

This sanitized harness preserves the exact prompts, generation options,
SSE accounting and window method. It omits private configuration, file logs,
process inspection and raw output storage. It passed syntax checks and nine
focused offline comparison/privacy tests; it has not been executed against a
live GPU server. Published results came from the original local harness. It is
a reproduction aid, not a model download or deployment tool.

## Separate mixed result and limits

Seven 1,024-token decodes alongside an eighth cold 64,993-token prompt
produce 117.493 versus 178.230 combined tokens/s DURING cold intake, with
zero prompt reuse; cold TTFT is 13.643391 versus 13.824827 seconds. This is
one qualified interval per arm, a different workload. Do not divide by eight
or merge with steady C8. The public harness implements the steady test only.

Additional finite checks cover six baseline-exact arithmetic/code/JSON/
marker/long-context/continuation outputs, simultaneous tool calls, vision,
disconnect/release and batch cancellation with seven baseline-exact survivors
and exact recovery. These do not establish general quality or long-term
crash stability. Counting can reuse experts more than varied real chats;
long prompts, histories, tools, images and mixed traffic can have other rates.

The JSON also preserves exclusions: the first protocol attempt had an extra
external request; the first cancellation attempt hit the benchmark client's
short read timeout during other long-prompt work. Clean confirmations passed.
Packed-Q6 (-1.005%) and PDL (+0.799%, noise scale) trials were reverted and
are absent from the saved recipe. These are not silent successful trials.

## Compact figure footnote

Eight distinct synthetic counting prompts; 512 output tokens/session; two
matched common decode windows per arm. Steady rates exclude admission. The
48% gain applies to matched steady C8 only; real workloads vary.

## Additional profile details

Both arms used the GSQ-RCO IQ3_S checkpoint with per-layer runtime projection
enabled (strength 1.0, layers 4–44). Every measured C8 stream reported zero
drafted and zero accepted draft tokens. The native drafter was configured,
but this C8 result measures batched target decode, not batch MTP.
Prompts were 51–53 tokens; first rounds were cold and second rounds had small
prompt-cache reuse in six baseline and five candidate streams.
