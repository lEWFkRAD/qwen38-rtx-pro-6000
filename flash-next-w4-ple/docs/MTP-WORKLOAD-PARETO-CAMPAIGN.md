# MTP workload acceptance, draft-depth, and FP8-KV campaign

This campaign prevents a fast short-chat benchmark from standing in for real
production behavior. Native-MTP acceptance is content dependent, so every arm
must report acceptance and latency separately for chat, tool/strict-JSON,
math, and long-retrieval workloads.

## Executed result — 2026-08-27

The production A0 arm (`steps=1`, `top-k=1`, `draft_tokens=2`, rejection
sampling, BF16 target/draft KV) passed all 34 corpus cases. Its runtime identity
was unchanged across the run, and all eligible successful cases carried valid
speculative metrics. One expected missing-metrics entry is the required
over-limit HTTP 400 case.

| Workload | Measured requests | Weighted accept length | Weighted accept rate | Median e2e output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Chat, forced 512 | 9 | 1.904 | 90.3% | 151.387 |
| Tool calls + strict JSON | 10 | 2.020 | 100.0% | 99.840 |
| Exact math | 10 | 2.167 | 94.4% | 35.713 |
| Long retrieval | 2 metric-bearing + 1 expected 400 | 1.774 | 77.1% | 14.817 |

The leak-free 119,735-prompt-token case completed 512 tokens and returned the
independently planted early/middle/late codes. The distinct 130,831-token case
returned its exact blind marker, and a 131,491-token request received the
strict context-limit HTTP 400. The immutable
[`A0 receipt`](../evidence/mtp-workload-a0-1-1-2-bf16kv-20260827.json) and
[`rolling histogram`](../evidence/mtp-workload-rolling-v2-20260827.json)
contain no prompts or response text.

A1 (`2 / 1 / 3`) was then cold-started as a c1-only screen. Before readiness
or any inference request, SGLang resolved both target and draft KV pools to
only 97,664 tokens. Because that is below the configured 131,072-token
context, A1 was rejected and rolled back. A2 was not attempted: a deeper arm
cannot recover the memory/context failure. The machine-readable
[`depth Pareto receipt`](../evidence/mtp-depth-pareto-20260827.json) records
the candidate hashes, memory result, and final disposition.

The exact A0 files were restored and passed a clean cold readiness gate. After
closing idle desktop GPU consumers before sizing, the final target and draft
pools resolved to 146,240 tokens, with 5.562 GB reported ready headroom. The
selected production arm therefore remains A0. The larger final pool is an
environmental sizing result, not a model-argument change.

## Tiny running histogram

The pinned SGLang endpoint exposes per-request speculative counters on a
non-streaming OpenAI chat request when `return_meta_info=true`. Use
[`mtp_workload_histogram.py`](../scripts/mtp_workload_histogram.py) to retain:

- raw completion, verify, correct-draft, and proposed-draft counts;
- weighted accept length and weighted accept rate by workload class;
- 0.25-wide accept-length bins from 1.0 through 4.0;
- recent per-case acceptance and end-to-end output throughput;
- exact-output/tool/JSON grades without retaining response text.

Build the canonical short/long corpus with
[`build_mtp_workload_cases.py`](../scripts/build_mtp_workload_cases.py) against
the exact local tokenizer. The canonical invocation uses the pinned runtime
image, no network, a read-only verified model, a read-only release tree, and a
new writable output directory:

```bash
sudo docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v /absolute/model:/models/flash-next:ro \
  -v /absolute/repo/flash-next-w4-ple:/release:ro \
  -v /absolute/new-run:/output \
  lmsysorg/sglang@sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae \
  python3 /release/scripts/build_mtp_workload_cases.py \
    --model-dir /models/flash-next \
    --output /output/workloads.jsonl
```

The input is JSONL with one object per request. Warmups may be retained in the
immutable receipt with `"measured": false`; they do not enter the summary,
rolling histogram, or campaign minima. Example:

```json
{"case_id":"math-703","workload_class":"math","request":{"messages":[{"role":"user","content":"Return only the product of 37 and 19."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}},"expect":{"status_code":200,"finish_reason":"stop","content_exact":"703","reasoning_empty":true}}
{"case_id":"tool-multiply","workload_class":"tool_json","request":{"messages":[{"role":"user","content":"Use multiply for 37 times 19."}],"tools":[{"type":"function","function":{"name":"multiply","description":"Multiply two integers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false}}}],"tool_choice":"required","temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}},"expect":{"status_code":200,"finish_reason":"tool_calls","reasoning_empty":true,"tool_call":{"name":"multiply","arguments":{"a":37,"b":19}}}}
```

Validate a corpus without issuing inference:

```bash
python3 flash-next-w4-ple/scripts/mtp_workload_histogram.py \
  --cases /path/to/workloads.jsonl \
  --output /unused/validate-only.json \
  --arm eagle-1-1-2-bf16kv \
  --campaign-grade \
  --expected-drafts-per-verify 1 \
  --expected-context-length 131072 \
  --expected-max-total-num-tokens 139072 \
  --expected-max-running-requests 4 \
  --expected-kv-cache-dtype bfloat16 \
  --expected-draft-kv-cache-dtype bfloat16 \
  --validate-only
```

Run it locally on the inference host only after the production idle guard:

```bash
python3 flash-next-w4-ple/scripts/mtp_workload_histogram.py \
  --cases /path/to/workloads.jsonl \
  --output /new/immutable/run/acceptance.json \
  --rolling /new/campaign/rolling-histogram.json \
  --arm eagle-1-1-2-bf16kv \
  --campaign-grade \
  --expected-drafts-per-verify 1 \
  --expected-context-length 131072 \
  --expected-max-total-num-tokens 139072 \
  --expected-max-running-requests 4 \
  --expected-kv-cache-dtype bfloat16 \
  --expected-draft-kv-cache-dtype bfloat16 \
  --base-url http://127.0.0.1:8002/v1
```

The collector refuses to overwrite an immutable receipt and refuses a
non-loopback endpoint unless explicitly allowed. The rolling file contains
counters and at most 64 recent measured samples per arm/class; it is an
operational view, not the immutable evidence source. `--validate-only` checks
the static corpus schema, model binding, and measured-case counts; it cannot
qualify runtime token lengths. A campaign qualifies only after the run receipt
contains one metric-bearing passed witness with `119424 <= prompt_tokens < 130500`
and `completion_tokens >= 512`, a distinct passed witness with
`prompt_tokens >= 130500`, and a measured long-retrieval HTTP 400 whose strict
error body proves the 131,072-token context rejection.
Use `--expected-drafts-per-verify 1`, `2`, or `3` for A0, A1, or A2 so the
receipt rejects telemetry from a differently normalized runtime. Any scheduler
retraction also invalidates speculative acceptance telemetry.

Campaign-grade runs bind `/get_server_info` both before and after inference to
the exact model, instance startup fingerprint, 131,072 context, resolved token
pool, maximum concurrency, target/effective-draft KV dtypes, EAGLE depth,
top-k, draft width, and rejection-sampling state. The two identity hashes must
match. Change the expected pool and dtype arguments to the measured values for
each FP8 or draft-depth arm; an arm label is never runtime evidence. The client
uses no API key by default and never inherits `OPENAI_API_KEY`. If a secured
endpoint is deliberately used, name its dedicated secret explicitly with
`--api-key-env SGLANG_CAMPAIGN_API_KEY`.

Recommended minimum evidence per arm:

- chat: two `measured:false` warmups, then nine unique-prefix forced 512-token
  c1 decodes;
- tool/strict JSON: at least ten exact graded calls spanning tool selection,
  argument serialization, and schema-constrained JSON;
- math: at least ten exact-answer cases;
- long retrieval: a leak-free witness with prompt tokens in `[119424,130500)`
  and at least 512 completion tokens, plus a distinct exact witness at 130,500
  prompt tokens or above. The long cases contain many verification rounds, so
  raw counters matter more than the small request count.

Never promote from the all-workload average alone. Publish each class, and
require every exact-output/tool/JSON/retrieval case to pass.

## Legal native-EAGLE depth sweep

On this pinned Qwen4-Exp path, PLE speculation requires `top-k=1`, and
rejection sampling also rejects `top-k>1`. With `top-k=1`, SGLang normalizes
draft width to `steps + 1`. The valid sweep is therefore:

| Arm | Steps / top-k / draft tokens | c1 verify width | c4 verify width |
| --- | ---: | ---: | ---: |
| A0 control | `1 / 1 / 2` | 2 | 8 |
| A1 | `2 / 1 / 3` | 3 | 12 |
| A2 conditional | `3 / 1 / 4` | 4 | 16 |

Do not test `top-k=2` or describe `steps=2,draft=4` as an independent arm; it
is invalid or normalized by this runtime.

For A1, change only:

```text
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
```

Screen with c1 first. Run A2 only if A1 improves a fresh, same-window A0 c1
median by at least 5%, preserves the lower tail and every exact gate, keeps both
KV pools at least 139,072, and leaves at least 2 GiB stressed free VRAM. Take
only non-dominated c1 arms to c4. At c4, reject a candidate whose median falls
more than 3% or whose lower decile falls more than 10% versus matched A0.

c4 is a veto, not the tuning target: verify width rises from 8 target tokens
per round at A0 to 12 at A1 and 16 at A2. A deeper arm that wins c1 but loses
c4 should be published as a latency-specialized Pareto point, not hidden or
installed as one universal winner.

## FP8 KV is patch-gated

Do **not** add `--kv-cache-dtype fp8_e4m3` to the pinned runtime as a flag-only
experiment. In the deployed QSA chunk-prefill path, the second 1,024-token
chunk gathers FP8 K/V from cache and reaches a BF16-by-FP8 Triton dot without a
cast or scale. Pinned Triton 3.7 rejects that mixed pair, so a flag-only arm is
expected to fail before long-context retrieval can be evaluated.

A conditional FP8-KV arm first needs a focused QSA patch that:

1. dequantizes cached FP8 K/V into the compute dtype for chunk-prefill;
2. applies K/V scales consistently on cache write, prefill, and XQA decode;
3. preserves BF16 QSA index buffers and FP32 Mamba state;
4. fails closed on an unsupported fallback; and
5. adds Qwen4Exp scale loading before any non-unit calibrated scale is used.

At the campaign A0 pool size of 139,072 slots, target plus draft FP8 KV would save about
1.724 GiB. At the existing requested 163,072-token cap, the expected net gain
after unchanged QSA side state is about 1.41 GiB, increasing pool excess over
the 131,072 request limit from 8,000 to 32,000 tokens. This is concurrency
headroom, not permission to raise the qualified per-request context.

After compile/numeric parity at production QSA/XQA geometry, stage target-FP8
with draft-BF16 first, then both FP8 at 139,072, and only then both FP8 at the
163,072 cap. A 2,048+ prompt is the first smoke gate. Promotion still requires
18/18 quality, exact tool/JSON/math, the leak-free `[119424,130500)` prompt plus
512-completion witness and distinct 130,500+ retrieval witness,
over-limit HTTP 400, the workload-class acceptance histogram, matched c1/c2/c4,
and zero restart/OOM/fatal/Xid evidence.

## Pareto receipt

Publish every non-dominated arm with these columns:

```text
arm, KV dtype, steps/top-k/draft, resolved target/draft pools,
stressed free VRAM, c1 median/p10, c2 median/p10, c4 median/p10,
chat/tool_json/math/long_retrieval weighted accept length,
per-class e2e latency or output throughput, exact gates, disposition
```

For each measured request, let `C` be completion tokens, `V` verification
rounds, `A` accepted drafts, and `P` proposed drafts. The collector publishes
weighted accept length `C/V` and accept rate `A/P`, requires `P = V *
(draft_tokens - 1)` for these fixed top-k-1 arms, and rejects any scheduler
retraction or inconsistent counter/histogram.

The production choice must survive tools and exact output. A clean-chat c1
headline is supporting evidence, not the production number.
