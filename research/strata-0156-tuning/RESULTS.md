# Stock Strata 0.1.56 serving benchmark — 2026-11-03

The selected `fused_prefill` serving profile improved measured throughput against the retained, production-qualified official 0.1.50 engine while preserving eight fully resident 131,072-token slots. This campaign used stock native attention, downloaded official binaries and serving-setting changes. It used no LSA, trained no model weights, built no engine code, and deployed nothing. The code, profiles and evidence are packaged with this report.

The native comparison completed three interleaved pairs, `A,B / B,A / A,B`, with the profile frozen before confirmation. A is official 0.1.50 with the approved serving settings; B is official 0.1.56 with the selected settings. Every arm passed the infrastructure/capacity checks. The finite quality fixtures had no new pass/fail regression, but **both engines still failed two arithmetic cases**. This is not a zero-error quality result.

## Fixed configuration and selected settings

Hardware was one RTX PRO 6000 Blackwell 96 GB, driver 615.71.09, with the existing 550 W policy. Both arms used the same Qwen3.8-Flash-Next IQ3_S assets, tokenizer/template, MTP assets and control vector: scale 1.0, project mode, per-layer direction, layers 4–44. All 24,576 expert slots remained GPU resident. Ordinary INT8 KV remained resident for all eight slots: requested `--max-context 131072 --kv-resident 131072 --batch 8`, with engine `kv_resident=0` denoting full residency.

The retained settings included `--prefill 32768`, `--spec 6 --spec-min-p 0.5`, prompt lookup, `--no-prefill-borrow`, `--conversation-cache-mib 0`, and `STRATA_BATCH_DECODE_SHARE=0.75`. The final profile made no argument overrides. It explicitly enabled the nine parallel-stack settings plus GDN and fused prompt experts:

```text
STRATA_CKPT_POOL=1           STRATA_CKPT_POOL_READY=4
STRATA_SLOT_CKPT_MOVE=1      STRATA_SLOT_COPY_DIRECT=1
STRATA_SLOT_COPY_BATCHSYNC=1 STRATA_BATCH_GDN_GROUPS=1
STRATA_BATCH_ATTN_ROWS=1     STRATA_BATCH_SELF_COMMIT=1
STRATA_QSA_SLOTS=1
STRATA_GDN_TC=1             STRATA_GDN_CONVL2=2
STRATA_PF_FUSED=1
```

The screen order was version-only, nine parallel settings, GDN defaults, then fused prefill. Four of the nine parallel settings were already enabled in the retained baseline. GDN/fused paths change floating-point rounding, so this profile required task-level quality checks; it was not assumed bit-identical. The reported final gain combines the engine version and selected settings and cannot be attributed to one switch.

Official 0.1.56 decoder, vision encoder and matched CUDA 13 libraries came from the signed public release, build `06698c674673+cuda13.0`, source `06698c674673008395ba6e971580538d6b1ea6e0`. Signed feed/component checks and independent OpenSSL verification were recorded. The baseline decoder SHA256 was `ae49d5da81b9b65d50af1108c4b3f8b289ee198d9992d66268b7e885cf1a4776`; candidate SHA256 was `21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788`. API tests retained the installed Python wrapper 0.1.42. See [VALIDATION-PROFILE.json](VALIDATION-PROFILE.json) and [official bundle verification](receipts/official-bundle-verification.json).

## Native confirmation

The fresh heldout set was frozen at seed `202611391` and reused across the three pairs. Solo and eight-stream generation each reached the requested **512 output tokens per stream**. These native generation prompts were short: solo 104 tokens; batch prompts 104–106 tokens. They are distinct from the approximately 4K API workloads below.

“Whole request” includes prompt/admission time through completion. “Steady” uses observed output-token arrivals within a trimmed common interval while every measured stream is live. It counts all actual streams; it is not eight times a solo rate. Every C8 common window retained 481–483 observed tokens per stream. The table gives reported medians; gains are calculated from those medians.

| Native measurement | Official 0.1.50 | Selected 0.1.56 | Change |
|---|---:|---:|---:|
| Eight streams, whole-request output tokens/s | 369.72 | 412.60 | +11.60% |
| Eight streams, steady aggregate output tokens/s | 433.13 | 457.28 | +5.58% |
| Solo, steady output tokens/s | 220.54 | 239.01 | +8.38% |
| Solo, whole-request output tokens/s | 209.31 | 218.51 | +4.39% |
| 32K retrieval request wall time | 4,462.0 ms | 4,019.5 ms | 9.92% shorter |
| 125K retrieval request wall time | 16,869.5 ms | 14,919.3 ms | 11.56% shorter |

All three paired steady-throughput ratios favored the selected profile: C8 `1.0454, 1.0625, 1.0592`; solo `1.0960, 1.0794, 1.0842`. The retrieval requests contained 32,076 and 125,077 prompt tokens and passed exact marker recovery in every arm. Their wall times include the answer, rather than isolating only prefill kernels. Evidence: [NATIVE-CONFIRMATION.json](NATIVE-CONFIRMATION.json), [full native report](receipts/final-heldout-r1/report.json), and its frozen fixtures/source snapshots.

The earlier October 22 C8 figure of 493.7 tokens/s and prior LSA figures used different engines, workloads or accounting. They do not provide an absolute-rate ranking against this run; the gains here use same-day matched controls. This campaign retained 131,072-token capacity and makes no longer-context claim.

Both engines failed `known_code_208`, answering **188** instead of **208**. On the fresh `code_exact` fixture, the correct sum of squares was **670**; the baseline answered **594**, and the selected profile answered **590**, consistently across the three pairs. These remain explicit failures. Canary, JSON and long-context marker cases passed; no previously passing case became failing. Finite synthetic fixtures do not establish general quality equivalence. Open prose/code continuations followed different generated trajectories, so equal token budgets do not imply identical generated text.

## Retained-wrapper API spot and features

The completed API throughput run used the same saved approximately 4K synthetic inputs across both arms, one solo 256-token response followed by eight synchronized 256-token streams. All streams reached their cap and the API metrics observed all eight engine slots active. The production sampling block was unchanged (`experimental_speed_projection=true`); request sampling values were omitted, and thinking was explicitly disabled to match the frozen inputs. Counts came from final SSE usage, never chunk counts.

| API measurement | Baseline | Selected profile | Change |
|---|---:|---:|---:|
| Eight clients, whole-request output tokens/s | 178.85 | 221.62 | +23.92% |
| Solo, whole-request output tokens/s | 140.55 | 148.65 | +5.76% |
| Solo first visible-content latency | 1,042.8 ms | 838.1 ms | 204.7 ms shorter |

This was **one API spot per arm**, not three paired API rounds. It includes the real wrapper, queue/admission and transport time and is not interchangeable with the native steady rate. Its inputs were already used by the earlier native screen, so it makes no new heldout claim. Evidence: [API throughput r3](receipts/api-throughput-final-r3/report.json).

Prompt reading/admission contributes more relative cost to the API's approximately 4K-input/256-output workload than to native short-input/512-output generation. The API's +23.92% and native whole-request +11.60% gains therefore describe their respective matched workloads, rather than one interchangeable decoder speedup.

The final candidate API feature check **passed all five cases**: unauthenticated status rejected with HTTP 401, exact text marker, forced tool call, high-reasoning `17×19=323`, and a fresh synthetic green-square image. Process-bound readiness, authentication, full capacity, matched vision, cleanup and absence of native LSA were checked. This is a bounded compatibility smoke test, not broad reasoning or vision evaluation. Evidence: [API features](receipts/api-features-final-r1/report.json).

## Mixed-load tradeoff

One completed diagnostic per arm established four generating streams, then submitted a fresh 32,076-token joining request. All four streams remained active until the joiner completed; marker recovery and cancellation/drain passed.

| Mixed-load measurement | Baseline | Selected profile |
|---|---:|---:|
| Longest observed silent interval during the join | 4,184 ms | 3,655 ms |
| Joining request first-token latency | 4,146 ms | 3,642 ms |

The selected profile still caused a multi-second pause during a long prompt read. A bounded additional `interactive8k` diagnostic changed only `--prefill` from 32,768 to 8,192, retaining decode share 0.75. It passed this diagnostic's correctness/lifecycle checks and reduced the longest pause to **981 ms**, but increased joiner first-token latency to **5,966 ms**. It was rejected as the default and did not receive full paired qualification. Evidence: [mixed r2](receipts/mixed-final-r2/report.json), [8K diagnostic](receipts/interactive8k-mixed-screen-r1/report.json).

## Corrections, limits and final state

Early collectors needed corrections for normal native admission metadata, private configuration path resolution, and TIME_WAIT-aware loopback binding. Their failed receipts remain unchanged and are not treated as model or performance results. See [HARNESS-CORRECTIONS.md](HARNESS-CORRECTIONS.md). CPU tests separately checked token accounting, strict JSON, finite timings, protocol/lifecycle and library identity.

The numbers describe this GPU, quantization, control vector, eight-slot allocation, selected settings, finite prompts and output budgets. Three native pairs are limited evidence, not a statistical confidence claim; the API and mixed-load checks are single diagnostics. Further tuning after consuming this heldout set needs a new heldout set.

The final guarded state recorded no GPU compute processes and no private credential files. Production and Sunshine remained inactive with MainPID 0. All three production configuration/contract hashes, the maintenance HOLD and recovery attempt history were unchanged. No service, route, recovery setting, model weight or production deployment was changed by this campaign. See [final-state.json](receipts/final-state.json). Benchmark work is complete; production remains paused.
