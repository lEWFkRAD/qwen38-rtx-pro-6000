# LFM2.5-24B-A2B — October 14, 2026

Evidence: [smoke receipt](receipts/lfm-smoke.json),
[long accounting suite](receipts/lfm-accounting.json),
[local rerun progress](receipts/lfm-rerun-progress.json), and
[later notebook closeout](receipts/lfm-notes.json).

## Fast short smoke

Official QAD Q4_0 GGUF revision
`0912d33141f77e10705345a6732bfd83f0252fdc`,
SHA-256 `d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5`,
ran with llama.cpp b11559 / `f2cc7282c`, CUDA 12.8, full GPU offload,
8,192 context and one slot.

All 14 synthetic smoke checks passed: arithmetic, structured JSON, deduplication,
instructions, logic, quoted injection, missing evidence, a simple code fix,
sorting, dependency planning, 3,629-token retrieval, multi-turn correction and
two native tool-call stages. No quality response was truncated.

Three capped 512-token speed samples ranged from 166.8 to 357.8 generated
tok/s, median **320.1**. These include reasoning tokens and are not useful
final-answer throughput. The speed generations intentionally hit their cap,
unlike the quality checks. Qwen and games shared the GPU. Model process
allocation was 13,772 MiB. Neither full 131K nor concurrent load was tested
in this smoke.

## Initial longer accounting suite did not qualify

Synthetic bank reconciliation, AP proposals and month-end close used two fixture
variants plus budget/tool controls. Sixteen scored attempts produced 209 model
turns, 308 tool calls and 541,195 generated tokens.

| Arm | Attempts | Submitted | Exact | Context / response cap |
| --- | ---: | ---: | ---: | --- |
| Baseline | 6 | 0 | 0 | 32K / 8K |
| Larger response | 1 | 0 | 0 | 32K / 16K |
| Read-only SQL assisted | 6 | 0 | 0 | 32K / 16K |
| Larger SQL control | 3 | 1 | 0 | 64K / 32K |
| **Total** | **16** | **1** | **0** | |

There were 14 truncated responses, one time-budget exhaustion and one submitted
workpaper. That bank reconciliation, without compaction, reported $15,541.09
bank and $17,536.08 books instead of $97,622.05 on both sides. It omitted book
adjustments, reversed the bank-error sign, missed a duplicate, and treated
export duplicates as book corrections.

No-compaction subset: 13 attempts, one submission, zero exact passes. Missing
submissions are completion failures; they are not measured numeric answer
accuracy. Repeated fixture/budget variants are not 16 independent task families.

The harness had material limitations: compaction retained task/notes/page index
rather than recent exchanges, the calculator could not execute general Python,
and truncation ended attempts. An early SQL startup/total-token-count defect
was excluded. Corrected SQL trials used prompt-token counts. Time limits
checked between requests could overshoot during the final request.
The result applies to this quantization/runtime/tool/prompt setup, not LFM
universally or a matched comparison with Qwen.

## Corrected rerun: partial, not a completed six-job suite

The revised harness provided persistent isolated Python, retained visible/tool
history without compaction, removed completed reasoning from later requests,
and added direct artifact submission plus source-based consistency gates.
It configured 131,072 context, 65,536 output cap, 16,384 reasoning budget,
and 30-minute/100-turn/250-tool budgets.

The later notebook closeout reports AP seed 137 exact in 84.1 seconds /
13 turns, bank seed 137 at the 1,800.9-second budget after 99 turns with
no accepted submission, and close interrupted when testing was stopped.
An interruption is not a scored failure.

The available local bank progress file is earlier/different: 1,115.38 seconds,
41 turns, no structured final result. It does not independently prove the
later 99-turn closeout. Both snapshots are retained and clearly attributed;
there is no fabricated aggregate rerun pass rate.

## Resident LFM worker plus Qwen reviewer

A subsequent owner-selected configuration co-resided LFM QAD Q4_0 with the
Qwen 27B Heretic NVFP4 reviewer. The notebook records real readiness canaries
and approximately 19.2 GiB free GPU memory in one shared-load snapshot.

LFM used four slots and 131,072 configured context per slot with
`ngram-simple` speculation. **Native LFM MTP/DSpark was not enabled.**
The Qwen reviewer used native EAGLE 2/1/3, FP8 KV, 262,144 maximum context,
270,336 shared KV tokens and two running requests. Shared KV capacity does
not prove two simultaneous full maximum-length histories.

Operational selection is separate from the accounting evaluation's outcome.
Full-context co-load stress and the wider rollout were not established by the
recorded canaries. This publication did not change or verify those services.
