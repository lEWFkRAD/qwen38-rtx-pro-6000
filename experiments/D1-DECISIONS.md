# d1 decision-model experiments — October 7–15, 2026

Evidence: [reranking receipt](receipts/d1-reranking.json),
[October 15 visual smoke](receipts/d1-visual.json), and
[computer-use notebook excerpts](receipts/d1-notes.json).

LiquidAI/d1-3B was pinned at
`da1fe36a861f24690f27f622dca1d8688503d113`, with BF16 weight SHA-256
`50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c`.
It selects supplied options rather than generating a long answer, so millisecond
decision latency cannot be compared directly with Qwen tokens per second.

## Retrieval and co-residency

The authored synthetic reranking screen passed 12/12 top-one cases across
30 candidates per query, versus 10/12 for simple term overlap. Median ranking
time was approximately 0.86 seconds including guards; mean warm per-passage
inference was approximately 12 ms. This was not an independent retrieval benchmark.

Three short warmed Qwen samples per condition yielded medians of
284.67 tok/s alone, 284.52 with d1 idle, 256.46 with d1 active, and 280.18
after tensor release. Active d1 reduced observed throughput by about 9.9%.
Minimum sampled free GPU memory was 7,543 MiB. This short single-client
test did not qualify eight busy full-context Qwen sessions plus vision.

## Replay improvements did not transfer to live per-button decisions

| Scope | Qwen-only | Hybrid d1 + Qwen | Interpretation |
| --- | --- | --- | --- |
| Three paired serial rounds over eight recorded observations | 24/24 correct; 0.939 s median decision | 24/24; 0.660 s; 9 d1 accepts, 15 fallbacks | 29.7% lower observed median; latency varied |
| One supervised pair of eight live Calculator clicks | 14.194 s active work | 14.386 s; 1 d1 accept, 7 fallbacks | Replay gain did not carry over |
| Deterministic exact-label execution with d1 checkpoints | Prior per-click baseline 14.194 s | 7.269 s active work | One trial; main saving was removing redundant model decisions |

Both live per-button paths completed all clicks correctly. Cold model/transport
startup was separate and can erase a short-task advantage. The checkpoint
trial used the same prior plan, deterministic control mapping, and three d1
state checks. It did not establish autonomous recovery.

A later supervised unexpected-state check did recover from divide-by-zero,
but also exposed a stale-frame acceptance. Capture hash/epoch/age and single-use
receipts were added outside the model; confidence cannot establish freshness.

## Snake and Excel follow-ups

The GPU Snake demo used exact board state plus supplied action features;
it was not screenshot-only autonomous gameplay. Warm GPU decision work was
about 20 ms, with roughly 94–199 ms live end-to-end examples. A dedicated
legal-action mask and game prompt produced a 60-move, eight-apple rollout
without abstentions in the recorded seed. That does not establish robust
general gameplay.

A subsequent Snake-only compiled-forward path measured 12.525 versus 9.215 ms
on a matched warm fixture (26.4% lower). A live 60-move/nine-apple run reached
8.82 moves/s after warmup; its first move took 6.463 s. This was a separate
game path, not an optimization qualified for ordinary Hermes visual decisions.

The recorded Excel rerun used a synthetic workbook and supervised exact UI
actions: 166.002 s wall time, 64.204 s timed operations, and 101.798 s coordinator
gaps. The nested d1 request total was 6.658 s, including 0.545 s GPU inference.
Blank/table checks passed, but the correct total and chart labels scored
35.98% and 69.91%, below the unchanged 70% gate. Saved formulas and chart values
were independently checked; d1 did not validate every checkpoint.
This was not autonomous spreadsheet execution or a fair end-to-end speed A/B.
The separate Windows CPU Atomic repair is outside the GPU performance scope.

## October 15 visual smoke and remaining bridge issue

Five synthetic screenshots repeated twice yielded **10/10** correct direct
resident-worker decisions. Coverage included payment-target selection, upload
state, a Spanish retry control, a permanent-delete modal, and enabled versus
disabled controls.

Median confidence was 0.9741 (minimum 0.7894). Median model inference was
**27.69 ms**, range 25.71–37.23; separate SSH-client wall time was
**1.548 s**, range 1.301–2.080. Model latency and transport time are distinct.

The Hermes visual-decision tool returned its fallback twice after fresh
captures; no action was executed. The cause in the capture-receipt/tool bridge
was not established. The direct worker pass does not qualify that integration,
coordinate prediction, adversarial robustness, or real-site reliability.
