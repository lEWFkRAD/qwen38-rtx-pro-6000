# KV streaming and slot count — October 7, 2026

Evidence: [KV comparison](receipts/kv-streaming.json) and
[four-versus-eight slots](receipts/four-slot.json).

## Stream KV while retaining the full context window

Adding `--kv-resident 32768` retained eight slots, 131,072 context, the same
IQ3_S model/vector and binary, INT8 KV, GPU vision, all 24,576 GPU experts,
PLE in RAM, and the no-prefill-borrow setting. Older KV moves to pinned host
RAM; the configured context is not reduced to 32K.

| Metric | Baseline | KV resident 32,768 |
| --- | ---: | ---: |
| Solo end-to-end median tok/s | 182.74 | 185.97 |
| Eight-request aggregate median tok/s | 272.19 | 270.31 |
| Cold approximately 128K batch last finish, s | 159.555 | 163.470 |
| Cached followup last finish, s | 36.014 | 35.986 |
| Minimum sampled available host RAM, GiB | 23.60 | 9.41 |
| Minimum sampled free GPU memory, MiB | 7,625 | 18,247 |

Startup free VRAM increased from 14,773 to 25,395 MiB: **10,622 MiB / 10.37 GiB**.
The memory moved into host RAM; it did not disappear. Whole-host swap rose
during loading, while the service retained its no-swap limit.

Both conditions passed 32/32 concurrent JSON/tool checks, all eight long
retrievals and cached followups, with eight decoding slots observed. Three
short-prompt rounds and sequential profiles provide bounded evidence, not a
general speedup or broad quality-equivalence claim. This profile was selected
on October 7.

## Four slots trade maximum batch throughput for host RAM

Both profiles below used the same 32K resident-KV window and 128K context.

| Metric | Eight slots | Four slots |
| --- | ---: | ---: |
| Loaded service proportional RAM, GiB | 43.513 | 37.365 |
| Available host RAM after load, GiB | 14.679 | 21.113 |
| One submitted request median tok/s | 187.92 | 184.95 |
| Four submitted, aggregate median tok/s | 233.17 | 232.63 |
| Eight submitted, aggregate median tok/s | 271.06 | 232.42 |

Service RAM saving was **6.148 GiB**. The four-slot profile queues excess
requests. Both profiles passed 32/32 JSON/tool checks and four long-context
retrievals plus cached followups with four simultaneous long decoders.
The four-slot sampled minima were 18.38 GiB available RAM and 27.52 GiB free VRAM.

The experiment first restored eight slots. A later same-day selection activated
the tested four-slot profile; an October 13 observation still recorded four
slots on a later Strata runtime. The test report's earlier "candidate only"
wording is therefore historical. No game/FPS qualification was performed.
