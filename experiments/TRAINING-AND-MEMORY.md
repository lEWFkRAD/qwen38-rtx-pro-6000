# Training and recurrent-memory experiments

Evidence: [LoRA notebook excerpts](receipts/training.json),
[PLE/working-memory excerpts](receipts/recurrent-memory.json), and
[October 11 integration receipt](receipts/living-weights.json), and
[compile-only followup](receipts/living-weights-followup.json).
Unless identified as a local receipt, these are contemporaneous notebook
records rather than newly reproduced training runs.

## September LoRA: real updates, limited gains

Initial work repaired synthetic-data admission, split/lineage checks and exact
grading. Explicit JSON-only prompting raised the four-family development screen
from 39/80 to 80/80; this was **prompt tuning, not weight training**.

Actual Flash-Next attention LoRA followed after a bounded disk-backed PLE loader
avoided materializing the approximately 95.37 GiB BF16 PLE table in host RAM.
The model exposed 456 adapter tensors across 228 attention matrices.
Four successful runs performed 1 + 32 + 128 + 128 = **289 optimizer updates**.
The newer strictly bound receipts cover 288; the first update remains separate
legacy evidence.

The v5 run did 128 new updates from a weight-only warm start, with fresh
optimizer state. The continuation restored 456 AdamW entries and scheduler
step 128, then advanced to global step 256 with 128 new updates. Same-base PEFT
reload matched adapter tensors and checked logits exactly. That does not
establish compatibility with the production inference path.

| Development screen | Base | Checkpoint 128 | Checkpoint 256 |
| --- | ---: | ---: | ---: |
| Paired questions | 20/24 | 21/24 | 22/24 |
| Free-response accounting | 3/4 | 2/4 | 3/4 |
| Free-response math | 1/4 | 2/4 | 2/4 |
| General instruction subset | 4/4 | 4/4 | 4/4 |
| Executed synthetic workflows | 10/10 | 10/10 | 10/10 |

The three free-response subsets total 8/12, 8/12 and 9/12. These rows combine
separately preserved evaluations, not a single rerun. Checkpoint 256 introduced
malformed JSON on an already-wrong rational-equation answer and retained a
paired retained-earnings regression. Checkpoint 128 remained the provisional
development incumbent; **neither adapter qualified for production**.

The v5 128-step run's recorded journal peak was 79.84 GiB active / 80.16 GiB
reserved. Training loss and changed adapter weights prove execution, not
general quality improvement. September 8 added no optimizer run; the proposed
lower-learning-rate continuation remained unfinished.

## September recurrent PLE bridge

A four-update bridge ran against real frozen Qwen and its packed PLE table.
The qualified experimental profile used an HC fallback plus unfused MoE
finalization; it was not the production serving profile. Earlier repeated-logit
instability and a wrongly targeted backend observer were resolved before the
scored screen. Baseline repeats and zero-bridge controls became exact;
removing the trained bridge restored baseline logits exactly.

| Frozen 32-case screen | Baseline | Zero bridge | Trained bridge |
| --- | ---: | ---: | ---: |
| Arithmetic | 4/8 | 4/8 | 5/8 |
| Symbolic state | 1/8 | 1/8 | 3/8 |
| Copy / compact JSON | 8/8 | 8/8 | 5/8 |
| General retention | 8/8 | 8/8 | 8/8 |
| Total | 21/32 | 21/32 | 21/32 |

Three strict gains and three losses produced no net gain. Formatting accounted
for all three losses and one gain. Warm forced continuation cost approximately
47.4 ms baseline versus 49.8 ms trained in a serial eager diagnostic.
This was not a production throughput benchmark.

A subsequent gain sweep over the **same already-seen cases** found 22/32 at
quarter strength, with one gain and no losses. It was a calibration candidate,
not fresh held-out validation. No hard-set confirmation or promotion is
established in the reviewed closeout.

## Working-memory probes: real features, near-chance transfer

The separate event-level recurrent-readout experiment captured 960 histories
at blocks 8, 24 and 40. Dataset splits were 256 train, 64 development, and
128 each for IID, new rules, new names, new wording and longer histories.
It tracked two modulo-eight registers; event labels were excluded from inputs.

An earlier capture timed out and lost in-memory features. The final durable
capture finished in 4,073 seconds with 21 exact identical-prefix audits.
All 36 CPU probe arms completed 300 updates each.

| Probe | Mean IID final exact-pair accuracy | Mean long-history accuracy |
| --- | ---: | ---: |
| RLT-inspired | 1.91% | 1.82% |
| Linear | 1.91% | 1.74% |
| GRU | 2.34% | 1.82% |
| MLP | 1.22% | 2.00% |

Uniform random pair guessing is 1/64 = 1.5625%; this is not a claim that
generated labels were balanced. **No reliable transfer or RLT advantage was
established.** These CPU readout updates are separate from the 289 LoRA
updates and the four-update PLE bridge. No research adapter was promoted.

## October Living Weights work: CPU and compile evidence only

The local October 11 integration receipt pins
`5792b4fbedf5cf97bbc4d45da2f4a9ab01f78907`: 379/379 fresh Linux CPU
tests, 32/32 build steps, and projection ReleaseSafe 20/20.

A later notebook entry pins
`6a8f9d1a55acddcd0d9de28598bb97931825b5b6`: group-32 Q4/Q6 affine
GEMV, dispatch and host-oracle fixtures; 399/399 host tests, 34/34 build
steps, Debug/ReleaseSafe 20/20 each, and an SM120 object compiled with nvcc.
**No CUDA device execution was performed.** Full forward, backward,
optimizer/resume, and full-model Living Weights training remained unimplemented
or unqualified. These checks are not additional GPU training results.
