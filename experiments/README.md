# RTX PRO 6000 experiment archive

This archive closes the gap after the September 2, 2026 publication through October 15, 2026. It covers inference, memory placement, training, recurrent-memory probes, and small decision models on the same 96 GB Blackwell workstation GPU.

These are dated experimental records. A historical promotion is not an instruction to deploy that configuration today. No benchmark was rerun or serving configuration changed to assemble this archive.

| Period | Experiment | Main finding |
| --- | --- | --- |
| Sep 4–23 | [SGLang, NVIDIA checkpoint, GLM and personal 27B](SEPTEMBER-SERVING.md) | Earlier first-response delivery improved; higher GLM token speed failed useful-answer quality |
| Sep 6–14; Oct 11 | [LoRA, recurrent PLE bridge, working memory and Living Weights](TRAINING-AND-MEMORY.md) | 289 LoRA updates; no production adapter; recurrent probes did not establish a reliable gain |
| Oct 3 | [TensorFold tuning](TENSORFOLD.md) | Four-stream INT8 configuration reached 383.08 aggregate tok/s median; long-cache growth needed a fix |
| Oct 5 | [Strata and experimental batched MTP](STRATA-AND-MTP.md) | Eight full histories qualified; experimental batched MTP remained slower despite exact token parity |
| Oct 7 | [KV streaming and four-slot profile](MEMORY-PROFILES.md) | 10.37 GiB startup VRAM saving; reducing eight slots to four saved another 6.15 GiB service RAM |
| Oct 7–15 | [d1 reranking, visual decisions and computer use](D1-DECISIONS.md) | Promising bounded decisions; replay gains did not transfer to the paired live-click trial |
| Oct 14 | [LFM2.5 smoke, accounting and resident pair](LFM25.md) | 14/14 short checks; initial long suite 0/16 exact; corrected rerun incomplete |

## Reading the measurements

- **Decode rate**, **whole-request throughput**, **aggregate batch throughput**, **first-token latency**, and **decision inference time** have different denominators. Do not sort them into one speed leaderboard.
- Context capacity configured in a server is distinct from simultaneous long-context qualification. Active slots differ from submitted requests and open chat windows.
- Engine comparisons also changed checkpoints, quantization, cache precision, vision, scheduling or slot counts. They do not isolate an engine's causal effect or establish equivalent quality.
- Development screens, repeated fixtures and smoke checks are not broad model rankings. Failed, rejected, interrupted and untested work is identified explicitly.
- Host RAM, GPU memory and shared desktop load affect fit. The reports distinguish MiB/GiB from decimal GB and describe sampled memory rather than claiming an instantaneous peak.

## Evidence and reproducibility

[PROVENANCE.md](PROVENANCE.md) explains receipt selection and evidence strength. The
[receipt index](receipts/index.json) binds each source snapshot by SHA-256. Selected
JSON fields retain measured precision; prompts, generated reasoning, credentials,
internal endpoints, personal paths and operational restoration scripts are omitted.
The September research summaries use attributed contemporaneous notebook excerpts
where the original raw run artifacts were not available in this checkout.

Run the offline integrity and reconciliation check from the repository root:

```sh
python experiments/validate.py
```

It verifies the complete archive manifest, local Markdown links, key reported
arithmetic and cross-receipt invariants. It performs no network access or GPU work.
The original release subtree and its checksum manifest remain unchanged.

## Latest recorded operational state

The October 14 notebook records a co-resident LFM worker and Qwen 27B reviewer
with successful readiness canaries; the wider rollout was incomplete. This
documentation task did not recheck live health, change model routes, or complete
that rollout. The October 15 d1 smoke is a separate bounded decision test.

October 13 driver 615 packages were staged, with 610.43.02 still loaded in that
snapshot. Installation/reboot and the proposed Local Inference Lab comparison
were not completed in the reviewed receipt. Its advertised profile is not an
additional measured result here.
