# Native Strata LSA experiment — November 3, 2026

An optional learned sparse-attention selector is integrated into the native
Strata CUDA engine for Qwen3.8-Flash-Next on one RTX PRO 6000 Blackwell 96 GB.
The implementation and bounded compatibility tests are complete. **It was not
promoted to production: the measured eight-session result was slower.**

In plain English, this gives us a working alternative for choosing which older
parts of a conversation the model attends to. It creates a basis for further
attention research. It does **not** enlarge the context window: both engines
retain 131,072 tokens per slot. Tools, images, reasoning and eight conversation
slots were existing capabilities that this experiment preserved.

| Recorded comparison | Control | LSA candidate |
| --- | ---: | ---: |
| Configured context per slot | 131,072 | 131,072 |
| Near-capacity retrieval prompt | 125,076 tokens, passed | 125,076 tokens, passed |
| Near-capacity request wall time | 16.91 s | 17.78 s |
| Eight-slot decode smoke, four windows | 414.6 rows/s | 296.5 rows/s |
| Additional projected-key state | — | about 4 GiB |

These are finite smoke measurements, not a general benchmark. One arithmetic
case failed in both engines: the correct result is 208 and both returned 188.
The failure is preserved. No new quality failure appeared in these fixtures;
general quality equivalence has not been established.

## Code, evidence and notes

- [Final integration results and decision](native/INTEGRATION-RESULTS.md)
- [Implementation, build command and dated handoff](native/README.md)
- [Integration source patch](native/native-lsa-integration.patch) and
  [changed native source files](native/strata-src/)
- [Native tests and all retained run receipts](native/)
- [Earlier preparation, synthetic GPU experiments and failures](preparation/README.md)
- [Historical SHATNER handoff notes](preparation/shatner-handoff/README.md) and [checkpoint receipt](preparation/shatner-handoff/flashnext-maps-README.md)
- [Activation capture, map training and calibration](preparation/flashnext-maps/README.md)
- [Pilot learned/PCA maps, temperatures, twelve training checkpoints and report](preparation/flashnext-maps/maps/pilot-v1/)
- [Whole-document corpus and provenance](preparation/flashnext-maps/corpus/manifest.json)
- [Plain-English explanation transcript](explanation/plain-english-transcript.txt)
- [Source provenance](PROVENANCE.md), [dependencies](DEPENDENCIES.md),
  [credits](CREDITS.md) and [license scope](LICENSE.md)

The native engine source commit is `876307a9ded78bbca941189e63230b33cfdc6e2a`,
on capture-only base `3f83801ea1862652806fc6ebddab7f6cf57f3af6`. Full source,
the exact tested decoder, raw captures and both prepared-data versions are in
the [research release downloads](https://github.com/lEWFkRAD/qwen38-rtx-pro-6000/releases/tag/strata-lsa-2026-11-03).
The repository contains readable source changes, scripts, maps, notes and
receipts; large tensor evidence is kept out of Git history.

## What was tested

The native loader checks all twelve 384×64 map pairs and rejects incompatible
inputs. CUDA Ward closure matches the pinned reference in strict tests.
The bridge tests use real captures from all twelve layers. The integrated
cache test records 1,315 checks, including accepted/rejected speculative
tokens, prefix copies, rebuilds and interleaved slots. Whole-engine checks
cover near-capacity retrieval, prefix reuse, eight slots and cancellation.
The retained API wrapper passed text, forced tool arguments, reasoning and a
fresh image. With LSA disabled, six serial fixtures returned the original
control's exact output token IDs.

The root CMake option defaults off. A feature-enabled build still uses the
original selector unless `STRATA_LSA_MAPS` is supplied. The supported research
configuration is CUDA, a single GPU, resident ordinary INT8 KV and context at
most 131,072. The pilot calibration test set has been consumed; further
tuning needs a new independent held-out set.

## Download and verify

[RELEASE-ASSETS.json](RELEASE-ASSETS.json) records byte counts, SHA-256 hashes
and download URLs. [PUBLICATION-MANIFEST.json](PUBLICATION-MANIFEST.json)
connects published files to their original source hashes and records text
redactions. Tensor, learned-map and executable bytes are unchanged.

The three large `.tar.zst` datasets extract from this directory:

```sh
tar --zstd -xf lsa-captures-20261103.tar.zst
tar --zstd -xf lsa-prepared-original-20261103.tar.zst
tar --zstd -xf lsa-prepared-compatible-20261103.tar.zst
tar -xzf strata-lsa-source-876307a.tar.gz
tar -xzf lsa-original.tar.gz -C preparation
tar -xzf lsa-research.tar.gz -C preparation
python3 validate.py
```

The final source archive supplies `native/strata-src/`. The two LSA source
archives supply `preparation/upstream/` and `preparation/upstream-research/`.
The original integrated checkpoint is a separate, unchanged archive with its
own relative layout and checksums; extract it into a separate directory.
Historical scripts retain experiment-specific assumptions. Public text copies
replace operator paths and the live maintenance owner with placeholders;
review those assumptions before intentional reproduction. No upload step
starts a model or changes the saved deployment.

`python3 validate.py` checks the public package's hashes and the recorded
decision. It does not rerun GPU tests or qualify a deployment. Production was
intentionally paused at the final handoff, and the candidate was not promoted.
