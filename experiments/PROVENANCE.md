# Evidence scope and publication method

Compiled on October 15, 2026 from saved experiment artifacts and the shared agent
notebook. The publication baseline was repository commit
`59f59a86245bc63ff7f74d775b890a0bd9888734`.

## Two evidence levels

1. **Selected local receipts:** machine-readable values extracted from saved JSON,
   with original source SHA-256 and an explicit field selection. These support
   recomputation of the tables without publishing whole operational snapshots.
2. **Notebook records:** exact selected lines from contemporaneous project notes,
   with source document SHA-256 and original section headings. These preserve
   results whose full raw artifacts were not present in the local source set.
   They are attributed records, not independently rerun experiments.

The [index](receipts/index.json) lists both. A source hash identifies the original
private artifact; it does not make an unavailable artifact independently
reproducible. Public receipt hashes in [SHA256SUMS](SHA256SUMS) verify the files
actually included here. The manifest excludes itself.

Selected JSON copies omit source command lines, model responses/reasoning, process
identities, private endpoints, usernames, local absolute paths, service backups,
and unrelated application data. Public model/source revisions remain where
useful. No weights, adapters, private workpapers or service configuration files
are distributed by this addition.

## Conflicting or incomplete records

- Later dated notebook entries supersede earlier restoration/promotion notes.
  Older reports remain history rather than instructions to restart a service.
- The corrected LFM rerun's available local bank progress file records
  1,115.38 seconds and 41 turns. The later closeout notebook records 1,800.9
  seconds and 99 turns. These are distinct snapshots. The archive retains both
  evidence levels and does not manufacture a complete six-job result file.
- The October 11 TensorFold 379-test integration receipt is available locally.
  The subsequent 399-test GEMV/compile result is notebook-reported; CUDA device
  execution was explicitly absent.
- GLM [PR #6](https://github.com/lEWFkRAD/qwen38-rtx-pro-6000/pull/6) was still open
  when checked for this update. It preserves an earlier progress snapshot.
  This archive adds the later failed-quality closeout without merging or
  rewriting that separate review. Installer-hardening PR #1 is also separate.

## Reproduction boundary

The archive supplies measured settings, outcomes, source pins where retained,
selected receipts, and an offline validator. It is not a turnkey launcher for
every historical experiment. Some paths depended on patched runtimes and owned
maintenance sessions. Replaying those operational scripts against today's
machine would not reproduce the original controlled conditions.

No new GPU, quality, training or live-service test was performed while preparing
this update. No external model/vendor performance claim is substituted for our
own measurements.
