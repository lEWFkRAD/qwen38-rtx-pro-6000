# Stock Strata 0.1.56 promotion — completed and independently verified

The operator authorized promotion of the measured fused_prefill serving profile.
Production resumed through its managed readiness path and passed the independent
read-only live audit. This is a separate deployment record. The completed
benchmark results and its historical final-state/selection receipts remain
unchanged.

Selected components: official Strata engine 0.1.56, build 06698c674673+cuda13.0,
source 06698c674673008395ba6e971580538d6b1ea6e0, with matched official vision and
CUDA 13 libraries. Decoder SHA256:
21cb8dd0c744c84176a019ef3cfebd27997968f211b197429df897cc6af92788.
The Python/API wrapper remains the retained 0.1.42 runtime. No model weights were
trained, no LSA runtime/maps are enabled, and no vendor model/runtime files are
redistributed with this note.

Verified retained policy: eight 131072-token slots, ordinary fully resident INT8
KV, all 24576 experts GPU resident, the same IQ3_S/model/MTP assets and control
vector 1.0 at layers 4–44/project/per-layer, prefill 32768 and decode share 0.75.
The twelve environment settings are recorded in VALIDATION-PROFILE.json. Shared
sampling/reasoning/output defaults and existing API compatibility/recovery are
retained. Sunshine is to remain stopped, with its autostart configuration intact.

Deployment outcome: completed; production healthy and normally qualified.
Independent contract/binary/library identity audit: PASSED. Installed component
hashes, the retained wrapper and exact approved decoder arguments, all twelve
effective settings, matching loaded CUDA/cuBLAS libraries, and the managed
process-bound readiness receipt were verified. No private gate override was used.
Live readiness/capacity/vision confirmation: PASSED, including eight slots,
131072 context, all 24576 experts, INT8/full residency, MTP 6/verifier 8/lookup 3
and matched image input. Exact qualified native arguments were retained.
Sunshine confirmation: inactive, MainPID 0; its enabled state remains unchanged.
Rollback: none needed or performed. The previous engine and protected configuration backup remain available.

The real production API canary PASSED all six cases: unauthenticated
status rejected with HTTP 401, exact text marker, forced tool-call JSON/count 7,
high-reasoning 17×19=323, a fresh green-square image, and prefix reuse (3937 cached
tokens on a 3944-token prompt). Returned tool calls were inspected, never executed.
The process identity and normal qualified gate were preserved. This is a bounded
compatibility check, not broad model-quality evaluation.

Sanitized [live canary evidence](promotion/live-canary.json) completed
2026-11-04T00:14:55.932942Z. The independent [audit summary](promotion/audit-summary.json)
completed 2026-11-04T00:16:52.359068Z.
These UTC times fall on the operator's November 3 local promotion date.

The independent audit sent only metadata GET requests: no extra completions,
tools or service changes. This public note contains no credentials, actual
serving configurations or private process/network identity. Benchmark gains are
finite same-day comparisons, not proof of general model quality or a promise
that every production request improves.

The deployment and canary reference code is in [promotion/](promotion/). These
node-specific copies contain portable redactions and depend on the retained
wrapper/recovery helpers; adapt role paths and authoritative ownership guards
for another machine. The original benchmark snapshot manifest is historical;
the current package hashes are in `SHA256SUMS.txt`.
