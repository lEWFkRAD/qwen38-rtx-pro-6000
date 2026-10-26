# Strata Lens 0.2.0

October 26, 2026 · Experimental release
Release tag: `strata-lens-v0.2.0`

0.2.0 records the first completed real GPU capture and adds a routing
visualization to the local viewer. The evidence discipline from
0.1.0-preview.1 is unchanged: the capture establishes that the artifacts
are structurally valid and reproducible. It does not establish what any
expert means, and there are no causal or steering claims in this release.

## First real capture (October 25, 2026)

Executed by Codex under a root-owned, operator-gated supervisor while
production inference was stopped, then restored byte-identical.

- Prompt: `numbers_01`, the synthetic counting prompt in
  `corpus/prompts.json`
- Run ID: `phase0a-20261026T004254-5debc649`
- Captured: 126 tokens across 48 layers, top-10 expert routes per token
  and layer, and the final residual state
- Validation: the harness validator passed on the run; all 512 experts
  appear in the routing trace; per token and layer, routing weights sum
  to 1.0 within [0.998, 1.0022]; layer 0 routes across 274 distinct
  experts and layer 47 across 187
- Restoration: production configuration hashes matched the pre-capture
  values, readiness checks passed, and an ordinary inference request
  returned the expected response before the run was declared complete
- Artifact SHA-256 (datasets stay local; `local/` is gitignored and the
  activation data is not published in this repository):
  - manifest.json `488618c63d189782c18e945204dad67684646ef6b525e2c552e1602d101e4031`
  - tokens.txt `78e69e599855cda89b0bbd0a821c1f529c1ef2ad80223f441eb45f58f3db4d62`
  - prefill_routes.bin `b7ba5c6b81349738e9e63bc525e72fd9f364e172c5b7ae59f6d2ddd26decc982`
  - prefill_final_r.bf16 `1fc6f255a8be5a58cbfd1e47291272f23860efee58517b924588637e598d3df2`

The harness changes in this release come from that run: the capture path
was exercised end to end for the first time, and `phase0_harness.py` and
its tests carry the fixes the run required.

## Viewer 0.2.0

`lens_ui.py` gains a read-only `GET /api/viz` endpoint that parses the
routing and residual artifacts under the harness validator's rules and
serves numbers only (no token IDs, no file paths). The UI adds:

- a 48 x 126 layer-by-token routing map (top-1 weight or routing
  entropy per cell)
- per-cell top-10 expert bars for the selected token and layer
- expert-load panels with a 512-expert grid per layer
- a final-residual panel (the capture preserves the final residual
  only; there is no per-layer residual in the artifacts)
- capture playback ("Replay Capture")

Repository test suite at this release: 72 tests, all passing, and
`sha256sum -c SHA256SUMS` verifies every published file.

## Still true from the preview

An expert appearing frequently is an observation, not proof that it has
a particular function. Strata Lens uses the hooks implemented by
[Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata);
see CREDITS.md. The October 22 eight-session throughput results are not
measurements of Strata Lens.
