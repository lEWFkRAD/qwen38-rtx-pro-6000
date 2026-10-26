# Strata Lens Phase 2 — design notes

Status: prep, 2026-10-26. Companion to the Phase 2-5 plan. This note
separates what is built, what is designed, and what is blocked on engine
instrumentation questions, so nobody has to guess which is which.

## Built in this prep

- **Corpus v2** (`corpus/prompts.json`): 13 prompts in designed classes
  (numbers x2, arithmetic near-pair, digit-free prose x2 as the control
  class, code near-pair, recall, reasoning, repetition, and the minimal
  pair pair_a/pair_b differing by exactly one content word). numbers_01
  is byte-identical to the Phase 0 original; its pinned reference hash
  still matches.
- **Run comparison** (`compare_runs.py` + `tests/test_compare.py`):
  validates both runs with the harness's own validators, then compares
  token streams, per-layer top-1 agreement, top-10 Jaccard overlap,
  weight deltas, and final-residual cosine/L2. Verdicts IDENTICAL /
  EQUIVALENT (within one BF16 ulp) / DIVERGENT / INCOMPARABLE, exit
  codes 0/1/2. Proven against the real Phase 1 capture: a copy compared
  to itself returns IDENTICAL, and a single expert id flipped in layer
  5 is detected and located (top-1 agreement 0.9921, worst layer 5).

## The repeatability gate

Phase 3 analysis does not start until the same prompt, captured twice
under the same attested profile, compares IDENTICAL or EQUIVALENT with
this tool. If a rerun diverges, the divergence is a finding about the
capture path (nondeterminism in the engine, the profile, or the
environment), and it gets understood before any expert statistics are
trusted.

## Artifact format v2 (designed, not yet buildable end to end)

Format v1, as validated today: `prefill_routes.bin` holds 48 records of
[int32 header (layer, 0, rows, 0) + rows x 10 uint16 expert ids +
rows x 10 BF16 weights]; `prefill_final_r.bf16` holds one final
residual row per token (10,240 BF16). The run is a one-shot prefill
(`--max-new 1`); the only routing-related engine flag the harness uses
is `--dump-routing` for the verifier sentinel, whose weights are zero
placeholders and which is explicitly non-research plumbing.

Two v2 additions are designed here. Both are honest about their
dependency: the harness can validate and version them today, but the
engine's native prefill path has to emit them, and no current flag
does.

### 1. Per-layer residuals (final position)

New optional artifact `prefill_layer_r.bf16`: for each of the 48
layers, the residual stream at the final token position after that
layer (48 x 10,240 BF16, about 0.98 MB per run at any prompt length).
This is the default tier: it shows how the representation develops
across depth without the storage of the full grid. The full grid
(all positions x all layers) is 48 x rows x 20,480 bytes, about
124 MB for a 126-token prompt, and stays an explicit opt-in if it is
ever built.

Open engine question: does the native prefill instrumentation expose
per-layer residual buffers at all, the way it exposes the final one?
If the answer is no, Phase 3 works with final residuals plus full
routing, which is already a substantial analysis surface, and this
item closes as not available rather than being faked.

**Spike result (2026-10-26, from the upstream source of the same
engine line, src/program/generate.cpp): YES.** The one-shot program
has `--dump-layers PATH`: session_loop fills a per-position "ladder"
of (n_layers + 1) x hc x n_embd f32 (layer 0 first, the +1 is the
input stage), and the position loop writes it per position, about
2.0 MB per position at this model's shape. Two caveats, both
recorded rather than smoothed over: (a) the flag disables the fused
gather/reduce path and the token-graph capture paths, so a
dump-layers run is a different execution profile; comparison sets
must stay homogeneous, never mixing dump-layers runs with plain
runs; (b) the v2 artifact should convert f32 to BF16 at the harness
boundary to match v1 conventions, which the format gains a field
to declare. Presence of the flag in the pinned 0.1.46 binary (a
later build of this line) is to be confirmed at provisioning time
by the harness's existing pin checks.

### 2. Decode-side routing

Today there is no decode artifact: the one-shot run generates at most
one token, and routing records cover prefill only. A decode capture
needs two changes, in order: (a) a run mode that generates N tokens,
which changes the run from a pure prefill probe into a generation and
therefore needs its own profile/attestation review, and (b) engine
emission of per-step top-10 routing in a `decode_routes.bin` that
mirrors the prefill record layout per generated step.

Open engine question: does the decode path make the same routing
decisions observable anywhere? Until both answers exist, decode
routing is designed here and built nowhere.

**Spike result (2026-10-26, same source): PLAUSIBLE, UNPROVEN.**
`--dump-routing` in the one-shot program records true ids and
weights per layer per position (the zero-weight records in
verifier_routing.bin come from the separate serve-path trace, not
from this flag). The position loop that writes the final residual
and the layer ladder processes generated positions as well as
prefill positions, so decode positions flow through the same
machinery; what source alone cannot prove is whether the
speculative/MTP decode dispatch invokes the same routing callback
for every generated token. The decider is empirical and cheap
inside a provisioned capture: a `--max-new 4` run whose routing
record count is checked against (prompt + generated) positions.
Until that test runs, decode routing stays unclaimed.

### Versioning rules

- A v2 manifest carries `"artifact_format": 2` and lists which
  optional artifacts a run includes; v1 runs remain valid and the
  validators treat the new artifacts as opt-in, the same pattern as
  the verifier routing artifact today.
- `compare_runs.py` compares v1 artifacts between any two runs; v2
  artifacts join the comparison when both runs carry them.

## What Phase 2 still needs from outside this prep

1. Jeff's approval and an attended window for each capture (the
   repeatability pair for numbers_01 first, then the corpus set).
2. ~~An engine-instrumentation answer on the two open questions
   above.~~ Answered 2026-10-26 (see the spike results inline):
   per-layer residuals are available via `--dump-layers` with a
   changed-execution-profile caveat; decode routing is plausible
   and gets its empirical proof inside the first provisioned
   Phase 2 capture. Format v2 can ship in Phase 2.
