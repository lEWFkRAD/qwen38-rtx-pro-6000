# Strata Lens — experimental capture harness

Strata Lens inspects Qwen3.8-Flash-Next residual states and expert routing
through Strata's existing native-prefill hooks. This first slice captures one
prompt's final residual and routing trace, establishing measurement before
interpretability or steering experiments.

**Status, October 25, 2026:** offline plans and unit/fake-engine tests pass.
No real GPU capture has succeeded or been qualified for this harness. There
are no published model activations, scientific results, or causal findings.

The idea came from [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).
Onyx Hermes scoped the MoE adaptation and drafted the probe corpus; Codex
developed and reviewed the harness and public packaging. See
[credits](CREDITS.md) and [license scope](LICENSE.md). This is independent
downstream tooling, not an upstream Strata release.

## Try it without a model

From the repository root, using Python 3.12 on Linux:

    python3 research/strata-lens/phase0_harness.py plan --prompt-id numbers_01
    python3 research/strata-lens/phase0_harness.py plan --token-ids '1,2,3'
    python3 -m unittest discover -s research/strata-lens/tests -v
    python3 research/strata-lens/validate.py

These commands do not invoke Strata, use the GPU, or touch a model service.
Plans are marked plan_only=true and input_attestation_verified=false.
Reference token IDs preview numbers_01 without assets; real prompt runs
retokenize against attested tokenizer/model inputs and check the token hash.

The [profile](profile.example.json) and
[input contract](input-contract.example.json) are incomplete templates and
cannot authorize execution. The single synthetic numbers_01 corpus prompt
is an engineering fixture, not reference factual material or an evaluation set.

## Pinned capture contract

- Strata 0.1.46 Linux CUDA 13 executable SHA-256:
  e54498e95ee5974c5a2b92b430aa867861baf884d217e7761dd68e2d905a1e2e.
- BUILD.json: version 0.1.46, source f-021-rel a7bdbfdabc9a.
- Engine source: a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e.
- Target: RTX PRO 6000 Blackwell SM120; tokenizer Python 3.12,
  regex 2026.9.29, pinned Strata tokenizer and gguf_reader.py.

The [dependency record](../../strata-c8/DEPENDENCIES.md) identifies the official
bundle/model assets. Other versions may have different hooks or formats.

The direct command has no generate subcommand. It uses --expert-cache auto,
--prefill 1024, --spec 4, a 1,024-token research context, and one generated
token. Production's 131,072-token context/eight slots are separate choices.
--mmap-experts is excluded following the original native-pack geometry failure.

Sentinel token 248046 is appended. Native prefill consumes all but the last
prompt token, so all real tokens enter prefill and the sentinel enters the
verifier. At most 1,022 real tokens fit this bounded capture.

| Output | Pinned layout | 126 real tokens |
| --- | --- | ---: |
| prefill_final_r.bf16 | Little-endian BF16, N × 10240: four 2,560-wide streams | 2,580,480 bytes |
| prefill_routes.bin | 48 records; int32(layer,pos,N,has_features), then N × 10 uint16 IDs and N × 10 BF16 weights | 242,688 bytes |

STRATA_PREFILL_DUMP_R_ALL requests final residuals; STRATA_DBG_FEAT requests
routing; empty STRATA_DBG_FEAT_LAYERS disables feature payloads. Validation
checks exact sizes, finite/nonzero residual rows, ordered layers 0–47,
position zero, IDs 0–511, ten distinct experts per token, and finite nonnegative
weights with BF16 sums between 0.98 and 1.02. Optional --include-verifier-routing
captures 4,224 bytes of legacy verifier plumbing with zero-placeholder weights.

Inherited engine, GPU and loader overrides are stripped. Fixed controls include
STRATA_PF_FUSED=0, production batch/copy flags (largely dormant here),
STRATA_ARENA_MMAP=1, CUDA_VISIBLE_DEVICES=0, and the pinned library directory.
The existing control vector is applied at strength 1.0, layers 4–44,
project/per-layer; captures describe that model variant.

## Provision a real experiment

Real execution requires an operator-owned profile, its separately reviewed
SHA-256, and a complete input contract whose SHA-256 is recorded in the profile.
Keep private copies in ignored local/ or outside the repo.

1. Independently verify engine/model/tokenizer/pack/MTP/vector assets against
   trustworthy records and a reviewed pack-building process. No automatic
   attestation generator is provided.
2. Adapt the profile's paths and unit/port. Supply exact path, size, mtime_ns
   and sha256 for every required input: build/runtime libraries and manifest,
   serving config, both GGUFs, expert profile, vector, corpus, both tokenizer
   sources, and every PACK/MTP member. Directory membership must be exact.
   The corpus entry is this checkout's absolute corpus/prompts.json path.
   Hashing unknown files alone is not provenance.
3. Review the contract/profile, set profile state to attested, and retain its
   digest separately. Real launch fully rehashes inputs, potentially reading
   over 100 GB and taking minutes. The public version does not retain the
   prototype's weaker size/mtime-only exceptions.
4. Supply a maintenance controller whose owned HOLD inhibits **all** recovery,
   restart and request-triggered startup paths. The profile declares this
   operator responsibility; it does not install a controller. An attended
   owner must stop production, restore it afterward and run a canary.
   A file named HOLD alone provides no isolation.

After provisioning, the command shape is:

    python3 research/strata-lens/phase0_harness.py \
      --profile /operator/path/profile.json \
      --profile-sha256 REVIEWED_64_HEX_DIGEST \
      run-offline --prompt-id numbers_01 \
      --confirm-production-stopped \
      --maintenance-owner EXACT_EXISTING_HOLD_OWNER

The gate requires a matching HOLD owner, successful systemctl show reporting
loaded/inactive/dead, MainPID=0, and successful ss with no listener. It repeats
immediately before launch. Active, activating, failed, unknown, malformed,
command-error and listener states are refused.

The harness never stops, restarts, restores or configures production. The
external controller must account for other GPU processes and prevent a second
server throughout the experiment. Gate checks are observations, not an atomic
system-wide GPU lock.

## Receipts and validation

Runs begin in .partial directories with success=false. Failure retains
diagnostics. Success requires valid artifacts, SHA-256 receipts, original
process-group cleanup, fsync and atomic finalization. Read the manifest rather
than inferring success from the directory name.

    python3 research/strata-lens/phase0_harness.py \
      --profile /operator/path/profile.json \
      --profile-sha256 REVIEWED_64_HEX_DIGEST \
      validate /path/to/completed-run

Offline validation checks recorded input attestation, exact command/environment,
sentinel, lifecycle receipts, binary structure and freshly hashed artifacts.
It needs the same profile/contract identities and recorded paths, but does not
require or rehash model files or prove they remain installed. Fake-engine and
local-prototype schema-2 receipts are rejected by the CLI.

Containment covers the original POSIX process group through timeouts, nonzero
exit, post-launch errors and handled INT/TERM/HUP/QUIT. Repeated signals cannot
interrupt cleanup. Daemonization or setsid can escape; SIGKILL, power loss and
interpreter/kernel failures require external recovery. Only the pinned
single-process engine is supported; this is not cgroup containment.

## Research limits

There are no per-layer residual ladders, serve-path log-probabilities, true
verifier weights, full-corpus results, steering experiments or causal claims.
Routing frequency is observational and confounded by token identity and length.
Next is a reviewed GPU smoke, then replicated matched discovery/held-out prompts
and specified analysis. Residual steering and expert ablation need separate
causal tests.

The synthetic suite covers artifact integrity, signal/process cleanup,
durability, quiescence, reference planning, content drift with preserved stat
metadata, and template/unreviewed profile refusal. It tests software behavior,
not engine-hook fidelity or scientific validity.
