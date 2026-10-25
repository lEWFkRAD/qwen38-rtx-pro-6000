# Strata Lens 0.1.0-preview.1

October 25, 2026 · Experimental prerelease  
Release tag: `strata-lens-v0.1.0-preview.1`

Strata Lens is a capture harness and local run viewer for examining
Qwen3.8-Flash-Next through Strata's native-prefill instrumentation. This
preview packages the first measurement step: capture a prompt's final
residual states and expert-routing trace, preserve their provenance, and
follow the run in a live, read-only browser UI.

The release is ready for offline planning, synthetic testing and local
inspection of run metadata. A completed real GPU capture is still a separate
milestone. There are no published activation datasets, causal steering
results or Lens throughput claims in this preview.

## Why Strata Lens

The broader research question is whether we can connect a model's internal
activity to useful, testable changes in its behavior. For this
mixture-of-experts model, that means looking at residual representations alongside the
experts selected for each token. An expert appearing frequently is an
observation, not proof that it has a particular function.

[Lyra C's neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)
inspired the direction. Strata Lens starts with a narrower engineering
problem: make a capture reproducible and auditable before adding analysis or
interventions. It uses the hooks already implemented by
[Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata);
it does not recreate neuron-explorer's per-neuron interventions or UI.

## What ships in this preview

The [capture harness](phase0_harness.py) combines a plan-only mode, an
operator-gated offline runner and a standalone artifact validator. It ships
with incomplete profile and input-contract templates, a small synthetic
counting prompt, reference token IDs, fake-engine tests and integrity
receipts. The templates cannot authorize a model launch.

The [live viewer](lens_ui.py) adds a loopback-only web server and responsive
dashboard without third-party web dependencies. It polls run manifests and
artifact metadata every two seconds, provides empty and disconnected states,
and has an explicitly labeled synthetic demo. It is an observer of the
capture workflow, not a model-control interface.

The publication also keeps three different kinds of evidence separate:

| Surface | What it establishes |
| --- | --- |
| Strata Lens preview | Offline capture mechanics, planning, validation and a local metadata viewer; no completed real GPU capture |
| [October 25 saved configuration](../../strata-c8/CURRENT-CONFIG-2026-10-25.md) | Sanitized settings observed that day, including the unresolved readiness state |
| [October 22 qualified recipe](../../strata-c8/README.md) | Historical eight-session benchmark evidence under that recipe's exact settings |

The earlier eight-session throughput results are not measurements of Strata
Lens or of the October 25 settings.

## Start the UI without a model

Use Python 3.12 on Linux. From the repository root:

```bash
python3 research/strata-lens/lens_ui.py --demo
```

Open **http://127.0.0.1:8765**. The demo uses synthetic run data so you can
explore the interface without model assets, CUDA or a running engine. Its
labels remain distinct from real capture data.

To follow local capture output instead:

```bash
python3 research/strata-lens/lens_ui.py --runs-dir /path/to/runs
```

With no options, the server watches the ignored
`research/strata-lens/runs/` directory. An empty view means there are no
discovered runs; starting the viewer does not create one. Stop it with Ctrl+C.

“Live” refers to refreshing the filesystem view, not streaming model
activations. Run status and success are recorded claims from manifests. The
dashboard does not independently rehash captures, establish engine-hook
fidelity or certify a scientific result. Use the CLI validator for artifact
validation.

Keep the server on loopback. This preview is not an authenticated shared
service, and it should not be exposed through a public tunnel or reverse
proxy. The browser cannot launch a model, manage a service, edit a profile or
write capture files.

## Plan a capture without a GPU

The supplied reference tokens let you inspect a command plan immediately:

```bash
python3 research/strata-lens/phase0_harness.py plan --prompt-id numbers_01
python3 research/strata-lens/phase0_harness.py plan --token-ids '1,2,3'
```

These are plan-only operations. They do not start Strata, reserve GPU memory
or change a serving process. Plans explicitly record
`input_attestation_verified=false`. The included `numbers_01` prompt is a
synthetic engineering fixture, not a factual benchmark or research corpus.

## Capture method and architecture

The operator supplies a reviewed profile and an input contract identifying
the executable, model, expert packs, MTP assets, control vector, tokenizer
sources and relevant configuration. The runner checks those identities,
checks the externally owned maintenance hold, and invokes the pinned Strata
binary directly. It sanitizes inherited engine and loader overrides and
records the actual command and environment.

The contract targets Strata engine **0.1.46**, source
`a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e`, and the RTX PRO 6000 Blackwell
SM120 target. Exact binary and model provenance are in
[CREDITS.md](CREDITS.md) and the
[dependency record](../../strata-c8/DEPENDENCIES.md). Compatibility with other
engine versions is not assumed.

The research command uses a 1,024-token context and prefill cap, requests one
generated token, and appends sentinel token `248046`. Native prefill consumes
all but the last prompt token; the sentinel lets every real prompt token
enter prefill while the sentinel enters the verifier. These are diagnostic
settings, distinct from the eight-slot, 131,072-token serving configuration.

For the included prompt's 126 real tokens, the two required outputs are:

| Artifact | Meaning | Expected size |
| --- | --- | ---: |
| `prefill_final_r.bf16` | Final residual rows, four 2,560-wide streams per token | 2,580,480 bytes |
| `prefill_routes.bin` | Routing records across 48 layers, ten selected experts per token | 242,688 bytes |

The validator checks exact layouts, finite and nonzero residual rows, layer
ordering, expert IDs and uniqueness, and finite nonnegative routing weights
whose BF16 sums fall within the stated tolerance. The configured control
vector remains applied; a future capture would describe that model variant,
not an unmodified base-model control.

Runs begin in `.partial` directories with `success=false`. Completion
requires valid artifacts, SHA-256 receipts, process-group cleanup, durable
writes and atomic finalization. Failures retain diagnostics. The viewer reads
the resulting manifests and metadata; it does not participate in execution.

## Provisioning a real run

Real execution requires the separately obtained assets, an independently
reviewed input contract and profile, and an attended maintenance procedure.
Follow the complete [capture guide](README.md#provision-a-real-experiment)
before invoking `run-offline`.

The public runner fully hashes its required inputs, potentially reading over
100 GB. Hashing a set of unknown files is not itself proof of provenance:
the operator must establish their origin and the pack-building process first.
The profile's digest must be reviewed and retained separately from the
profile supplied to the command.

An external controller must own a maintenance hold that prevents all restart
and request-triggered startup paths. The harness requires production to be
loaded, inactive, dead and without a listener, and repeats that check just
before launch. A file named `HOLD` alone is not isolation. The harness does
not stop or restore production, and its state checks are not an atomic GPU
lock. The operator remains responsible for other GPU users, restoration and
the post-run canary.

Process containment covers the original POSIX process group, including
timeouts, launch failures and handled INT, TERM, HUP and QUIT signals.
Daemonization can escape that group; SIGKILL and machine failures need
external recovery. This is not cgroup containment.

## Validation and reproducibility

The release passed **63 Lens tests** (38 capture-harness tests and 25 viewer
tests), plus the existing nine benchmark tests and all three publication
validators. The viewer suite includes actual loopback HTTP requests, privacy
projection, malformed manifests, missing artifacts, symlinks and bounded
directory scans. The tests need permission to bind a temporary loopback port;
they make no external network requests and use no GPU.

Browser interaction checks covered demo and empty live views, pause/resume,
manual refresh, disconnection labeling and a narrow viewport without horizontal
overflow. Screenshot capture was unavailable in the verification environment;
pixel-level visual review is not claimed.

Run the offline checks from the repository root:

```bash
python3 -m unittest discover -s research/strata-lens/tests -v
python3 research/strata-lens/validate.py
python3 -m unittest discover -s strata-c8/tests -v
python3 strata-c8/validate.py
python3 experiments/validate.py
```

The harness tests cover reference planning, refusal of incomplete or
unreviewed profiles, input content drift, artifact integrity, quiescence
checks, durability and signal/process cleanup. The viewer has separate tests
for its local, read-only inspection behavior. Synthetic and fake-engine tests
exercise software behavior; they do not demonstrate fidelity of a live
engine's hooks.

A completed real run must also pass the CLI `validate` command with the same
reviewed profile and contract identities. That command freshly hashes the
capture artifacts and checks the recorded command, environment, lifecycle
and input attestation. The viewer's recorded-success label is not a substitute
for this check. See [receipts and validation](README.md#receipts-and-validation)
for the command and its exact scope.

## Current configuration and operational status

The [October 25 snapshot](../../strata-c8/CURRENT-CONFIG-2026-10-25.md) changes
two settings relative to the sanitized October 22 example: prefill rises
from 16,384 to 32,768, and `fit_max_tokens=true` enables output-budget fitting.
The engine, eight resident sessions, 131,072-token context and full INT8 KV
policy remain unchanged. No speed improvement is attributed to those changes.

At the recorded October 25 check, the service was `activating/start-post`
with 14 automatic restarts. Earlier checks showed expert-cache capacity below
the required 24,576 experts while desktop GPU clients occupied additional
memory. The snapshot preserves those observations; it is not a new healthy
deployment qualification. Publishing this release does not alter that state
or establish recovery.

## Credits and license scope

The work depends on distinct upstream contributions:

- **Lyra C (`lyramakesmusic`)** created neuron-explorer, the methodological
  inspiration. No source, assets or sweep data from that project are vendored.
- **Niko1221 and the Strata contributors** created the inference engine and
  native-prefill residual and routing hooks.
- **The Qwen team** supplied the base model and MTP weights.
- **ISTA-DASLab** supplied the GSQ/RCO methods and original quantized GGUF
  shards. The full research-author citations are preserved in the credits.
- **alesha-pro** supplied the Strata package distribution, control-vector
  extraction and associated measurements; **huihui-ai** supplied the
  abliterated checkpoint from which that direction was extracted.
- **The repository maintainer** supplied project direction and the Onyx
  RTX PRO 6000 environment. **Onyx Hermes** prepared the initial adaptation,
  probe corpus and offline runner; **Codex** developed and reviewed the public
  harness, validation, tests, local viewer and release packaging.

[CREDITS.md](CREDITS.md) records exact revisions, authorship boundaries and
source links. [LICENSE.md](LICENSE.md) explains the downstream Apache 2.0
scope and retained upstream notices. Model licenses remain separate. This
release distributes no model weights, CUDA engine binaries, control vectors,
packs or tokenizer assets, and does not imply upstream endorsement.

## Next milestones

1. Complete an attended, provenance-checked GPU smoke capture, validate its
   artifacts, restore serving and pass the operational canary.
2. Replicate captures with matched discovery and held-out prompts, recording
   tokenization and length controls before interpreting routing differences.
3. Add analysis views for validated captures, with explicit distinctions
   between measured values, summaries and hypotheses.
4. Specify separate causal tests before attempting residual steering or
   expert ablation, including controls and held-out behavioral checks.

Per-layer residual ladders, serve-path log probabilities, meaningful verifier
routing weights and steering controls are outside this preview. The next
release should advance those claims only when the corresponding measurements
and tests exist.
