# Credits and provenance

Strata Lens is an independent downstream experiment. Its capture harness,
validation, tests and documentation build on the work credited below. These
credits do not imply upstream participation in, review of, or endorsement of
this experiment.

## Methodological inspiration

[Lyra C (`lyramakesmusic`)](https://github.com/lyramakesmusic) created
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), the
inspiration for exploring model behavior through steering, retained rollouts
and random-direction controls. Strata Lens adapts that research question to
Strata's residual and MoE routing instrumentation. This initial capture slice
does not reproduce neuron-explorer's per-neuron interventions or its UI.

This contribution does not vendor neuron-explorer code, assets or sweep data.
On 2026-10-25, its public repository root and `pyproject.toml` did not declare
a license. The reference here is conceptual attribution; it is not a claim
that its source is available under this repository's license.

## Engine and instrumentation

[Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata)
created Strata, including the native-prefill residual and expert-routing hooks
used here. Strata is MIT-licensed, with copyright held by Niko1221 and the
Strata contributors. Its license is preserved in the existing
[Strata MIT notice](../../strata-c8/LICENSES/Strata-MIT.txt).

The capture contract targets engine **0.1.46**, source
[`a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e`](https://github.com/Niko1221/Strata/tree/a7bdbfdabc9a3e7c83e3dbc2441d0933569d773e),
from the official [v0.2.1 engine release](https://github.com/Niko1221/Strata/releases/tag/v0.2.1).
The executable SHA-256 is
`e54498e95ee5974c5a2b92b430aa867861baf884d217e7761dd68e2d905a1e2e`.

The harness imports Strata's separately installed `strata_tokenizer.py`;
that module uses `gguf_reader.py`. Neither source file is vendored here.
Their required hashes must be recorded in the operator-supplied, reviewed
input contract; no machine-specific attested contract is published here. The runtime also
depends on separately installed Python and `regex`. Strata's own upstream
acknowledgements, including llama.cpp / ggml, remain available in its
[credits](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md#credits).

## Model, quantization and control vector

| Contribution | Credit and source |
| --- | --- |
| Base model and MTP weights | [Qwen team — Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| GSQ/RCO quantization and original GGUF shards | [ISTA-DASLab — Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09) |
| Package distribution, control-vector extraction and associated measurements | [alesha-pro — Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF](https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF/tree/d4af3be5701549435e0770d625f79dd2def1358f) |
| Abliterated checkpoint from which that direction was extracted | [huihui-ai — Huihui-Qwen3.8-Flash-Next-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated) |

The retained alesha-pro model card identifies the quantized shards as
byte-for-byte copies of ISTA-DASLab revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`. The downloaded package is pinned to
`d4af3be5701549435e0770d625f79dd2def1358f`. Its separate
`Huihui-Qwen3.8-Flash-Next-refusal-direction-r.gguf` file has SHA-256
`59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38`.
The original GSQ-RCO weights are unchanged; the configured projection applies
the extracted direction at runtime.

The retained package license is **Qwen Community License 1.0**. The ISTA-DASLab
model card states that its quantized weights inherit the base model's license;
its tooling has separate repository licenses. No model weights, control
vectors, generated packs, tokenizer assets or CUDA binaries are distributed
with Strata Lens. See the existing [asset and runtime provenance](../../strata-c8/DEPENDENCIES.md).

ISTA-DASLab requests citation of both underlying methods:

- Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtic,
  Maximilian Kleinegger and Dan Alistarh (2026),
  [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556).
- Michael Helcig and Dan Alistarh (2026),
  [Model Compression with Exact Budget Constraints via Riemannian Manifolds](https://arxiv.org/abs/2605.00649).

## Local project work

The repository maintainer supplied project direction and the RTX PRO 6000
Onyx environment. **Onyx Hermes** prepared the initial adaptation scoping,
probe corpus and first offline runner. **Codex** reviewed the execution
assumptions and developed the capture harness, artifact validation, tests,
local read-only viewer and release documentation. These are agent work roles, not additional human
authors or claims of authorship over any upstream component.

The scientific status remains experimental: synthetic tests validate the
harness mechanics; a real GPU capture and causal steering results are separate
validation milestones.
