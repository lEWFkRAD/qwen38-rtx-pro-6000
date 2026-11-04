# Credits and contribution boundaries

This is independent downstream measurement work. No upstream participation, review or endorsement is implied.

- **Niko1221 and the Strata contributors** supplied the inference engine, speculative verification, batching, tokenizer/template, API and vision integration. This experiment uses official native 0.1.56 with retained API wrapper 0.1.42; those are distinct dependencies, not an API-wrapper upgrade. See [DEPENDENCIES.md](DEPENDENCIES.md) and the retained [Strata MIT notice](../strata-0156-tuning/LICENSES/Strata-MIT.txt).
- **The Qwen team, ISTA-DASLab, alesha-pro and huihui-ai** supplied the model, quantization/packaging and separate control-vector contributions described in the [prior asset documentation](../strata-0156-tuning/DEPENDENCIES.md). The custom IQ3_S/vector configuration is retained. This work does not claim authorship of those assets or transfer their model/output rights.
- **Python, Jinja2, regex, NVIDIA CUDA and the other runtime/tooling contributors** supplied dependencies. No vendor executable, CUDA library or weight file is republished here.
- **The repository maintainer** supplied the objective, hardware, deployment authorization and publication direction. **Codex sessions and collaborating agents** wrote the bounded live collectors, independent synthetic fixtures/reference checks, CPU protocol tests, collector-correction records, numeric analysis and privacy-aware packaging. These are work roles, not extra human authors or ownership claims over upstream algorithms.

The new work is measurement and collector validation. It includes no new inference algorithm, LSA integration, model fine-tuning or claim that a shorter output is faster decoding. A collector correlation fix changes how evidence is classified; it does not make the engine faster or rewrite the failed earlier attempt.

The prompts and exact checks are fictional/synthetic. Raw client conversations, documents, credentials and actual serving configurations are excluded. Public reports omit raw free-text model output while retaining counts, hashes and the observed small synthetic answers needed to understand quality failures. Original source and normalized public hashes are recorded separately.

Original downstream benchmark source/tests, synthetic inputs and documentation follow the [earlier Apache source notice](../strata-0156-tuning/LICENSE.md). Upstream Strata code retains its MIT terms, and model-derived answers retain applicable model/output terms. No general quality-equivalence, universal throughput or simultaneous eight-deep generation qualification follows from a finite speed screen.
