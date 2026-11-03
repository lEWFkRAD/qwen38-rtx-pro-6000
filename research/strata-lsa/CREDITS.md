# Credits and authorship boundaries

This is independent downstream research and engineering. Publication does not imply upstream participation in, review of, or endorsement of the experiment.

- **Prefrontal AI and the Latent Sparse Attention authors/contributors** created [Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention), its learned projection/indexing design, Ward clustering and selection/reference implementations, training support and research material. The initial source is pinned to `906be195c1052aaa74f95852ad1917ce1926ba64`. Its Apache 2.0 license and [third-party notices](LICENSES/LSA-THIRD-PARTY-NOTICES.md) are retained.
- **Niko1221 and the Strata contributors** created [Strata](https://github.com/Niko1221/Strata), including the inference engine, INT8 cache, attention readers, speculative execution, conversation handling and vision integration adapted here. The engine base is `1aff6d14117b88171fcfcec8f2b55fdc47f2b165`. Its [MIT notice](LICENSES/Strata-MIT.txt) remains applicable to retained and modified Strata code.
- **The Qwen team** supplied [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and its MTP model weights. **ISTA-DASLab** supplied the GSQ/RCO methods and original quantized GGUF shards. **alesha-pro** packaged those unchanged quantized shards and extracted the separate control vector; **huihui-ai** supplied the abliterated checkpoint used to derive that direction. These are distinct model, quantization and packaging contributions. Their exact asset revisions are in [DEPENDENCIES.md](DEPENDENCIES.md).
- **The llama.cpp/ggml, PyTorch, Triton, NVIDIA CUDA/CCCL and other dependency contributors** supplied the build/runtime/reference infrastructure. The LSA upstream third-party notice separately credits the MIT-licensed NSA kernel adaptation from flash-linear-attention and records the terms of its optional datasets. Retain the original notices inside source archives.
- **The repository maintainer** supplied project direction, the workstation and the publication. **Codex sessions and collaborating agents** prepared the isolated research environment, retained failure diagnoses, added the symmetric CPU Ward reference repair, captured actual Flash-Next attention states, trained/evaluated the pilot maps, implemented the native adapter and live cache, wired the three inference paths, added lifecycle/replay tests, ran the bounded comparisons and assembled the public evidence. These are agent work roles, not additional human authors or ownership claims over upstream components.

The native CUDA Ward implementation and index adapter follow the pinned upstream LSA contract. They do not turn the upstream method into a new downstream scientific discovery. The added work is the Flash-Next/Strata-specific capture, map preparation, native integration, state handling, testing and measured outcome.

The technical/code portions of the pilot corpus retain the attribution and licenses of their pinned LSA or Strata sources. The operations/narrative documents and later whole-model canaries are deterministic fictional or synthetic material authored for this experiment. No staff messages, private client documents or unrelated workpapers were used.

For the underlying quantization methods, the retained source cards request citation of:

- Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtic, Maximilian Kleinegger and Dan Alistarh (2026), [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556).
- Michael Helcig and Dan Alistarh (2026), [Model Compression with Exact Budget Constraints via Riemannian Manifolds](https://arxiv.org/abs/2605.00649).

Measured results belong to this bounded hardware/configuration/fixture set. There is no claim of general quality improvement, extended model context, lower memory use or production speed superiority. The tested candidate was not promoted.
